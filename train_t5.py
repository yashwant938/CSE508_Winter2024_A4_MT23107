"""Local T5 training on the original 6,000 reviews, selecting on validation only."""
import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
from pathlib import Path
import time

from comparison_data import (load_protocol, seq2seq_collate, source_hashes,
                             t5_example, visible_review)
from evaluate import load_original_dataset
from train import runtime_info, seed_everything, sha256_file
from training_data import write_json


def validation_loss(model, loader, device, amp):
    import torch
    model.eval()
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Encoder-decoder labels are NOT manually shifted, unlike causal LM.
            count = int((batch["labels"] != -100).sum())
            with amp():
                loss = model(**batch).loss
            total += float(loss) * count
            tokens += count
    return total / tokens


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--gpt2-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data", default="preprocessed_data.pkl")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        p.error("Use a new output directory; existing training runs are preserved")
    config, manifest, freeze = load_protocol(args.protocol)
    seed_everything(config["seed"])
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
    t5 = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"], local_files_only=True)
    gpt = AutoTokenizer.from_pretrained(str(Path(args.gpt2_run) / "checkpoint"), local_files_only=True)
    frame = load_original_dataset(args.data)
    rows, stats = {}, {}
    for split in ("train", "validation"):
        selected = frame.iloc[[r["row"] for r in manifest["splits"][split]]]
        rows[split] = [t5_example(t5, visible_review(gpt, review, config["gpt2_input_tokens"]), summary, config)
                       for summary, review in selected[["Summary", "Text"]].itertuples(index=False, name=None)]
        stats[split] = {"examples": len(selected), "max_source_tokens": max(len(r["input_ids"]) for r in rows[split]),
                        "target_tokens": sum(len(r["labels"]) for r in rows[split]),
                        "truncated_targets": sum(len(t5.encode(s.strip(), add_special_tokens=False)) >= config["target_tokens"] for s in selected.Summary)}
    collator = partial(seq2seq_collate, pad_token_id=t5.pad_token_id)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(rows["train"], batch_size=config["batch_size"], shuffle=True, generator=generator, collate_fn=collator)
    val_loader = DataLoader(rows["validation"], batch_size=config["evaluation_batch_size"], collate_fn=collator)
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"], revision=config["revision"],
                                                 local_files_only=True, use_safetensors=True).to(args.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=config["learning_rate"])
    steps = math.ceil(len(train_loader) / config["gradient_accumulation"]) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * config["warmup_fraction"]), steps)
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable CUDA required for this GPU protocol")
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.device == "cuda" else nullcontext
    source_names = ["train_t5.py", "comparison_data.py", "train.py", "evaluate.py", "training_data.py"]
    write_json(output / "environment.json", {**runtime_info(), "protocol": freeze,
        "comparison_code_sha256": source_hashes(source_names), "parameter_count": sum(p.numel() for p in model.parameters()),
        "precision": "BF16 autocast, float32 parameters" if args.device == "cuda" else "float32"})
    write_json(output / "config.json", config)
    write_json(output / "token_statistics.json", stats)
    for name in source_names:
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(name).read_bytes())
    started = time.perf_counter()
    initial = validation_loss(model, val_loader, args.device, amp)
    print(json.dumps({"event": "initial_validation", "nll": initial, "token_statistics": stats}), flush=True)
    history, best, best_epoch, step = [], float("inf"), None, 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, tokens, group_tokens = 0.0, 0, 0
        for index, batch in enumerate(train_loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            count = int((batch["labels"] != -100).sum())
            with amp():
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            (loss * count).backward()
            total += float(loss.detach()) * count
            tokens += count
            group_tokens += count
            if (index + 1) % config["gradient_accumulation"] == 0 or index + 1 == len(train_loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_tokens)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_tokens = 0
                step += 1
                if step % 25 == 0:
                    print(json.dumps({"event": "progress", "epoch": epoch, "step": step, "total_steps": steps,
                                      "nll": total / tokens, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
        val = validation_loss(model, val_loader, args.device, amp)
        if not math.isfinite(val):
            raise RuntimeError("Nonfinite validation loss")
        selected = val < best
        if selected:
            best, best_epoch = val, epoch
            model.config.use_cache = True
            model.save_pretrained(output / "checkpoint", safe_serialization=True)
            t5.save_pretrained(output / "checkpoint")
            model.config.use_cache = False
        history.append({"epoch": epoch, "optimizer_steps": step, "train_target_token_nll": total / tokens,
                        "validation_target_token_nll": val, "selected": selected,
                        "elapsed_seconds": round(time.perf_counter() - started, 3)})
        write_json(output / "training_history.json", {"initial_validation_target_token_nll": initial,
            "epochs": history, "selected_epoch": best_epoch, "selected_validation_target_token_nll": best})
        print(json.dumps(history[-1]), flush=True)
    write_json(output / "checkpoint_sha256.json", {p.name: sha256_file(p) for p in sorted((output / "checkpoint").iterdir()) if p.is_file()})
    write_json(output / "training_complete.json", {"optimizer_steps": step, "selected_epoch": best_epoch,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None})


if __name__ == "__main__":
    main()
