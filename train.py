"""Reproducible GPT-2 fine-tuning; test references are never used for selection."""
import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from evaluate import DATA_SHA256, load_original_dataset
from training_data import collate, make_manifest, training_example, write_json


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def runtime_info():
    import numpy, pandas, torch, transformers
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": numpy.__version__, "pandas": pandas.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "git_commit": commit,
            "code_sha256": {name: sha256_file(Path(__file__).with_name(name))
                            for name in ("train.py", "training_data.py", "evaluate_checkpoint.py", "evaluate.py")}}


def validate_config(config):
    positive = ("train_examples", "validation_examples", "test_examples", "input_tokens", "target_tokens",
                "epochs", "batch_size", "gradient_accumulation", "evaluation_batch_size", "bootstrap_samples")
    if any(not isinstance(config[k], int) or config[k] < 1 for k in positive):
        raise ValueError("Counts and token budgets must be positive integers")
    if config["target_tokens"] < 2 or config["input_tokens"] + config["target_tokens"] > 1024:
        raise ValueError("Invalid GPT-2 context/target budget")
    if not 0 <= config["warmup_fraction"] < 1 or config["learning_rate"] <= 0 or config["max_grad_norm"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    if len(config["revision"]) != 40 or any(c not in "0123456789abcdef" for c in config["revision"]):
        raise ValueError("Pin the model to a full immutable revision")


def validation_loss(model, loader, device, amp):
    import torch
    model.eval()
    loss_sum, tokens = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            count = int((batch["labels"][:, 1:] != -100).sum())
            with amp():
                loss = model(**batch).loss
            loss_sum += float(loss) * count
            tokens += count
    return loss_sum / tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--data", default="preprocessed_data.pkl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be empty; use a new directory to preserve the previous experiment")
    seed_everything(config["seed"])
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install a CUDA wheel or explicitly choose --device cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"], local_files_only=args.local_files_only)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    frame = load_original_dataset(args.data)
    manifest = make_manifest(frame, tokenizer, config, DATA_SHA256)
    write_json(output / "run_config.json", config)
    write_json(output / "split_manifest.json", manifest)
    metadata = runtime_info()
    metadata.update(data_sha256=DATA_SHA256, config_sha256=sha256_file(output / "run_config.json"),
                    split_manifest_sha256=sha256_file(output / "split_manifest.json"), device_requested=args.device,
                    precision="bfloat16 autocast, float32 parameters" if args.device == "cuda" else "float32",
                    selection="minimum validation target-token NLL at epoch boundaries; test is excluded")
    write_json(output / "environment.json", metadata)
    rows, token_stats = {}, {}
    for split in ("train", "validation"):
        selected = frame.iloc[[item["row"] for item in manifest["splits"][split]]]
        rows[split] = [training_example(tokenizer, review, summary, config["input_tokens"], config["target_tokens"])
                       for summary, review in selected[["Summary", "Text"]].itertuples(index=False, name=None)]
        token_stats[split] = {"examples": len(selected),
                             "supervised_tokens_including_eos": sum(sum(t != -100 for t in row["labels"]) for row in rows[split]),
                             "truncated_targets": sum(len(tokenizer.encode(" " + summary.strip(), add_special_tokens=False)) >= config["target_tokens"] for summary in selected.Summary)}
    write_json(output / "token_statistics.json", token_stats)
    generator = torch.Generator().manual_seed(config["seed"])
    collator = partial(collate, pad_token_id=tokenizer.eos_token_id)
    loader = DataLoader(rows["train"], batch_size=config["batch_size"], shuffle=True, generator=generator, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(rows["validation"], batch_size=config["evaluation_batch_size"], shuffle=False, collate_fn=collator, num_workers=0)
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["revision"],
                                               local_files_only=args.local_files_only, use_safetensors=True,
                                               attn_implementation="eager").to(args.device)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # AdamW decay excludes bias and normalization parameters.
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config["weight_decay"]},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=config["learning_rate"])
    steps_per_epoch = math.ceil(len(loader) / config["gradient_accumulation"])
    total_steps = steps_per_epoch * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["warmup_fraction"]), total_steps)
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This run uses BF16; choose CPU or a GPU supporting BF16")
    amp = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if args.device == "cuda" else nullcontext
    started = time.perf_counter()
    best_loss, best_epoch, step = float("inf"), None, 0
    history = []
    initial_val = validation_loss(model, val_loader, args.device, amp)
    print(json.dumps({"event": "initial_validation", "target_token_nll": initial_val, "train_examples": len(rows['train'])}), flush=True)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_nll_sum, train_tokens, group_tokens = 0.0, 0, 0
        for batch_index, batch in enumerate(loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            count = int((batch["labels"][:, 1:] != -100).sum())
            with amp():
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            # Accumulate SUM loss, then normalize by actual target tokens across
            # the group. Short summaries and the final partial batch stay correct.
            (loss * count).backward()
            train_nll_sum += float(loss.detach()) * count
            train_tokens += count
            group_tokens += count
            if (batch_index + 1) % config["gradient_accumulation"] == 0 or batch_index + 1 == len(loader):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_tokens)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_tokens = 0
                step += 1
                if step % 25 == 0:
                    print(json.dumps({"event": "progress", "epoch": epoch, "step": step, "total_steps": total_steps,
                                      "train_target_token_nll": train_nll_sum / train_tokens,
                                      "gradient_norm": float(grad_norm), "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
        val_loss = validation_loss(model, val_loader, args.device, amp)
        if not math.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
            model.config.use_cache = True
            model.save_pretrained(output / "checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(output / "checkpoint")
            model.config.use_cache = False
        record = {"epoch": epoch, "optimizer_steps": step, "train_target_token_nll": train_nll_sum / train_tokens,
                  "validation_target_token_nll": val_loss, "checkpoint_selected": improved,
                  "elapsed_seconds": round(time.perf_counter() - started, 3)}
        history.append(record)
        write_json(output / "training_history.json", {"initial_validation_target_token_nll": initial_val,
                   "epochs": history, "selected_epoch": best_epoch, "selected_validation_target_token_nll": best_loss})
        print(json.dumps(record), flush=True)
    write_json(output / "checkpoint_sha256.json", {path.name: sha256_file(path) for path in sorted((output / "checkpoint").iterdir()) if path.is_file()})
    write_json(output / "training_complete.json", {"optimizer_steps": step, "selected_epoch": best_epoch,
               "elapsed_seconds": round(time.perf_counter() - started, 3),
               "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None})


if __name__ == "__main__":
    main()
