"""Validation-only decoding selection, then a locked fresh-test comparison."""
import argparse
import gc
import json
from pathlib import Path
import time

from comparison_data import (assert_hashes, load_protocol, read_json, source_hashes,
                             t5_input, visible_review)
from evaluate import baseline, generate_batch, load_original_dataset, rouge_scores, words
from evaluate_checkpoint import summarize
from train import runtime_info, seed_everything, sha256_file
from training_data import write_json

SOURCE_NAMES = ["compare_summarizers.py", "comparison_data.py", "evaluate.py", "evaluate_checkpoint.py", "train.py", "training_data.py"]


def generate_t5(model, tokenizer, texts, config, beams):
    import torch
    rows = [{"input_ids": t5_input(tokenizer, text, config["t5_input_tokens"])} for text in texts]
    batch = tokenizer.pad(rows, padding=True, return_tensors="pt")
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.inference_mode():
        ids = model.generate(**batch, max_new_tokens=config["target_tokens"], min_length=0,
            do_sample=False, num_beams=beams, length_penalty=1.0, repetition_penalty=1.0,
            no_repeat_ngram_size=0, early_stopping=beams > 1)
    return tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def verify_partial(partial, identity, rows):
    if partial["identity"] != identity:
        raise ValueError("Cannot resume: experiment identity changed")
    examples = partial["examples"]
    if len(examples) > len(rows) or [x["source_row"] for x in examples] != [x["row"] for x in rows[:len(examples)]]:
        raise ValueError("Cannot resume: rows missing, duplicated or reordered")
    if any(not isinstance(x["prediction"], str) or x["batch_seconds"] < 0 for x in examples):
        raise ValueError("Invalid partial predictions")


def evaluate_model(kind, model_path, beams, rows, frame, gpt_tokenizer, config, identity, output, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    path = Path(output)
    partial = read_json(path) if path.exists() else {"identity": identity, "examples": []}
    verify_partial(partial, identity, rows)
    if len(partial["examples"]) == len(rows):
        return partial["examples"]
    if kind == "t5":
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, use_safetensors=True).to(device).eval()
        generate = lambda texts: [s.strip() for s in generate_t5(model, tokenizer, texts, config, beams)]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, use_safetensors=True,
                                                    attn_implementation="eager").to(device).eval()
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        generate = lambda texts: generate_batch(model, tokenizer, texts, config["gpt2_input_tokens"], config["target_tokens"])
    model.config.use_cache = True
    generate(["The package arrived intact and the food tasted fresh."])
    for start in range(len(partial["examples"]), len(rows), config["evaluation_batch_size"]):
        selected = rows[start:start + config["evaluation_batch_size"]]
        raw = [frame.iloc[r["row"]].Text for r in selected]
        texts = [visible_review(gpt_tokenizer, t, config["gpt2_input_tokens"]) for t in raw] if kind == "t5" else raw
        if device == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        predictions = generate(texts)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        if len(predictions) != len(selected):
            raise ValueError("Incomplete model batch")
        for row, prediction in zip(selected, predictions):
            partial["examples"].append({"source_row": row["row"], "prediction": prediction,
                "batch_seconds": elapsed, "batch_size": len(selected), "batch_start": start})
        # Atomic replace supports interruption without truncating the only saved copy.
        temporary = path.with_suffix(".tmp")
        write_json(temporary, partial)
        temporary.replace(path)
        if (start + len(selected)) % 64 == 0:
            print(json.dumps({"model": kind, "beams": beams, "completed": start + len(selected), "total": len(rows)}), flush=True)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return partial["examples"]


def validate_facts(facts, manifest, frame, gpt, config):
    if facts.get("reviewer") != "agent-authored; no independent human review":
        raise ValueError("Missing evaluation provenance")
    items = facts["examples"]
    if [x["source_row"] for x in items] != manifest["detail_audit_rows"]:
        raise ValueError("Detail audit must include exactly the prespecified rows in order")
    for item in items:
        text = visible_review(gpt, frame.iloc[item["source_row"]].Text, config["gpt2_input_tokens"])
        if len(item["facts"]) != 2:
            raise ValueError("Exactly two prespecified product facts per review are required")
        for fact in item["facts"]:
            if not fact["fact"].strip() or not fact["evidence"].strip() or fact["evidence"] not in text:
                raise ValueError("Fact evidence must quote the actual visible input")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("validation", "test"))
    p.add_argument("--protocol", required=True)
    p.add_argument("--gpt2-run", required=True)
    p.add_argument("--t5-run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--facts")
    p.add_argument("--data", default="preprocessed_data.pkl")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = p.parse_args()
    root = Path(args.output)
    result_path = root / ("validation_selection.json" if args.phase == "validation" else "fresh_test.json")
    if result_path.exists():
        p.error("Completed results are immutable; do not repeatedly evaluate the fresh test")
    config, manifest, freeze = load_protocol(args.protocol)
    seed_everything(config["seed"])
    from transformers import AutoTokenizer
    checkpoints = {}
    for kind, run in (("gpt2", args.gpt2_run), ("t5", args.t5_run)):
        if not (Path(run) / "training_complete.json").is_file():
            raise ValueError("Training incomplete")
        checkpoints[kind] = read_json(Path(run) / "checkpoint_sha256.json")
        assert_hashes(Path(run) / "checkpoint", checkpoints[kind])
    if checkpoints["gpt2"] != freeze["gpt2_checkpoint_sha256"]:
        raise ValueError("GPT-2 checkpoint differs from frozen original")
    if read_json(Path(args.t5_run) / "environment.json")["protocol"] != freeze:
        raise ValueError("T5 was trained under another protocol")
    gpt = AutoTokenizer.from_pretrained(str(Path(args.gpt2_run) / "checkpoint"), local_files_only=True)
    frame = load_original_dataset(args.data)
    identity = {"phase": args.phase, "protocol": freeze, "checkpoints": checkpoints,
                "source_sha256": source_hashes(SOURCE_NAMES), "device": args.device,
                "environment": {k: v for k, v in runtime_info().items() if k not in ("git_commit", "code_sha256")}}
    split = "validation" if args.phase == "validation" else "fresh_test"
    rows = manifest["splits"][split]
    if args.phase == "validation":
        candidates = config["decoding_candidates"]
    else:
        selection = read_json(root / "validation_selection.json")
        for key in ("protocol", "checkpoints", "source_sha256", "device", "environment"):
            if selection["identity"][key] != identity[key]:
                raise ValueError(f"Selection identity changed before final test: {key}")
        if not args.facts:
            p.error("Provide source-grounded facts prepared before inspecting final predictions")
        validate_facts(read_json(args.facts), manifest, frame, gpt, config)
        identity["facts_sha256"] = sha256_file(args.facts)
        identity["selection_sha256"] = sha256_file(root / "validation_selection.json")
        lock = root / "test_lock.json"
        if lock.exists() and read_json(lock) != identity:
            raise ValueError("Final-test lock changed")
        if not lock.exists():
            write_json(lock, identity)
        candidates = [selection["selected_beams"]]
    model_outputs = {}
    for beams in candidates:
        key = f"t5_beam{beams}"
        model_outputs[key] = evaluate_model("t5", str(Path(args.t5_run) / "checkpoint"), beams, rows,
            frame, gpt, config, {**identity, "system": key}, root / f"{split}_{key}.json", args.device)
    if args.phase == "test":
        model_outputs["gpt2"] = evaluate_model("gpt2", str(Path(args.gpt2_run) / "checkpoint"), 1, rows,
            frame, gpt, config, {**identity, "system": "gpt2"}, root / "fresh_test_gpt2.json", args.device)
    predictions = {key: [r["prediction"] for r in value] for key, value in model_outputs.items()}
    predictions["lead_16_words"] = [baseline(visible_review(gpt, frame.iloc[r["row"]].Text, config["gpt2_input_tokens"])) for r in rows]
    references = [frame.iloc[r["row"]].Summary for r in rows]
    scores = {key: [rouge_scores(p, r) for p, r in zip(preds, references)] for key, preds in predictions.items()}
    summaries = {key: summarize(values, config["seed"], config["bootstrap_samples"]) for key, values in scores.items()}
    timings = {key: {"warm_generation_seconds": sum(x["batch_seconds"] / x["batch_size"] for x in records),
                     "examples_per_second": len(records) / sum(x["batch_seconds"] / x["batch_size"] for x in records)}
               for key, records in model_outputs.items()}
    result = {"identity": identity, "sample_size": len(rows), "scores": summaries, "timing": timings,
        "mean_output_words": {key: sum(len(words(x)) for x in values) / len(values) for key, values in predictions.items()},
        "examples": [{"source_row": row["row"], "reference": references[i],
            "visible_review": visible_review(gpt, frame.iloc[row["row"]].Text, config["gpt2_input_tokens"]),
            "predictions": {key: values[i] for key, values in predictions.items()},
            "scores": {key: values[i] for key, values in scores.items()}} for i, row in enumerate(rows)]}
    if args.phase == "validation":
        result["selected_beams"] = min(candidates, key=lambda b: (-summaries[f"t5_beam{b}"]["rougeL"]["mean"], b))
    else:
        result["paired_differences"] = {}
        for a, b in ((f"t5_beam{candidates[0]}", "gpt2"), ("gpt2", "lead_16_words"), (f"t5_beam{candidates[0]}", "lead_16_words")):
            deltas = [{metric: x[metric] - y[metric] for metric in ("rouge1", "rougeL")} for x, y in zip(scores[a], scores[b])]
            result["paired_differences"][a + "_minus_" + b] = summarize(deltas, config["seed"], config["bootstrap_samples"])
    write_json(result_path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("examples", "identity")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
