"""Evaluate a saved selected checkpoint against unchanged GPT-2 and lead-16."""
import argparse
import gc
import json
from pathlib import Path
import time

from evaluate import DATA_SHA256, baseline, generate_batch, load_original_dataset, rouge_scores
from train import runtime_info, seed_everything, sha256_file
from training_data import validate_manifest, write_json


def summarize(scores, seed, samples):
    import numpy as np
    rng = np.random.default_rng(seed)
    result = {}
    for metric in ("rouge1", "rougeL"):
        values = np.asarray([score[metric] for score in scores])
        means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
        result[metric] = {"mean": float(values.mean()), "bootstrap_95_ci": np.quantile(means, [.025, .975]).tolist()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", default="preprocessed_data.pkl")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Use a new output path; existing evaluation results are immutable")
    run = Path(args.run)
    if not (run / "training_complete.json").is_file():
        parser.error("Training has not completed")
    config = json.loads((run / "run_config.json").read_text())
    manifest = json.loads((run / "split_manifest.json").read_text())
    environment = json.loads((run / "environment.json").read_text())
    for filename, key in (("run_config.json", "config_sha256"), ("split_manifest.json", "split_manifest_sha256")):
        if sha256_file(run / filename) != environment[key]:
            raise ValueError(f"{filename} was modified after training started")
    hashes = json.loads((run / "checkpoint_sha256.json").read_text())
    for filename, expected in hashes.items():
        if sha256_file(run / "checkpoint" / filename) != expected:
            raise ValueError(f"Saved checkpoint changed: {filename}")
    validate_manifest(manifest)
    if manifest["data_sha256"] != DATA_SHA256:
        raise ValueError("Unexpected manifest dataset")
    seed_everything(config["seed"])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    frame = load_original_dataset(args.data)
    indices = [item["row"] for item in manifest["splits"]["test"]]
    rows = frame.iloc[indices]
    reviews, references = rows.Text.tolist(), rows.Summary.tolist()
    predictions = {"lead_16_words": [baseline(review) for review in reviews]}
    timings = {}
    for label, model_path, revision in (("pretrained_gpt2", config["model"], config["revision"]),
                                         ("fine_tuned_gpt2", str(run / "checkpoint"), None)):
        tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, local_files_only=args.local_files_only, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, local_files_only=args.local_files_only,
                                                    use_safetensors=True, attn_implementation="eager").to(args.device).eval()
        model.config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        # Warm-up uses a fixed synthetic review, not a held-out example/reference.
        generate_batch(model, tokenizer, ["The package arrived intact and the food tasted fresh."], config["input_tokens"], config["target_tokens"])
        if args.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        predictions[label] = []
        for start in range(0, len(reviews), config["evaluation_batch_size"]):
            predictions[label].extend(generate_batch(model, tokenizer, reviews[start:start + config["evaluation_batch_size"]], config["input_tokens"], config["target_tokens"]))
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        timings[label] = {"warm_generation_seconds": elapsed, "examples_per_second": len(reviews) / elapsed,
                          "mean_seconds_per_example": elapsed / len(reviews)}
        print(json.dumps({"model": label, "examples": len(predictions[label]), **timings[label]}), flush=True)
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    if any(len(value) != len(references) for value in predictions.values()):
        raise RuntimeError("A prediction is missing from the held-out evaluation")
    scores = {label: [rouge_scores(p, r) for p, r in zip(values, references)] for label, values in predictions.items()}
    summaries = {label: summarize(values, config["seed"], config["bootstrap_samples"]) for label, values in scores.items()}
    paired = {}
    for comparator in ("lead_16_words", "pretrained_gpt2"):
        deltas = [{metric: a[metric] - b[metric] for metric in ("rouge1", "rougeL")}
                  for a, b in zip(scores["fine_tuned_gpt2"], scores[comparator])]
        paired["fine_tuned_minus_" + comparator] = summarize(deltas, config["seed"], config["bootstrap_samples"])
    result = {"schema_version": 1, "sample_size": len(references), "data_sha256": DATA_SHA256,
              "config_sha256": environment["config_sha256"], "split_manifest_sha256": environment["split_manifest_sha256"],
              "checkpoint_files_sha256": hashes, "run_config": config, "environment": runtime_info(),
              "generation": "greedy; review-only left-padded prompts; float32 inference; same budgets for both models",
              "metric": "mean per-example ROUGE-1/L F1; lowercase alphanumeric tokens; no stemming; complete reference",
              "uncertainty": f"paired percentile bootstrap, {config['bootstrap_samples']} resamples of the fixed test sample; does not measure training-seed variability",
              "limitations": ["Small, single-seed fine-tuning experiment; no claim about production quality.",
                              "Exact normalized-review and exact token-prompt duplicates removed; semantic near-duplicates may remain.",
                              "ROUGE measures lexical overlap, not factuality, usefulness, or safety.",
                              "Historical notebook/pilot scores use a different split and are not directly comparable."],
              "scores": summaries, "paired_differences": paired, "timing": timings,
              "examples": [{"source_row": row, "prompt_sha256": manifest["splits"]["test"][i]["prompt_sha256"],
                            "reference": references[i], "predictions": {k: v[i] for k, v in predictions.items()},
                            "scores": {k: v[i] for k, v in scores.items()}} for i, row in enumerate(indices)]}
    write_json(args.output, result)
    print(json.dumps({"scores": summaries, "paired_differences": paired}, indent=2))


if __name__ == "__main__":
    main()
