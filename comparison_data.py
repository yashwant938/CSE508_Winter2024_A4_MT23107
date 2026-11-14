"""Freeze fresh row IDs before training; keep the original training/validation rows."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from evaluate import DATA_SHA256, load_original_dataset, words
from train import sha256_file
from training_data import digest, prompt_ids, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def visible_review(tokenizer, text, budget):
    suffix_length = len(tokenizer.encode(" TL;DR", add_special_tokens=False))
    ids = tokenizer.encode(text, add_special_tokens=False)[:budget - suffix_length]
    return tokenizer.decode(ids, clean_up_tokenization_spaces=False)


def t5_input(tokenizer, text, limit):
    ids = tokenizer.encode("summarize: " + text, add_special_tokens=True)
    if len(ids) > limit:
        raise ValueError("T5 would see truncated evidence; increase the frozen budget before running")
    return ids


def t5_example(tokenizer, text, summary, config):
    ids = t5_input(tokenizer, text, config["t5_input_tokens"])
    labels = tokenizer.encode(summary.strip(), add_special_tokens=False)[:config["target_tokens"] - 1]
    labels.append(tokenizer.eos_token_id)
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def seq2seq_collate(rows, pad_token_id):
    import torch
    source_length = max(len(x["input_ids"]) for x in rows)
    target_length = max(len(x["labels"]) for x in rows)
    result = {k: [] for k in ("input_ids", "attention_mask", "labels")}
    for row in rows:
        result["input_ids"].append(row["input_ids"] + [pad_token_id] * (source_length - len(row["input_ids"])))
        result["attention_mask"].append(row["attention_mask"] + [0] * (source_length - len(row["input_ids"])))
        result["labels"].append(row["labels"] + [-100] * (target_length - len(row["labels"])))
    return {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}


class NearDuplicates:
    """Exact 5-word-shingle Jaccard search via an inverted index; not semantic matching."""
    def __init__(self, threshold):
        self.threshold = threshold
        self.sets = []
        self.index = defaultdict(set)

    @staticmethod
    def shingles(text):
        tokens = words(text)
        return {tuple(tokens[i:i + 5]) for i in range(len(tokens) - 4)}

    def add(self, text):
        shingles = self.shingles(text)
        index = len(self.sets)
        self.sets.append(shingles)
        for shingle in shingles:
            self.index[shingle].add(index)

    def matches(self, text):
        shingles = self.shingles(text)
        if not shingles:
            return False
        overlaps = Counter(i for s in shingles for i in self.index.get(s, ()))
        return any(count / (len(shingles) + len(self.sets[i]) - count) >= self.threshold
                   for i, count in overlaps.items())


def source_hashes(names):
    # Source text hashes use LF on both Windows and Linux; data/model hashes are raw bytes.
    return {name: hashlib.sha256(Path(name).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
            for name in names}


def assert_hashes(root, hashes):
    for name, expected in hashes.items():
        if sha256_file(Path(root) / name) != expected:
            raise ValueError(f"Artifact changed: {name}")


def assert_fresh(manifest):
    historical = manifest["historical_exclusions"]
    fresh = manifest["splits"]["fresh_test"]
    for key in ("row", "review_sha256", "prompt_sha256", "t5_prompt_sha256", "visible_sha256"):
        old = {r[key] for r in historical}
        new = [r[key] for r in fresh]
        if old.intersection(new) or len(new) != len(set(new)):
            raise ValueError(f"Fresh test overlaps historical data or itself: {key}")


def prepare(args):
    from transformers import AutoTokenizer
    output = Path(args.output)
    if output.exists():
        raise ValueError("Choose a new protocol directory; frozen artifacts are not overwritten")
    config = read_json(args.config)
    old = read_json(Path(args.old_run) / "split_manifest.json")
    old_config = read_json(Path(args.old_run) / "run_config.json")
    old_env = read_json(Path(args.old_run) / "environment.json")
    for filename, key in (("split_manifest.json", "split_manifest_sha256"), ("run_config.json", "config_sha256")):
        if sha256_file(Path(args.old_run) / filename) != old_env[key]:
            raise ValueError("Original training provenance changed")
    if old["data_sha256"] != DATA_SHA256 or old_config["input_tokens"] != config["gpt2_input_tokens"]:
        raise ValueError("Dataset or GPT-2 input budget differs from original training")
    checkpoint_hashes = read_json(Path(args.old_run) / "checkpoint_sha256.json")
    assert_hashes(Path(args.old_run) / "checkpoint", checkpoint_hashes)
    gpt = AutoTokenizer.from_pretrained(str(Path(args.old_run) / "checkpoint"), local_files_only=True)
    t5 = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"], local_files_only=True)
    frame = load_original_dataset(args.data)
    original_rows = {r["row"] for split in old["splits"].values() for r in split}
    legacy_rows = {r["source_row"] for r in read_json("results/pilot.json")["examples"]}
    excluded = original_rows | legacy_rows
    def record(i):
        text = frame.iloc[i].Text
        visible = visible_review(gpt, text, config["gpt2_input_tokens"])
        return {"row": i, "review_sha256": digest(normalized(text)),
                "prompt_sha256": digest(json.dumps(prompt_ids(gpt, text, config["gpt2_input_tokens"]))),
                "t5_prompt_sha256": digest(json.dumps(t5_input(t5, visible, config["t5_input_tokens"]))),
                "visible_sha256": digest(normalized(visible))}, visible
    historical = []
    seen = {k: set() for k in ("review_sha256", "prompt_sha256", "t5_prompt_sha256", "visible_sha256")}
    near = NearDuplicates(config["near_duplicate_jaccard"])
    for i in sorted(excluded):
        row, visible = record(i)
        historical.append(row)
        for key in seen:
            seen[key].add(row[key])
        near.add(visible)
    # T5 normalization can collapse strings that GPT-2 distinguishes. Audit it explicitly.
    by_id = {r["row"]: r for r in historical}
    train_t5 = {by_id[r["row"]]["t5_prompt_sha256"] for r in old["splits"]["train"]}
    val_t5 = {by_id[r["row"]]["t5_prompt_sha256"] for r in old["splits"]["validation"]}
    if train_t5 & val_t5:
        raise ValueError("T5 tokenization creates a train/validation collision")
    candidates = sorted(set(range(len(frame))) - excluded, key=lambda i: digest(f"fresh:{config['seed']}:{i}"))
    selected, skipped = [], Counter()
    for i in candidates:
        summary, text = frame.iloc[i][["Summary", "Text"]]
        if not isinstance(text, str) or not isinstance(summary, str) or not text.strip() or not summary.strip():
            skipped["empty"] += 1
            continue
        row, visible = record(i)
        if any(row[k] in seen[k] for k in seen):
            skipped["exact_duplicate"] += 1
            continue
        if near.matches(visible):
            skipped["near_duplicate"] += 1
            continue
        selected.append(row)
        for key in seen:
            seen[key].add(row[key])
        near.add(visible)
        if len(selected) == config["fresh_test_examples"]:
            break
    if len(selected) != config["fresh_test_examples"]:
        raise ValueError("Not enough fresh rows")
    audit_ids = sorted([r["row"] for r in selected], key=lambda i: digest(f"audit:{config['seed']}:{i}"))[:config["detail_audit_examples"]]
    manifest = {"schema_version": 2, "data_sha256": DATA_SHA256, "source_rows": len(frame),
                "historical_exclusions": historical, "legacy_pilot_rows": sorted(legacy_rows),
                "skipped_candidates": dict(skipped), "detail_audit_rows": audit_ids,
                "splits": {"train": [by_id[r["row"]] for r in old["splits"]["train"]],
                           "validation": [by_id[r["row"]] for r in old["splits"]["validation"]], "fresh_test": selected}}
    assert_fresh(manifest)
    write_json(output / "config.json", config)
    write_json(output / "manifest.json", manifest)
    write_json(output / "freeze.json", {"config_sha256": sha256_file(output / "config.json"),
        "manifest_sha256": sha256_file(output / "manifest.json"), "gpt2_checkpoint_sha256": checkpoint_hashes,
        "preparation_source_sha256": source_hashes(["comparison_data.py", "evaluate.py", "training_data.py"]),
        "selection": "T5 epoch: lowest validation target-token NLL; decoding: highest validation ROUGE-L, tie -> fewer beams. GPT-2 unchanged.",
        "fresh_test": "512 new corrected-experiment rows; exclude all prior train/validation/test and legacy pilot; exact and 5-word Jaccard >=0.8 exclusions on visible text",
        "detail_review": "24 hash-selected fresh-test rows, two source-supported product facts each labeled before predictions; agent-authored, not independent human evaluation",
        "limitations": ["Only Text/Summary retained: no product IDs or product-disjoint guarantee.",
            "Original coursework data use and base-model pretraining contamination cannot be ruled out.",
            "Lexical near-duplicate filter does not guarantee semantic independence.",
            "Same input text and training examples/epochs, not identical tokenizers or compute."]})
    print(json.dumps({"frozen": str(output), "fresh_test": len(selected), "historical_exclusions": len(historical), "skipped": dict(skipped)}), flush=True)


def load_protocol(path):
    root = Path(path)
    freeze = read_json(root / "freeze.json")
    assert_hashes(root, {"config.json": freeze["config_sha256"], "manifest.json": freeze["manifest_sha256"]})
    manifest = read_json(root / "manifest.json")
    assert_fresh(manifest)
    return read_json(root / "config.json"), manifest, freeze


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/t5_comparison.json")
    p.add_argument("--old-run", required=True)
    p.add_argument("--data", default="preprocessed_data.pkl")
    p.add_argument("--output", required=True)
    prepare(p.parse_args())
