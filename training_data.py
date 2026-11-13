"""Target-only causal-LM examples and deterministic, prompt-disjoint splits."""
import hashlib
import json
import re
from pathlib import Path

PROMPT_SUFFIX = " TL;DR"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_ids(tokenizer, review, input_tokens):
    suffix = tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)
    if input_tokens <= len(suffix):
        raise ValueError("Input budget must leave room for review tokens")
    return tokenizer.encode(str(review), add_special_tokens=False)[:input_tokens - len(suffix)] + suffix


def training_example(tokenizer, review, summary, input_tokens, target_tokens):
    if target_tokens < 2 or not str(summary).strip():
        raise ValueError("A nonempty summary and at least two target tokens are required")
    prompt = prompt_ids(tokenizer, review, input_tokens)
    # Tokenize separately so the prompt/target boundary is exact and the summary
    # survives review truncation. EOS is a real target, even when EOS is also PAD.
    target = tokenizer.encode(" " + str(summary).strip(), add_special_tokens=False)[:target_tokens - 1]
    target += [tokenizer.eos_token_id]
    return {"input_ids": prompt + target, "attention_mask": [1] * (len(prompt) + len(target)),
            "labels": [-100] * len(prompt) + target}


def collate(rows, pad_token_id):
    import torch
    length = max(len(row["input_ids"]) for row in rows)
    result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for row in rows:
        padding = length - len(row["input_ids"])
        result["input_ids"].append(row["input_ids"] + [pad_token_id] * padding)
        result["attention_mask"].append(row["attention_mask"] + [0] * padding)
        result["labels"].append(row["labels"] + [-100] * padding)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def make_manifest(frame, tokenizer, config, data_sha256):
    # Deduplicate full normalized reviews, then the ACTUAL truncated token prompt.
    # The first source row is the canonical reference; no metric guides selection.
    seen_reviews, seen_prompts, records = set(), set(), []
    skipped = {"empty": 0, "normalized_review_duplicate": 0, "prompt_duplicate": 0}
    for row, (summary, review) in enumerate(frame[["Summary", "Text"]].itertuples(index=False, name=None)):
        if not isinstance(summary, str) or not isinstance(review, str) or not summary.strip() or not review.strip():
            skipped["empty"] += 1
            continue
        review_hash = digest(" ".join(re.findall(r"\w+", review.casefold())))
        if review_hash in seen_reviews:
            skipped["normalized_review_duplicate"] += 1
            continue
        seen_reviews.add(review_hash)
        prompt_hash = digest(json.dumps(prompt_ids(tokenizer, review, config["input_tokens"])))
        if prompt_hash in seen_prompts:
            skipped["prompt_duplicate"] += 1
            continue
        seen_prompts.add(prompt_hash)
        order = digest(f"{config['seed']}:{prompt_hash}")
        bucket = int(order[:8], 16) % 10
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        records.append({"row": row, "review_sha256": review_hash, "prompt_sha256": prompt_hash,
                        "order": order, "split": split})
    records.sort(key=lambda record: record["order"])
    partitions = {}
    available = {}
    for split in ("train", "validation", "test"):
        candidates = [record for record in records if record["split"] == split]
        available[split] = len(candidates)
        count = config[f"{split}_examples"]
        if not 0 < count <= len(candidates):
            raise ValueError(f"Requested {count} {split} examples; only {len(candidates)} available")
        partitions[split] = [{k: v for k, v in record.items() if k not in ("order", "split")}
                             for record in candidates[:count]]
    manifest = {"schema_version": 1, "data_sha256": data_sha256, "seed": config["seed"],
                "input_tokens": config["input_tokens"], "source_rows": len(frame), "skipped": skipped,
                "available_unique_prompts": available, "splits": partitions}
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    for key in ("row", "review_sha256", "prompt_sha256"):
        values = [item[key] for split in manifest["splits"].values() for item in split]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate or cross-split {key}")


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
