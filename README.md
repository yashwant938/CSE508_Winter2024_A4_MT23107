# GPT-2 Review Summarization

A CSE508 coursework experiment exploring GPT-2 summarization for Amazon Fine Food Reviews. The original notebook is preserved as an experiment record; the new evaluator provides a separate, reproducible review-only generation and scoring path.

## What changed

The notebook's historical evaluation is unreliable: its generation inputs can contain the reference summary, and one loop scores only the first item in each batch. `evaluate.py` keeps reference summaries out of generation inputs, decodes only new tokens, and scores every example, including a partial final batch. It compares a 16-word lead baseline with a selected GPT-2 model using per-example ROUGE-1 and ROUGE-L F1. The metric implementation uses lowercase alphanumeric tokens without stemming.

**The original fine-tuned `model.pt` is not included.** The published [pilot results](results/REPORT.md) therefore evaluate the public pretrained GPT-2 model, not the coursework fine-tuned model. The original notebook still contains legacy imports, Kaggle paths, and training-loss/padding issues; it is not a clean end-to-end reproduction. Historical notebook scores should not be used as evidence of model quality.

## Reproduce the pilot

Use Python 3.12 on a supported platform. The model is downloaded from Hugging Face on the first run (several hundred MB). CPU inference works but takes time. The direct dependency versions used for the pilot are listed in `requirements-evaluation.txt`; this is not a complete transitive lockfile and availability of PyTorch builds varies by platform.

```bash
python -m venv .venv
# Activate the virtual environment using your platform's instructions.
python -m pip install -r requirements-evaluation.txt
python -m unittest discover -s tests -v
python evaluate.py --model openai-community/gpt2 \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --sample-size 513 --batch-size 4 --input-tokens 128 --output-tokens 24
```

The evaluator uses the repository's `preprocessed_data.pkl`, the first 513 entries of the notebook's seeded 25% test split (`random_state=42`), and deterministic greedy decoding. The file's exact SHA-256 is checked and its pickle globals are restricted before loading. Pickle is executable data: never load arbitrary or untrusted pickle files. Results include the source hash, model revision, settings, dependency versions, per-example outcomes and measured generation time. The public checkpoint is a pretrained baseline, not a replacement for the missing fine-tuned checkpoint.

If the original trusted notebook state dictionary becomes available, add `--state-dict /path/to/model.pt` while using the corresponding base model and revision. The evaluator loads only weights (`weights_only=True`), checks architecture keys strictly, and records a SHA-256. This path cannot establish whether a supplied checkpoint was actually trained on the notebook split. Obtain model artifacts only from sources you trust.

## Scope and limitations

- The dataset contains preprocessed reviews and short user-written titles; automatic token-overlap scores are limited proxies for usefulness and factuality.
- The 513-review sample is a deterministic subset, not an independently annotated or deduplicated test set. The notebook's random split may contain duplicates, and no confidence interval or human evaluation is reported.
- The lead baseline returns the first 16 review words; GPT-2 generates up to 24 new tokens. The public base model has not been fine-tuned for this task.
- Prompts preserve the notebook's ` TL;DR` separator. Reference summaries are passed only to the scorer, and all generated batch elements are evaluated.
- The original training notebook needs modern dependency updates, corrected loss masking for padding, and verified model artifacts before its fine-tuning results can be reproduced.

## Repository map

- `CSE508_Winter2024_A4_MT23107..ipynb` — historical training and generation notebook.
- `preprocessed_data.pkl` — original preprocessed dataset artifact.
- `evaluate.py`, `tests/` — corrected evaluator and focused regression checks.
- `results/` — pilot metrics and examples, including limitations.
