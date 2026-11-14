# GPT-2 Review Summarization

A reproducible fine-tuning and evaluation experiment for short summaries of
Amazon Fine Food Reviews, extending the original CSE508 assignment.

## New: T5 comparison and product-detail audit

Compared the saved GPT-2 checkpoint with **fine-tuned T5-small**, using the same
6,000 training reviews, validation-only selection and **512 new test reviews**.
ROUGE-L F1 was **0.1275 GPT-2**, **0.1255 T5**, and **0.0974 lead-16**. The paired
T5/GPT-2 interval includes zero; neither establishes a clear advantage.

A prespecified 24-review, agent-authored audit found that the generated titles
usually omitted selected product facts and sometimes introduced unsupported claims.
Higher title overlap does not establish factual or detailed summarization.

[Full comparison and limitations](results/t5-comparison/REPORT.md) ·
[Reproduce the comparison](COMPARISON.md) · [22 offline tests](tests/)

## Original GPT-2 reproducibility work

The supported workflow trains a fresh, pinned GPT-2 model with summary-only
loss, saves a reloadable checkpoint and tokenizer, and compares it with
pretrained GPT-2 and a 16-word extractive baseline on the same untouched test
partition. It records configuration, split IDs, dependency versions, checkpoint
hashes, per-example predictions and bootstrap intervals.

[Reproduce training and evaluation](REPRODUCING.md) ·
[Fixed experiment configuration](configs/pilot.json) ·
[Tests](tests/test_training.py)

## Measured pilot result

On 512 untouched test reviews, fine-tuned GPT-2 achieved **ROUGE-L F1
0.1364**, compared with
**0.0947** for the first-16-words baseline
and **0.0421** for the pinned pretrained model.
This is a single-seed experiment on 6,000 training reviews, not a production
accuracy claim. [Results, raw predictions and limitations](results/reproducible-pilot/REPORT.md).

## Corrected training and evaluation

- **Target-only loss:** prompt and padding labels are `-100`; the true summary
  end-of-sequence token stays supervised even though padding uses the same ID.
- **Separate token budgets:** a long review cannot truncate away its summary.
  Training uses right padding; generation uses left-padded review-only prompts.
- **Split isolation:** normalized duplicate reviews and identical truncated model
  inputs are removed before deterministic hashed train/validation/test splits.
- **Validation-only selection:** the best epoch is selected by target-token
  validation loss. Test references do not select the checkpoint or decoding settings.
- **Saved artifacts:** actual fine-tuned `model.safetensors`, tokenizer, model and
  run configurations, training history, source/code/checkpoint hashes and row IDs.
- **Comparable held-out scores:** the fine-tuned and pretrained models receive
  the same prompts and greedy generation budget. Every test example is scored
  against its full reference, alongside the deterministic lead-16 baseline.

## Run

Use Python 3.12 and the [platform-specific setup instructions](REPRODUCING.md).

```bash
python -m unittest discover -s tests -v
python train.py --config configs/pilot.json --output runs/pilot --device cuda
python evaluate_checkpoint.py --run runs/pilot --output results/reproducible-pilot/evaluation.json --device cuda --local-files-only
```

The fixed pilot uses 6,000 training, 512 validation and 512 test examples, two
epochs, a 192-token prompt budget and 32 target tokens including EOS. CUDA
training uses BF16 autocast and float32 parameters; CPU training is available
with `--device cpu`. The first training run downloads the pinned public GPT-2
base weights. No paid API is used.

Checkpoint binaries and local model caches are excluded from Git history. The
training command creates the complete checkpoint under `runs/pilot/checkpoint`;
the evaluation command reloads and verifies it before scoring. A saved inference
checkpoint does not include optimizer state for mid-epoch resume.

## Evidence and limits

This is a single-seed, small-data experiment on review titles as reference
summaries, not a production summarization system. ROUGE-1/L F1 uses lowercase
alphanumeric tokens without stemming and measures lexical overlap, not
factuality. Bootstrap intervals quantify test-sample variation, not variability
across training runs. Semantic near-duplicates and base-model pretraining
contamination are not ruled out.

The original fine-tuned assignment `model.pt` was never included. The new
experiment retrains from the pinned public base model; it does not reconstruct
or claim to recover that missing historical checkpoint. The
[older pretrained-only pilot](results/REPORT.md) uses a different, non-deduplicated
split and should not be compared numerically with the new experiment.

The historical notebook retains its original coursework code, including Kaggle
paths and flawed training/evaluation cells, with a prominent warning. Use the
Python scripts for corrected work and do not cite historical notebook scores as
model-quality evidence.

## Repository map

| File | Purpose |
| --- | --- |
| `train.py`, `training_data.py` | Target masking, token-weighted training and reproducible splits |
| `evaluate_checkpoint.py` | Verified checkpoint reload, baselines and paired held-out evaluation |
| `configs/pilot.json` | Frozen training, split and generation protocol |
| `tests/` | Loss, padding/EOS, split leakage, data loading, generation and metric checks |
| `preprocessed_data.pkl` | Hash-checked original 39,999-row review/summary artifact |
| `evaluate.py` | Shared scoring and review-only generation; older pilot CLI |
| `REPRODUCING.md` | Environment, commands, artifacts and limitations |
| `CSE508_Winter2024_A4_MT23107..ipynb` | Historical assignment notebook |
