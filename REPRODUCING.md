# Reproducible GPT-2 summarization experiment

The supported workflow is `train.py` followed by `evaluate_checkpoint.py`. The
assignment notebook is retained as historical work; its loss and evaluation are
not the basis for the new results.

## What is corrected

- The review and ` TL;DR` prompt receive label `-100`. Only summary tokens and a
  genuine end-of-sequence token contribute to causal language-model loss.
- Padding also receives label `-100` and attention mask `0`. Although padding and
  EOS share a token ID, the real summary EOS remains supervised.
- Review and target budgets are separate. Long reviews cannot truncate away the
  target. Training is right-padded; batched generation is left-padded and returns
  every completion without decoding the prompt as part of the prediction.
- Gradient accumulation averages over actual supervised tokens, including the
  final partial accumulation group. Validation NLL is target-token weighted.
- Normalized duplicate reviews and identical truncated model prompts are removed
  before a fixed hashed split. The first source row supplies the canonical target.
  Semantic near-duplicates and pretraining contamination are not ruled out.
- A fresh public GPT-2 checkpoint is fine-tuned. Selection uses validation NLL
  at epoch boundaries; test targets do not enter the training loop or selection.

The [GPT-2 documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
describes its internally shifted causal labels and ignored `-100` positions.
The tests also compare the model loss with an independent shifted cross-entropy
calculation, including variable-length padded examples.

## Environment

Python 3.12 is used. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For CPU, replace `cu128` with `cpu` and pass `--device cpu` below. The recorded
GPU protocol uses BF16 autocast with float32 parameters, gradient checkpointing
and the eager attention implementation. It requires a BF16-capable GPU; the
recorded run uses an RTX 4050 Laptop GPU with 6 GB VRAM. No paid APIs are used.

The model ID and immutable upstream revision are in `configs/pilot.json`.
The source pickle is accepted only after its exact SHA-256 matches the repository
artifact, through an allowlisted unpickler. Do not replace it with an unknown
pickle. The experiment processes the supplied review/summary dataset locally.

## Train and save a real checkpoint

```powershell
$env:HF_HOME = "$PWD/.hf-cache"
.\.venv\Scripts\python.exe train.py --config configs/pilot.json --output runs/pilot --device cuda
```

The first run downloads the pinned public base model. After caching it, add
`--local-files-only` to train without network access. The output directory must
be empty; previous experiments are never overwritten.

The fixed pilot uses 6,000 training examples, 512 validation examples and 512
test examples, two epochs, microbatch size 2 and eight accumulation steps.
The token budgets are 192 prompt tokens and 32 target tokens including EOS.
This is a deliberately limited experiment, not training on the full dataset.

Saved artifacts include:

| Artifact | Purpose |
| --- | --- |
| `checkpoint/model.safetensors` | Actual fine-tuned weights, selected by validation NLL |
| `checkpoint/config.json` and tokenizer files | Reloadable model and tokenizer |
| `run_config.json` | All training, generation and split settings |
| `split_manifest.json` | Exact row IDs and review/prompt hashes for all three splits |
| `environment.json` | Library versions, hardware, data/config/manifest/code hashes |
| `token_statistics.json` | Supervised token counts and target truncation counts |
| `training_history.json` | Initial and per-epoch validation NLL and selection |
| `checkpoint_sha256.json` | Checkpoint and tokenizer integrity hashes |
| `training_complete.json` | Completion, selected epoch, elapsed time and peak memory |

This saves a reloadable inference checkpoint, not optimizer/RNG state for
mid-epoch resumption. Reproduce training by restarting from the pinned base model
with the recorded seed and environment. Seeds and deterministic algorithms reduce
variation; they do not promise bit-identical results across platforms, library
versions or GPU models. See [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Evaluate once on the untouched test partition

```powershell
.\.venv\Scripts\python.exe evaluate_checkpoint.py --run runs/pilot --output results/reproducible-pilot/evaluation.json --device cuda --local-files-only
```

The evaluator checks configuration, manifest and checkpoint hashes, then compares
three systems on the same test examples:

1. The first 16 words of the review, a deterministic extractive baseline.
2. The pinned pretrained GPT-2 model without fine-tuning.
3. The saved validation-selected fine-tuned GPT-2 checkpoint.

Both GPT-2 models receive review-only prompts, use greedy generation and share
the same input/output budgets. Full references are used for scoring even when a
training target would have been truncated. Evaluation is float32 on the selected
device. Timing excludes model loading and a fixed synthetic warm-up example.

Results contain every prediction, reference, per-example score, mean ROUGE-1/L
F1, percentile bootstrap intervals and paired differences. Tokenization is
lowercase alphanumeric without stemming, matching `evaluate.py`. These numbers
are not directly interchangeable with a differently configured ROUGE library.
Bootstrap intervals quantify variation within this test sample, not training-seed
variability. ROUGE does not establish factuality or production usefulness.

Do not tune after inspecting this test set and then describe it as untouched.
Future experiments should use validation data for tuning and reserve a new
independent test set for a fresh final claim.
