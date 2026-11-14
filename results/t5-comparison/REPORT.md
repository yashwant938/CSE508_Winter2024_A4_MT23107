# GPT-2 versus fine-tuned T5-small: fresh-test comparison

14 November 2026. Local training and inference; no paid inference APIs.

**Neither model established a clear ROUGE advantage over the other. Both beat
lead-16 on title overlap, but usually omitted selected product details.** This is
an evaluation result, not evidence of production-quality summarization.

## Final results

512 newly reserved reviews from the existing 39,999-row **Amazon Fine Food
Reviews** artifact. References are review titles, not detailed factual summaries.

| System | ROUGE-1 F1 | ROUGE-L F1 | Mean output words |
| --- | ---: | ---: | ---: |
| Fine-tuned GPT-2, greedy | 0.1278 | 0.1275 | 2.82 |
| Fine-tuned T5-small, greedy | 0.1271 | 0.1255 | 2.90 |
| First 16 whitespace-delimited review words | 0.1041 | 0.0974 | 13.55 |

The length column uses the metric's alphanumeric tokenizer. Standalone punctuation
in the preprocessed dataset explains why lead-16 averages fewer than 16 such words.

Paired ROUGE-L differences, with 95% percentile intervals from 2,000 example-level
bootstrap resamples:

- T5 minus GPT-2: **−0.0019 [−0.0196, +0.0156]**. This includes zero and does not
  establish superiority or equivalence.
- GPT-2 minus lead-16: **+0.0300 [+0.0119, +0.0482]**.
- T5 minus lead-16: **+0.0281 [+0.0102, +0.0463]**.

The historical GPT-2 score of 0.1364 used a different test sample. Do not describe
the new 0.1275 as a model regression: the GPT-2 checkpoint and decoding are unchanged.
ROUGE is lowercase alphanumeric, unstemmed, per-example F1 against the complete
reference, matching the earlier implementation. It is not factual accuracy.

[Final predictions and scores](fresh_test.json) · [Validation selection](validation_selection.json)

## Product-detail and source-support review

Before final predictions, a hash-selected **24-review subset** received two
source-supported facts per review. The frozen rubric asks whether each **complete
fact** survives, and separately reviews every emitted claim against the text
actually visible to the models. Partial details count as missing. Generic
positive sentiment does not preserve a named flavour, packaging property or
comparison. Pronouns may refer to the reviewed product; some sparse reviews have
only use/sentiment facts. These choices and their rationales are inspectable.

| System | Selected facts preserved | Summaries with all claims supported | Summaries with unsupported/contradicted claims |
| --- | ---: | ---: | ---: |
| GPT-2 | 0/48 | 15/24 | 6/24 |
| T5-small | 4/48 | 16/24 | 5/24 |
| Lead-16 | 20/48 | 24/24 | 0/24 |

Three summaries per generative system contain uncertain claims; uncertainty is
not counted as full support. Across all systems, 111 atomic claims were reviewed.
Copying a source prefix makes lead-16's source support unsurprising, but does not
guarantee completeness, readability or independently correct product claims.
Lead-16 is also much longer. The comparison does **not** isolate model architecture
from output length or show that lead-16 is universally better.

Examples from the prespecified audit:

- **Row 38556:** the review describes a **Torani syrup pump** dispensing the right
  amount. Both models generate **“Great syrup”**, changing the product identity.
- **Row 12239:** the source reports treats making a dog ill and recovery after
  stopping. GPT-2 generates **“My dog loves them”** with no supporting enjoyment
  evidence. T5's **“My dog was sick”** keeps illness but omits the treat attribution
  and recovery.
- **Row 38600:** T5's **“Tastes better than the senseo brand”** preserves the
  comparison; it still omits the synthetic-tasting Southern Pecan criticism.
- **Row 36764:** GPT-2's **“Great Product”** reverses a negative overall review;
  T5's **“guar gum is unnoticed”** conflicts with an explicit additive complaint.

This is **single-agent, non-blind, non-independent review**, not human-validated
factuality accuracy. The 24 cases are a small descriptive sample, and the
two-fact rubric is deliberately strict. Some labels require interpretation of
degraded preprocessed wording. Source support means support in that text, not
external verification of medical, dietary or product claims. No model, decoding
or fact-selection changes were made after observing final outputs.

[Prespecified facts and rubric](detail_facts.json) · [Authored annotations](audit_annotations.json) ·
[Expanded claim reviews](detail_audit.json) · [Counts and denominators](detail_scores.json)

## Training and selection

- Pinned `google-t5/t5-small` revision
  `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`, **60,506,880 parameters**.
  The implementation follows Hugging Face's [T5 summarization workflow](https://huggingface.co/docs/transformers/tasks/summarization):
  `summarize:` prefix, encoder-decoder targets, and ignored label padding.
- T5 uses the **same 6,000 training and 512 validation rows** as the saved GPT-2
  experiment. Both receive the same review prefix, defined by GPT-2's 192-token
  prompt budget including ` TL;DR`. T5's 512-token safety ceiling never silently
  truncates this shared evidence; longest training input was 284 T5 tokens.
- Both train for **two epochs / 750 optimizer steps**, effective batch size 16.
  T5 uses AdamW at 3e-4, 5% warm-up, BF16 autocast with float32 parameters. GPT-2
  retains its original 5e-5 learning rate and seed. Data exposure is matched;
  parameter count, optimizer settings, tokenizers and compute are not identical.
- T5 summary targets reserve EOS within 32 tokens; padding is `-100`. Encoder-
  decoder loss counts every target token without applying an extra causal shift.
  Gradient accumulation is weighted by actual target-token counts. Five training
  titles were truncated; no validation titles were truncated.
- **Epoch 2 selected by validation NLL:** 3.3497, versus epoch 1's 3.3944. NLLs
  across different tokenizers are not directly comparable.
- After checkpoint selection, two prespecified decoding candidates were evaluated
  on validation: greedy ROUGE-L **0.1347**, four-beam **0.1311**. Greedy was selected.
  Both use 32 maximum generated tokens, no sampling, length penalty 1.0.
- Training took **386.4 seconds** on an RTX 4050 Laptop GPU, with peak allocated
  CUDA memory **1,283,574,784 bytes**. Saved weights/tokenizer were reloaded and
  SHA-256 verified before validation and test.
- Fresh-test warm throughput at batch size four: **28.75 examples/s T5**,
  **21.03 examples/s GPT-2**. These are single-run tokenization/generation timings
  with CUDA synchronization, excluding model load, shared-prefix preparation and
  warm-up. They are not service latency or a repeated hardware benchmark.

[Training history](training_history.json) · [Configuration](protocol/config.json) ·
[Environment](environment.json) · [Checkpoint hashes](checkpoint_sha256.json)

## What “fresh” means here

The new sample excludes the union of **7,453 historical source rows**: original
training, validation, reported test, and the older pretrained-only pilot. It
also excludes normalized full-review duplicates, exact GPT-2 prompts, exact T5
inputs, normalized visible prefixes, and visible-prefix 5-word-shingle Jaccard
similarity at least 0.8, both against historical inputs and within the new sample.
Ten candidates were rejected as exact duplicates; no additional near-duplicate
candidate was encountered before filling the 512 slots.

The configuration, row identities, audit-row selection and exclusion rules were
frozen before T5 training. Validation selected checkpoint/decoding. Source-backed
facts were written before final test generation; their hash is in the final-test
lock. The evaluator saves partial batches atomically and only resumes under the
same checkpoint/config/source/environment identity. This run completed without
resuming or changing the final protocol.

“Fresh” means **unused by these corrected experiments**, not a newly collected
external dataset. Only Text/Summary columns survive, so product-disjoint splitting
cannot be established. Semantic duplicates, the old coursework notebook's data
use, and base-model pretraining contamination are not ruled out. One training seed
per model and example-level bootstrap intervals do not measure training variability
or product-cluster uncertainty. These test results are now observed; subsequent
decisions need validation data and any new final claim needs another audited holdout.

[Frozen manifest](protocol/manifest.json) · [Protocol freeze](protocol/freeze.json) ·
[Final-test lock](test_lock.json) · [Reproduction commands](../../COMPARISON.md)

## What the project can credibly claim

> Compared fine-tuned GPT-2 and T5-small on 512 newly held-out reviews using
> validation-selected decoding, paired ROUGE analysis and a source-grounded
> product-detail audit; saved reloadable checkpoints and reproducible split manifests.

For a longer project description, include that both models improved title overlap
over lead-16, while the detail audit exposed generic outputs and unsupported claims.
Do not claim T5 beat GPT-2, a factuality accuracy percentage, or reliable product
summarization. The next experiment should develop **detail-bearing target summaries
and validation criteria** before reserving another test set; changing decoding
against this now-observed test would invalidate a new held-out claim.
