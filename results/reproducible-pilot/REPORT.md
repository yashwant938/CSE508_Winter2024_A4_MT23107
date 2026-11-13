# Recorded GPT-2 fine-tuning experiment

A fresh GPT-2 checkpoint was trained on **6,000 unique
review prompts**, selected on 512 validation examples,
and evaluated on **512 untouched test examples**. The
training and decoding configuration was fixed before the run. This is a new
supervised run, not the missing original coursework checkpoint.

## Held-out result

| System | ROUGE-1 F1 | ROUGE-L F1 | ROUGE-L 95% bootstrap interval |
| --- | ---: | ---: | --- |
| lead_16_words | 0.1028 | 0.0947 | [0.0850, 0.1050] |
| pretrained_gpt2 | 0.0446 | 0.0421 | [0.0376, 0.0466] |
| fine_tuned_gpt2 | 0.1392 | 0.1364 | [0.1181, 0.1555] |

The paired ROUGE-L difference between fine-tuned GPT-2 and lead-16 is
**+0.0417**, with a percentile bootstrap 95% interval of
[+0.0230, +0.0607]. Intervals use
2,000 resamples of test examples with seed 20261113.
They quantify this sample's variation, not training-seed variability or a broader
population guarantee. Scores are fractions, not percentages of correct answers.

All systems use the same test rows. Both GPT-2 models receive review-only prompts
and greedy generation, with 192 input tokens and at most
32 generated tokens. The baseline extracts the first 16
whitespace-separated review words. Full reference titles are scored, using mean
per-example ROUGE-1 and ROUGE-L F1 with lowercase alphanumeric tokens and no
stemming. Reference titles are never included in generation prompts.

See [every prediction and score](evaluation.json). Historical notebook and
pretrained-only pilot scores use different splits and must not be compared with
these figures.

## Split and training record

The source contains 39,999 rows. Deduplication removes
2,583 normalized-review duplicates
and 5 further identical truncated token
prompts. Stable hashing creates approximately 80/10/10 train/validation/test
partitions; the fixed pilot takes 6,000/512/512 examples from them. Unused rows
remain unused. This does not eliminate semantic near-duplicates or establish
absence of GPT-2 pretraining contamination.

Prompt and padding labels are masked with `-100`; summary tokens and real EOS
remain supervised. Right-padded training reserves separate prompt and target
budgets. Gradient accumulation and validation loss are weighted by actual
supervised token counts. The selected checkpoint is **epoch
2**, chosen solely by validation NLL:

| Stage | Validation target-token NLL |
| --- | ---: |
| Pinned base model, before training | 4.8564 |
| Epoch 1 | 3.3833 |
| Epoch 2 | 3.3564 |

Training completed 750 optimizer steps in
17.61 minutes on an NVIDIA RTX 4050 Laptop GPU,
using BF16 autocast, float32 parameters and deterministic algorithms. Peak
allocated CUDA memory was 2.41
GiB; this excludes other processes and driver memory. The exact settings,
token counts, library versions and manifests are included beside this report.

## Timing

| System | Warm generation seconds | Examples/second |
| --- | ---: | ---: |
| pretrained_gpt2 | 107.48 | 4.76 |
| fine_tuned_gpt2 | 21.97 | 23.31 |

Evaluation uses float32 GPU inference. Times include tokenization and batched
generation, after a fixed synthetic warm-up, excluding model loading and scoring.
Different output lengths affect throughput. This is a serial laptop experiment,
not a concurrency, latency-SLA or production throughput benchmark.

## Quality limitations and visible errors

The references are short, user-written review titles. Lexical overlap can reward
generic titles and does not measure faithfulness or usefulness. Fine-tuned output
has 300 distinct strings across 512 examples and
0 empty outputs. The five most frequent
outputs are: 'Great product' (22), 'Great coffee' (17), 'Yum!' (14), 'Great tea' (13), 'Great Product' (12).

The following are the three largest ROUGE-L regressions relative to lead-16,
selected mechanically after scoring, not flattering examples:

- Row 1345: reference `Wish bone Fat Free Ranch Dressing`; fine-tuned prediction `Great Salad`; lead-16 `wish bone fat free ranch salad dressing , 16-ounce bottle ( pack 6 ) great salad`.
- Row 4627: reference `Ener-G Foods English Muffins`; fine-tuned prediction `Great for breakfast`; lead-16 `ener-g foods english muffins , 15.1-ounce units ( pack 6 ) one favorite gluten free items`.
- Row 33141: reference `organic hibiscus tea from Davidson's teas`; fine-tuned prediction `Great tea`; lead-16 `organic hibiscus tea davidson 's teas excellent . really enjoyed hot cold .`.

This is one small-data run, without seed replication, human factuality review or
production validation. The now-published test set must not be called untouched
after using it to tune future experiments.

## Artifacts and reproduction

- [Configuration](run_config.json), [split IDs and hashes](split_manifest.json),
  [training history](training_history.json) and [token statistics](token_statistics.json).
- [Training environment](environment.json) and [full dependency snapshot](environment-freeze.txt).
- [Checkpoint file hashes](checkpoint_sha256.json) and [local archive manifest](local_checkpoint_archive.json).
- [Commands and methodology](../../REPRODUCING.md).

The actual weights and tokenizer are saved locally in a verified ZIP delivered
with this task. They are not committed to Git or publicly hosted. Re-running the
training script produces a reloadable checkpoint. The original assignment
checkpoint is not available. Optimizer state for mid-epoch resume is not saved.

Training started from an uncommitted working tree based on the revision recorded
in `environment.json`; the exact training implementation is preserved in commit
`b549370` and by the recorded file hashes. The local archive also preserves the
exact training and evaluation source files. The later evaluator edit only makes
its bootstrap-description text reflect the configured sample count. Numerical
scoring and the fixed experiment configuration are unchanged.

Validation: **10 offline unit tests pass**, including independently calculated
causal loss, padding versus real EOS, split isolation, generation coverage and
rejection of changed split/configuration artifacts.
