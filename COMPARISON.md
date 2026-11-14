# Reproduce the T5/GPT-2 comparison

Use the Python 3.12 environment and pinned dependencies in [REPRODUCING.md](REPRODUCING.md).
No extra T5 package is required: the pinned repository provides a fast tokenizer JSON.
See [results and limitations](results/t5-comparison/REPORT.md).

1. Reproduce or obtain the original verified GPT-2 run, including its checkpoint,
   split manifest, configuration, environment and checkpoint hashes. Commands below
   call it `runs/pilot-2026-11-13`.
2. Download the pinned public T5-small files into the same `HF_HOME` used by training:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "google-t5/t5-small",
    revision="df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
    allow_patterns=["config.json", "generation_config.json", "model.safetensors",
                    "tokenizer.json", "tokenizer_config.json", "spiece.model"],
)
```

On Windows without symlink support, Hugging Face normally copies cache files.
For this machine's download, its symlink-support probe was overridden to return
false, so all six files were downloaded as ordinary files. No model code was
downloaded or executed with `trust_remote_code`.

3. Run all offline correctness and evidence-integrity tests:

```bash
python -m unittest discover -s tests -v
python -m pip check
```

The completed experiment has 22 passing tests, including an independent T5 loss
calculation, padding/EOS, split isolation, identity-checked evaluation resume,
recorded-score recomputation and complete audit denominators.

4. Reproduce the frozen protocol and training in **new output directories**:

```bash
python comparison_data.py --old-run runs/pilot-2026-11-13 --output runs/t5-reproduction/protocol
python train_t5.py --protocol runs/t5-reproduction/protocol --gpt2-run runs/pilot-2026-11-13 --output runs/t5-reproduction/training
python compare_summarizers.py validation --protocol runs/t5-reproduction/protocol --gpt2-run runs/pilot-2026-11-13 --t5-run runs/t5-reproduction/training --output runs/t5-reproduction/evaluation
python compare_summarizers.py test --protocol runs/t5-reproduction/protocol --gpt2-run runs/pilot-2026-11-13 --t5-run runs/t5-reproduction/training --output runs/t5-reproduction/evaluation --facts results/t5-comparison/detail_facts.json
```

Training and evaluation use `local_files_only=True`. GPU is the default; `--device cpu`
is available, but timing and numeric results need not be identical across hardware.
These commands reproduce the **already observed** fixed experiment. They do not
create a new untouched test set. Changing the seed alone is insufficient for future
claims: the preparation code's historical exclusions cover the earlier pilots;
future experiment code must additionally exclude this comparison's observed data.

T5 saves the selected inference checkpoint, not optimizer/RNG state for mid-training
resume. If training is interrupted, restart in a new output directory. Validation
and test evaluation save each completed batch atomically and can resume by repeating
the same command under unchanged identities. A completed result is never overwritten.

The final-test command requires the predeclared fact labels and locks their hash,
the selection result, checkpoint hashes, source hashes and environment before
generating anything. References are passed to scoring only; generation functions
receive review inputs, not summary targets. Labels do not influence model selection.

5. Inspect the manually authored review and recompute the published counts:

```bash
python score_detail_audit.py --output runs/detail-scores-recomputed.json
```

To expand the compact annotation file into a new inspectable review artifact:

```bash
python score_detail_audit.py --compile-annotations results/t5-comparison/audit_annotations.json --audit runs/detail-audit-recompiled.json --output runs/detail-scores-recompiled.json
```

This expands explicit agent labels; it is **not an automatic factuality judge**.
New predictions need new source-grounded review. Tests and byte hashes protect
artifact integrity, not the correctness of subjective annotation decisions.

Checkpoint binaries remain outside Git. The local T5 archive includes weights,
tokenizer, exact source, configuration, protocol, results and dependency versions;
its hash is recorded in `results/t5-comparison/local_checkpoint_archive.json`.
The original GPT-2 archive is still required to reproduce both systems without
retraining GPT-2. No paid inference provider is called.
