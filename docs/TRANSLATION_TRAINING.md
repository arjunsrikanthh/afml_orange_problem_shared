# Bounded translation fine-tuning

`scripts/train_translation.py` fine-tunes the supplied fixed
`TranslationTransformer` from the noisy Team 23 weights. It uses only IDs in
`data/23_Team_Toxic/translation_split.json`: fitting IDs provide optimization
labels and development IDs provide teacher-forced validation loss. Audit IDs
are recorded in the checkpoint contract, while audit English is discarded when
the corpus is parsed and never retained by the trainer.
The trainer does not run generation or compute BLEU/ChrF++.

The target sequence is encoded as BOS, tokens, EOS. Each batch passes
`target[:-1]` to the decoder and scores `target[1:]`; target padding is ignored
by cross entropy. The embedded vocabulary IDs from the notebook are preserved.
The supplied architecture and parameter count are checked through
`prepare_model`. Nested tensors are disabled on MPS because this encoder path
is unsupported there.

## Smoke run

Use the local environment and the bounded runner. This performs only two
fitting batches and one epoch, then writes an atomic checkpoint and summary:

```sh
python3 scripts/run_job.py run --id translation-trainer-smoke-001 \
  --timeout-seconds 600 -- .venv/bin/python scripts/train_translation.py \
  --smoke --epochs 1 --batch-size 2 \
  --output-dir results/training/translation-smoke-001
```

## Bounded training and resume

Set an explicit finite epoch count. A real run must still be one bounded
`run_job.py` job at a time:

```sh
python3 scripts/run_job.py run --id translation-trainer-001 \
  --timeout-seconds 7200 -- .venv/bin/python scripts/train_translation.py \
  --epochs 3 --batch-size 8 --output-dir results/training/translation-001
```

Resume only from the matching `checkpoint_last.pt` and use the same inputs,
split, seed, batch size, learning rate, device, and smoke settings. A resume
may write to a fresh empty output directory, which is useful when preserving an
immutable parent run. Existing non-empty directories are rejected unless they
contain only the current `checkpoint_last.pt` and `summary.json` for a
same-directory continuation. Resume is fail-closed when the trainer contract,
model state, input hashes, split IDs, or configuration differs:

```sh
.venv/bin/python scripts/train_translation.py --epochs 5 \
  --output-dir results/training/translation-001 \
  --resume results/training/translation-001/checkpoint_last.pt
```

Every newly written checkpoint contains model, optimizer, scheduler, epoch,
step, Python / NumPy / Torch / available accelerator RNG states, configuration,
input hashes, all split IDs, and a lineage block. The lineage records the source
Git revision and, for a continuation, the parent checkpoint path, SHA-256,
epoch, step, and parent summary path/SHA-256 when available. Source provenance
also records Git availability/dirty state and SHA-256 hashes of the trainer and
supporting source files, so a dirty or Git-free environment is explicit rather
than being misrepresented by a bare revision. New checkpoints also carry
cumulative prior history, filtered to the parent epoch/step and accompanied by
history completeness and gap information. Legacy v1 checkpoints without
lineage remain loadable; their prior summary is linked only for validated
records that belong to the resumed checkpoint.
Checkpoints are written through a temporary file and atomic replacement. The
summary reports fitting/development counts, cumulative and current-run
validation history, and the same lineage; it contains no audit English.

Synthetic coverage is in `tests/test_train_translation.py` and checks shifted
targets, padding behavior, deterministic batches, checkpoint round-trip, and
fail-closed resume. The coordinator should run the tests and inspect the
checkpoint before any Team 23 smoke or longer job.
