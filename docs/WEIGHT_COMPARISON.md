# Paired weight candidate comparison

`scripts/compare_weight_candidates.py` compares the fitting-only global affine
mapping with one or more already-produced schema2 `ScalarResidualMLP`
checkpoints. It is an evaluation script: it does not train, tune architecture
or seeds, create checkpoints, or read clean audit targets.

The script regenerates the existing position-only split from the same seed,
block size, and guard size used by `weight_baselines.py` and
`train_scalar_mlp.py`. It validates the checkpoint schema, strict model state,
input SHA-256 values, split specification, split-label SHA-256, progress fields,
and finite model tensors. Legacy schema1, malformed, non-finite, or mismatched
checkpoints fail closed.

Development positions are paired by index: both candidates are evaluated on the
same clean development values. A block is a contiguous development run in one
half together with the same-offset run in the other half when the frozen split
has equal linked halves. For each linked block the report contains count, SSE,
MSE, the MLP-minus-affine MSE difference, and the winner. Global RMSE is computed from
the sum of all development SSE divided by the total development count; it is not
the arithmetic mean of block RMSE values. Wins and losses are block counts.

The bootstrap resamples contiguous blocks with replacement using a declared seed;
same-offset blocks in the two equal-length halves are resampled together. Its
statistic is the count-weighted MLP RMSE minus affine RMSE for each resample,
with a percentile 95% interval. Positions within a block and corresponding
positions in the two halves can be dependent, so the interval is descriptive
and does not claim independent-position confidence coverage.

Example, using existing ignored checkpoints:

```sh
.venv/bin/python -m scripts.compare_weight_candidates \
  --checkpoint results/reproduction/scalar_seed2301/checkpoint.pt \
  --checkpoint results/reproduction/scalar_seed2302/checkpoint.pt \
  --bootstrap-seed 2301 --bootstrap-repetitions 2000
```

The JSON report is printed to stdout and includes checkpoint hashes, clean/noisy
input hashes, split and label hashes, candidate metrics, paired block details,
bootstrap settings, and `audit_metrics: null`. Synthetic tests in
`tests/test_compare_weight_candidates.py` cover count-weighted RMSE, paired
block wins/losses, schema2 and legacy validation, and audit exclusion.
