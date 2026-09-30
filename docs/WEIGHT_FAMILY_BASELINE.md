# Per-parameter tensor affine restoration

`BASE-FAMILY-001` tests whether fitting one affine mapping per named parameter tensor improves the fixed scalar global-affine incumbent. The parameter map is derived at runtime from the supplied `TranslationTransformer.named_parameters()` order and contains 68 contiguous tensors totaling 2,961,630 values.

The two equal-length training halves are paired at the same offset only as an explicit unverified-layout hypothesis. The guarded position split is reused from `scripts/weight_baselines.py` (seed 2301, 4096-value blocks, 32-value boundary guards). Coefficients use fitting positions only. Development positions are scored in original units. Audit clean values and hidden test labels are never indexed for metrics.

Each tensor needs at least 64 fitting positions. Otherwise, or if its fitting predictor variance or coefficients are invalid, its prediction falls back to the fitting-split global affine mapping. The JSON result records counts, coefficients, shapes, offsets, hashes, and per-family development RMSE, but no raw weights.

The frozen failure criterion is strict: the aggregate family-calibrated development RMSE must be below the global-affine incumbent `0.4982864015`. Passing this scalar gate would still require a downstream translation check before promotion; it does not establish test or leaderboard performance.

Run with:

```text
.venv/bin/python scripts/weight_family_baseline.py
```

The ignored result is `results/baselines/weight_family_v1.json`.
