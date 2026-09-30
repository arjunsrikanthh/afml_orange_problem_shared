# Team 23 weight profile

Descriptive audit only. No model, fit, split, validation target, checkpoint, or submission was created.

## Scope and reproducibility

- Team: `23`; source directory: `data/23_Team_Toxic/`.
- Generated: `2026-09-29T15:31:21.962135+00:00`.
- Python: `3.12.14`; NumPy: `2.5.3`.
- Exact command: `.venv/bin/python scripts/profile_weights.py`
- Inputs were loaded with `numpy.load(..., mmap_mode="r")`; residual calculations were performed in float64 for numerical stability.

## Manifest hash and array checks

| File | Manifest SHA-256 | Observed SHA-256 | Result |
|---|---|---|---|
| `train_weights_clean.npy` | `aa8c6c8b743fc68cc0fa6dd7bb7a9690bbeace57894beec81d8f1d56a044edc4` | `aa8c6c8b743fc68cc0fa6dd7bb7a9690bbeace57894beec81d8f1d56a044edc4` | PASS |
| `train_weights_noisy.npy` | `e5292a4b2f0c43e789f7bb2558d98fc2f11b7e9734eed28e77a78efddeae395a` | `e5292a4b2f0c43e789f7bb2558d98fc2f11b7e9734eed28e77a78efddeae395a` | PASS |
| `test_weights_noisy.npy` | `b751ffee729b96418148a7d7c684d3e8cc08ba46f0e225991f7a3811eb754397` | `b751ffee729b96418148a7d7c684d3e8cc08ba46f0e225991f7a3811eb754397` | PASS |

| Array | Shape | Dtype | Finite / count | Mean | Std | Min | Max |
|---|---:|---|---:|---:|---:|---:|---:|
| `train_weights_clean.npy` | `[5923260]` | `float32` | 5923260 / 5923260 | 0.000932268 | 0.721588 | -5.39217 | 4.87503 |
| `train_weights_noisy.npy` | `[5923260]` | `float32` | 5923260 / 5923260 | 0.00296439 | 0.939931 | -5.45276 | 5.62983 |
| `test_weights_noisy.npy` | `[2961630]` | `float32` | 2961630 / 2961630 | 0.00311562 | 0.941134 | -5.30572 | 5.18473 |

All three arrays have the manifest-declared one-dimensional shapes and float32 dtypes. The finite counts equal the element counts: no NaN or infinity was observed.

### Value quantiles

| Array | q00 | q01 | q05 | q25 | q50 | q75 | q95 | q99 | q100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train clean` | -5.39217 | -2.06986 | -1.29921 | -0.150602 | 0.000172066 | 0.151875 | 1.30038 | 2.06756 | 4.87503 |
| `train noisy` | -5.45276 | -2.65769 | -1.70262 | -0.218943 | 0.000590342 | 0.22293 | 1.71092 | 2.66338 | 5.62983 |
| `test noisy` | -5.30572 | -2.66008 | -1.70942 | -0.225923 | 0.00133615 | 0.234564 | 1.71032 | 2.66004 | 5.18473 |

## Training clean/noisy residual diagnostics

Residual definition: `r = train_weights_noisy - train_weights_clean`, aligned by array position.

| Metric | Value |
|---|---:|
| Count | 5923260 |
| Residual mean (bias) | 0.00203212 |
| Residual standard deviation (scale) | 0.620518 |
| Mean absolute residual | 0.389312 |
| Identity RMSE | 0.620521 |
| Noisy/clean Pearson correlation | 0.751292 |
| Noisy-on-clean affine diagnostic slope | 0.978623 |
| Noisy-on-clean affine diagnostic intercept | 0.00205205 |

Residual quantiles:

| Statistic | q00 | q01 | q05 | q25 | q50 | q75 | q95 | q99 | q100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Residual | -4.10308 | -1.76486 | -1.11678 | -0.142704 | 0.000477087 | 0.145534 | 1.12478 | 1.77071 | 4.10989 |
| Absolute residual | 0 | 0.0020237 | 0.0102016 | 0.0539173 | 0.144106 | 0.606487 | 1.42804 | 1.9923 | 4.10989 |

- Fraction with `abs(r) > 2 * residual_std`: `0.0772782`.
- Fraction with `abs(r) > 3 * residual_std`: `0.015127`.
- Maximum absolute residual: `4.10989`.

### Global lag correlation

Residual Pearson correlations between `r[:-lag]` and `r[lag:]` over the entire flattened vector:

| Lag | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Correlation | 0.357865 | 0.236888 | 0.00280416 | 0.00193687 | 0.00376798 |

These are global flattened-array diagnostics. They can be driven by concatenation boundaries, repeated tensor layouts, or long-range position effects; they do not establish local stochastic dependence within a model tensor and do not justify a denoiser or generalization claim.

## Train/test noisy distribution comparison

The test clean target is unavailable, so this section compares only the observed noisy training and noisy test arrays.

| Metric | Value |
|---|---:|
| Test noisy mean minus train noisy mean | 0.000151233 |
| Test noisy std / train noisy std | 1.00128 |
| Test fraction below train noisy q01 | 0.0100451 |
| Test fraction above train noisy q99 | 0.00994047 |

Differences in corresponding test-minus-train noisy quantiles:

| Quantile | q00 | q01 | q05 | q25 | q50 | q75 | q95 | q99 | q100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Difference | 0.147042 | -0.00239404 | -0.00679944 | -0.00698035 | 0.000745811 | 0.0116339 | -0.000597256 | -0.00334183 | -0.445092 |

## Limits and interpretation

- The identity RMSE is a descriptive training-pair diagnostic, not a validation or test score.
- This descriptive audit did not use a validation split. A separate guarded development split and baseline evaluation are recorded in `docs/BASELINES.md`. No clean test target exists, so no leaderboard or denoiser claim is supported here.
- Flattened lag correlations ignore verified tensor boundaries and parameter semantics.
- Aggregate distributions can hide tensor-family or magnitude-dependent behavior; this audit does not infer the model layout.
- Raw array values were not written to the report; only aggregate statistics and hashes are recorded.
