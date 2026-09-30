# Team 23 development baselines — 2026-09-29

These measurements use only the assigned Team 23 files. They are development diagnostics, not Kaggle or final audit scores. Raw arrays, sentence-level outputs, and split IDs stay in ignored local directories.

## Weight restoration

Command: `.venv/bin/python scripts/weight_baselines.py`. The deterministic position-only assignment uses seed 2301, 4096-scalar blocks in each equal-length half, and a 32-scalar guard on each side of a split boundary. Corresponding offsets in the two halves share a split label as a conservative precaution; the halves' provenance is unverified. There are 3,500,736 fitting, 1,140,608 development, 1,230,844 audit, and 51,072 guard positions. The split specification SHA-256 is `68f88dcc5e6c502bd73d5936f3de0a7d9398f5074c0bb4d7b97603fd837b1978`.

| Method | Development RMSE, original units |
| --- | ---: |
| Identity | 0.6502356483 |
| Fitting-split bias correction | 0.6502238480 |
| Fitting-split global affine | 0.4982864015 |

The affine clean prediction is `0.5765571629 * noisy - 0.0009096430`. Coefficients were fitted on fitting positions only. Neither development nor audit labels enter the fit; audit RMSE has not been computed. The separate 0.620521 training-pair identity RMSE in `docs/WEIGHT_PROFILE.md` is a whole-array descriptive statistic and is not comparable as a held-out score. The observed affine gain needs a second split and downstream translation check before promotion.

The paired input hashes are recorded in ignored `results/baselines/weight_v1.json` and `data/23_Team_Toxic/manifest.json`. No clean test weights are available.

## Translation with supplied noisy test weights

Command: `python3 scripts/run_job.py run --id translation-noisy-dev-20260929 --timeout-seconds 600 -- .venv/bin/python -m scripts.translation_noisy_baseline --device mps --max-len 25`. The process and output files were separately validated. Run time was about 9.7 seconds. The output has 45 unique development IDs, no empty predictions, and no audit or test-query rows.

The frozen source-group split has 210 fitting, 45 development, and 45 audit rows; its ignored JSON SHA-256 is `65fddacd277c3bdb87a7f3bb879dda220957ae5f8cdef966f5f3b1ee1abf92bf`. The script preserves the notebook's embedded token IDs, loads the 2,961,630 supplied noisy test weights into the fixed architecture, and decodes by greedy argmax for at most 25 generated tokens after BOS. It does no fine-tuning. MPS evaluation disables PyTorch's unsupported nested-tensor path.

| Corpus metric, 45 development rows | Value |
| --- | ---: |
| BLEU-4 | 0.0689422062 |
| ChrF++ | 2.9290667190 |
| Exact match | 0 / 45 |

SacreBLEU 2.6.0 signatures: BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`; ChrF++ `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`. The very low scores make a useful noisy-weight reference; they do not identify the cause of the degradation. Per-row predictions and exact input hashes are in ignored `results/baselines/translation_noisy_dev.csv` and `.json`.

## Next decision gate

Use the affine baseline as the weight-RMSE incumbent. Train a bounded candidate only with the documented checkpoint contract, compare on the same development positions, and leave both audit sets untouched. Check restored-weight translation quality with this same tokenizer and decoding recipe before selecting a final artifact.

## Bounded candidate outcomes at pause

- Two scalar residual MLP epochs produced development RMSE 0.7490505 and 0.7526667, both worse than global affine 0.4982864. This candidate was rejected without audit evaluation.
- Per-parameter-tensor affine yielded RMSE 0.4982366206, just 0.00004978 below global affine. The equal-half layout hypothesis is unverified; this tiny difference does not warrant promotion.
- Applying the global affine map to the noisy test vector produced a finite, exact-length float32 artifact with SHA-256 `ff432d3fb2badc3d4f5fb4ed9be87e51a476d08e2248a7223bba40fef91ba863` in ignored `results/candidates/affine_v1.npy`. Untuned translation on it scored BLEU-4 0.07165094 / ChrF++ 3.88705355, with 45 nonempty predictions.
- Fine-tuning the fixed Transformer from that affine artifact for five fitting-split epochs yielded development BLEU-4 1.76787586 / ChrF++ 16.95046421 / exact 0/45, with no empty predictions. The preserved checkpoint is ignored `results/training/translation-affine-fit5-20260929/checkpoint_epoch5.pt`. A bounded continuation to ten epochs reached teacher-forced development loss 3.767251; generated translations from that checkpoint remain unevaluated. No audit metric was computed.
