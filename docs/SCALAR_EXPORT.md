# Scalar checkpoint export

`scripts/export_scalar_weights.py` accepts only the corrected schema2 scalar MLP checkpoint from `scripts/train_scalar_mlp.py`. It verifies the checkpoint configuration, compatible model state, finite state values, exact clean/noisy input hashes, and the reconstructed position split including its label hash. The local Team 23 manifest and test-weight hash are also required.

Inference runs in bounded chunks over the one-dimensional float32 test vector. The exporter requires the production count of 2,961,630 values, writes a float32 NPY and an `id,weight` CSV with IDs `0` through `2,961,629` in order, then parses both outputs to verify exact count, finite values, and float32 equality. A JSON sidecar records the checkpoint, input, split, test, NPY, and CSV SHA-256 values. `audit_metrics` remains null because this export has no clean test target.

Example:

```bash
.venv/bin/python -m scripts.export_scalar_weights \
  --checkpoint results/checkpoints/scalar_mlp.pt \
  --npy-output results/candidates/scalar_mlp_restored.npy \
  --csv-output results/candidates/scalar_mlp_submission.csv
```

The checkpoint and raw Team 23 files remain inputs; this procedure does not establish clean-test or leaderboard provenance.
