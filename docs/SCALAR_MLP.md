# Scalar residual MLP trainer

`scripts/train_scalar_mlp.py` defines a small CPU/MPS-compatible residual MLP:

```text
noisy scalar -> Linear(1, hidden) -> Tanh -> Linear(hidden, 1) -> residual + noisy scalar
```

It uses the fixed position-only assignment from `scripts/weight_baselines.py`.
Only fitting rows contribute gradients. Development RMSE is computed in original
weight units after each bounded run. Audit clean values are never indexed by the
trainer and `audit_metrics` is always `None`.

## Corrected objective (checkpoint schema 2)

Version 1 returned column predictions against vector targets, silently broadcasting the training loss to all batch pairs. Its near-mean checkpoint is invalid evidence about scalar model capacity and must not be resumed. Version 2 preserves input shape and requires exact prediction/target shape agreement before paired MSE. Architecture, optimizer, schedule and split are unchanged for the controlled rerun. Regression tests distinguish aligned loss from all-pairs loss and reject legacy v1 resumes.

## Bounded command

After the data preflight has passed, a deliberately small run is:

```bash
.venv/bin/python scripts/train_scalar_mlp.py \
  --epochs 2 --max-epochs-this-run 1 --batch-size 4096 --hidden 32 \
  --checkpoint results/checkpoints/scalar_mlp.pt
```

Resume is explicit and fail-closed:

```bash
.venv/bin/python scripts/train_scalar_mlp.py \
  --resume --epochs 2 --max-epochs-this-run 1 \
  --checkpoint results/checkpoints/scalar_mlp.pt
```

The checkpoint contains model, optimizer, scheduler, epoch/step, Python,
NumPy, Torch and available MPS RNG states, configuration, input file hashes,
split hash and split counts. It is written to a temporary file in the target
directory and atomically replaced. Resume rejects missing fields, changed
configuration, changed input hashes, changed split labels, incompatible model
state, or unavailable MPS state. One epoch is the default run chunk; the
total `--epochs` value must remain fixed across resume calls. Epoch-specific
batch permutations give the same order after resume as an uninterrupted run.

The arrays are memory mapped by the CLI. Training materializes only each batch;
development evaluation materializes only one evaluation chunk. The default
configuration is bounded, but runtime depends on the Mac, PyTorch build, MPS
availability, and the number of fitting scalars. The historical Team 23 v1 run is retained as a rejected artifact because it used
the broadcast objective. Record corrected v2 smoke and candidate results separately.

Run the synthetic tests with:

```bash
.venv/bin/python -m unittest tests.test_train_scalar_mlp -v
```
