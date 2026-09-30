# One-time frozen candidate audit

`scripts/audit_frozen_candidate.py` consumes a schema 1 freeze document and
performs one fail-closed audit of the selected Team 23 candidate. Invoke it as:

```sh
python3 -m scripts.audit_frozen_candidate \
  --freeze configs/frozen_recipe_v1.json \
  --output-dir results/final_audit \
  --device mps
```

The tool first validates all metadata, source/artifact/input hashes, checkpoint
contracts, split IDs, and recovery history without indexing held-out targets.
It then atomically claims `results/audit_ledger/<freeze-sha256>.json` with
exclusive creation before creating the requested output directory or indexing a
held-out target. Any consumed, started, or failed claim is refused even when a
different `--output-dir` is supplied. Existing output directories and raw-data
paths are refused.

The freeze status, exact supported runtime module set, recipe, six raw input
hashes, artifact hashes, and runtime source hashes are
checked first. Scalar checkpoint schema 2/fixed split validation and the
translation checkpoint contract, completed scalar epoch 2, translation epoch
1–20 through step 540, split IDs, lineage, and initialization hash are then
checked. The exact seven scalar, five translation and two decoding fields are required;
unknown, omitted, changed or wrongly typed recipe values are rejected. A preflight failure creates no claim
and consumes no audit targets.

After those gates, the validated scalar model is applied to the full noisy
training vector only at label 2 positions. Scalar RMSE is accumulated as total
squared error divided by the total count over those positions, in chunks and
original units on CPU. The exported test vector is never used for a clean-target
audit score; its clean targets are unavailable. Translation uses the requested
validated backend
and exactly the frozen audit IDs, the existing greedy argmax decoder with
`max_len=25`, and the existing raw-reference SacreBLEU BLEU-4, ChrF++, and exact
metrics. `translation_audit.csv` contains only `id,source,reference,prediction`.
The report records all input/artifact/runtime hashes, metric versions and
signatures, empty/cap/repetition diagnostics, environment, and ledger state.

The test suite uses synthetic arrays and mocked contract validation only. It does
not read Team 23 audit labels, run model evaluation, train, or select a method.
