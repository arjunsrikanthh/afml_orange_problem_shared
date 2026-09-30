# Team 23 loader and tokenizer audit

Checked on 2026-09-29 against the six files in `data/23_Team_Toxic/`. The input manifest and machine-readable parameter table are ignored local artifacts at `data/23_Team_Toxic/manifest.json` and `reports/generated/loader_audit.json`. Reproduce with `.venv/bin/python scripts/audit_loader.py`.

## Weight contract: passed

- Assigned notebook SHA-256: `d99a021e8747bb9b69248ed586c664eda02fb56aa1fa5472e287b770c5a73d6c`.
- `TranslationTransformer` from the supplied notebook has 68 named parameter tensors totaling **2,961,630** float32 values, matching the Team 23 test vector exactly. Its `state_dict` has one additional non-parameter positional-encoding buffer, `pos_encoder.pe` with shape `(1, 256, 128)`.
- Parameter names and order match `state_dict` order after excluding that buffer. The notebook loader iterates `named_parameters()`; the saved audit records each name, shape, count, and offset.
- Loading the actual noisy test vector and flattening the model reproduced every float32 value exactly. A deterministic full-length sentinel vector also round-tripped exactly. The audit checks the length before calling the starter's permissive loader, which otherwise silently accepts a short or long vector.
- The supplied causal mask has zero on/below its diagonal and negative infinity above it. A small padded forward pass produced finite `(2, 4, 5982)` logits on CPU.

## Tokenizer contract: starter defect, corrected adapter passed

Both embedded vocabularies assign `<pad>=0`, `<bos>=1`, `<eos>=2`, `<unk>=3`. The starter code instead initializes `<pad>=0`, `<unk>=1`, `<bos>=2`, excludes `<eos>` from the loaded vocabulary, and falls back to `eos_id=bos_id=2`. Its generated sequence begins with ID 2; its `decode()` treats that first ID as EOS and returns an empty string. The starter as supplied is unsafe for translation evaluation.

`scripts/team23_tokenizer.py` reads the embedded maps from the assigned notebook and preserves their original IDs and tokenization rules. Source and target maps have 5,897 and 5,932 entries, respectively; the fixed architecture has 5,947 and 5,982 embedding rows. Every observed training source token, training English token, and test source token is in the embedded maps (zero OOV in each category), and all mapped IDs fit their embedding tables. The 50 extra embedding rows per side remain part of the supplied architecture; their purpose is not inferred.

Before any fine-tuning or final notebook run, use the corrected tokenizer in the translation pipeline. Preserve the supplied model dimensions and parameter order. The raw boilerplate remains unchanged.

## MPS execution check

PyTorch 2.14.0 reported MPS available. The default evaluation path failed because `aten::_nested_tensor_from_mask_left_aligned` is unavailable on MPS. Setting `model.transformer.encoder.use_nested_tensor = False` after construction allowed the same small forward pass on MPS; its maximum absolute logit difference from CPU was about `1.64e-6`. This is a runtime execution setting and adds no parameters. A mismatched mask type warning remains; resolve it in the eventual evaluation path without changing the fixed model architecture. This check is a forward smoke test, not a training benchmark or resource budget.

## Measured short training-step smoke

One synthetic-token optimizer step loaded the assigned noisy test weights, used the corrected IDs, and ran the supplied model on MPS through the bounded runner (`results/jobs/translation-mps-smoke-20260929/`). Batch size was 32 and sequence length 17. The step completed in 2.917 seconds with finite loss; process peak RSS was 495,517,696 bytes, and MPS allocation after the step was 60,128,000 bytes. No actual corpus label, validation score, checkpoint, or trained candidate was produced. This is a resource smoke, not an end-to-end training estimate.

## Remaining gates

- Use the frozen 210/45/45 translation split at `data/23_Team_Toxic/translation_split.json`; its corpus hash matches the input manifest. Construct a separate leakage-aware weight split.
- Require a complete checkpoint contract before long training. The one-step smoke does not validate resume or a full-epoch time estimate.
- Verify a corrected tokenizer in a runnable translation path and then compare noisy/restored candidates only on the fixed local splits.

PyTorch references: [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html), [TransformerEncoder source and nested-tensor path](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py).
