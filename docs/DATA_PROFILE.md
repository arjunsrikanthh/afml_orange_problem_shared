# Team 23 corpus/query profile

Audit scope: `data/23_Team_Toxic/train_parallel_corpus.csv`,
`data/23_Team_Toxic/test_queries.csv`, and `data/23_Team_Toxic/manifest.json`.
This report contains aggregate evidence only; it intentionally excludes sentence
text and identifier values.

## Input and manifest checks

- The manifest lists all six required Team 23 files. All six were present.
- SHA-256 checks for all six files matched the manifest.
- The CSV headers, row counts, and unique-ID counts matched the manifest entries.
- CSVs were parsed with UTF-8 (BOM accepted), standard CSV quoting, and one header
  row. No parse error was observed.

## Schema and counts

| File | Header fields | Data rows | Unique IDs | Empty IDs | Duplicate ID groups / extra rows |
|---|---|---:|---:|---:|---:|
| `train_parallel_corpus.csv` | `id`, `asdfghjkl`, `english` | 300 | 300 | 0 | 0 / 0 |
| `test_queries.csv` | `id`, `asdfghjkl` | 200 | 200 | 0 | 0 / 0 |

### Empty values

| File | Empty source values | Empty English values | Rows with an empty field |
|---|---:|---:|---:|
| Training corpus | 0 | 0 | 0 |
| Test queries | 0 | not applicable | 0 |

## Duplicate and overlap checks

The training corpus has 300 unique source strings and 300 unique English strings.
There are no exact duplicate source groups, exact duplicate English groups, or
exact duplicate source/English pair groups. All 300 source/English pairs are
distinct.

The test file has 200 unique source strings and no exact duplicate source groups.

Source overlap was checked in both directions at the unique-value level and at
the test-row level:

| Comparison | Unique train sources | Unique test sources | Intersection | Test rows matching train |
|---|---:|---:|---:|---:|
| Exact source text | 300 | 200 | 0 | 0 |
| NFKC + whitespace collapse + casefold | 300 | 200 | 0 | 0 |
| Same normalization plus digit-run masking and long hex-like-token masking | 300 | 200 | 0 | 0 |

The normalized source/English pair check also produced 300 unique normalized
pairs, with zero collision groups and zero extra rows.

## Length profile

Lengths are measured in Unicode code points for characters and whitespace-delimited
items for tokens. Percentiles use linear interpolation.

| Field | File | Min | P25 | Median | P75 | Max | Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Source characters | Train | 32 | 56.00 | 68.00 | 80.75 | 104 | 68.91 |
| Source tokens | Train | 5 | 9.00 | 10.00 | 12.25 | 15 | 10.59 |
| English characters | Train | 28 | 44.00 | 53.00 | 63.00 | 91 | 53.98 |
| English tokens | Train | 5 | 9.00 | 10.00 | 12.25 | 15 | 10.59 |
| Source characters | Test | 38 | 59.00 | 71.00 | 83.00 | 107 | 70.18 |
| Source tokens | Test | 6 | 9.00 | 11.00 | 12.25 | 15 | 10.79 |

No source or English value changed under Unicode NFKC normalization in this
audit.

## Near-template and normalization diagnostics

Two conservative diagnostics were run without retaining collision members in the
report:

1. Canonical normalization: Unicode NFKC, whitespace collapse, and casefold.
   It produced no within-file source collisions, no train/test source overlap,
   and no normalized pair collisions.
2. Template masking: the same normalization followed by masking digit runs and
   long hex-like tokens. It produced no within-training source collisions and no
   train/test source overlap.

These are observed diagnostics, not a semantic similarity or paraphrase detector.
Zero collisions under these masks does not rule out close paraphrases, shared
unmasked slots, or leakage through a broader template family. A later audit may
add a reviewed tokenizer or similarity method, but its threshold and false-match
rate must be recorded before using it for split construction.

## Recommended fixed, leakage-aware validation split

Do not create split IDs as part of this profile. For the later translation audit,
reserve approximately 45 rows for development and 45 rows for a final untouched
audit, leaving about 210 for fitting. Construct groups before assigning rows,
using at least:

- exact source text;
- the canonical normalized source;
- the conservative template diagnostic above; and
- any manually reviewed close-template/paraphrase groups found by a subsequent
  audit.

Assign complete groups, rather than individual rows, to fitting, development,
or audit with a deterministic seed and a documented rule. Balance only on
source-side diagnostics such as encoded-source length buckets and source-vocabulary
coverage; do not use held-out English text to choose the split. Freeze the
assignment before model comparison, select on development only, and evaluate the
frozen finalist once on audit. With 300 pairs, report paired errors and
uncertainty instead of treating small metric differences as decisive.

## Exact checks and limitations

- Checks executed: manifest presence and SHA-256 agreement; CSV header/row/ID
  agreement; empty-field counts; exact duplicate IDs, sources, English strings,
  and source/English pairs; normalized collisions; conservative template
  collisions; train/test source overlap; character and whitespace-token length
  summaries.
- This audit covers the two CSVs only. It does not validate the Transformer
  tokenizer, vocabulary IDs, masks, weight layout, or model-loading contract.
- The English field exists only in the training corpus; test translation targets
  are unavailable here.
- Whitespace-token counts are a diagnostic, not the model tokenizer's token count.
- The template rule is intentionally narrow and can miss semantic or structural
  similarity. No split assignments, training, or model evaluation were performed.
