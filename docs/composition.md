# Composition, measured

Numbers below are **text bytes after cleaning**, not file sizes (JSONL adds
~10 % of metadata). The final mix is produced by `build`, which applies
deduplication and the quotas; the "fetched" column is the pool it draws from.

## Fetched pool

| | EN | RU | total |
|---|---:|---:|---:|
| text | 4.9 GB | 2.4 GB | **7.3 GB** |
| documents | 690k | 105k | **795k** |

At the measured 3.34 bytes/token on the 8K bilingual BPE this is ≈ 2.15B
tokens before dedup and quota selection.

## By domain

| domain | fetched share | notes |
|---|---:|---|
| mathematics | 26 % | OpenWebMath is the bulk; Wikipedia math adds definitions |
| science | 22 % | arXiv full text dominates; cosmology curated separately |
| encyclopedic | 16 % | Wikipedia EN/RU |
| code | 14 % | CodeParrot Clean + CPython stdlib |
| literature | 11 % | Gutenberg novels, Russian classics from Wikisource |
| ML | 5 % | arXiv abstracts, textbooks, framework docs |
| logic | 2 % | Lean mathlib proofs + Wikipedia logic |
| CS / engineering | 3 % | curated Wikipedia categories (chips, VLSI, XP, agile) |
| humour | <1 % | Gutenberg satire, transcripts |
| identity | <0.1 % | synthetic |

## Language ratio

60/40 **by bytes** is the contract. Because Cyrillic costs ~1.8× more bytes per
character, the same split is ≈ 55/45 by bytes but only ≈ 33 % Russian by
tokens with the 8K BPE. Both numbers are reported in `tokenizer_meta.json`;
the byte contract is the stable one, the token share depends on the tokenizer.

## Shortfalls (kept visible)

* **logic** — 10 % quota, ~2 % realised. Formal proofs exist (mathlib), but
  general logic prose is thin.
* **ML** — 12 % quota, ~5 % realised. arXiv abstracts are high quality but
  short; textbooks are paywalled.
* **humour** — 1 % quota, ~1.5 MB realised. Public-domain satire and dialogue
  are small by nature.
* **Russian code** — the code quota is filled with English source; Russian
  code comments are not a coherent corpus of their own.

The mixer emits these as `shortfalls` in `corpus_report.json` rather than
padding the quota with mislabelled text.

## Dedup and held-out

* MinHash, 64 permutations, 5-shingles, Jaccard threshold 0.8.
* Held-out: 2 % of each (language, domain) bucket, at most one document
  boundary of overshoot.
* Buckets under 5 MB (identity, rare slices) are **not** split, so they stay in
  training in full.
