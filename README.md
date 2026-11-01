# KronCorpus

**The bilingual English/Russian pretraining corpus of KronLLM Q10** — the
10M-parameter model this project ships (9,938,886 parameters, Kronecker-factorised
weights). It is published separately because the data is useful on its own: a
small, fully documented, science-heavy bilingual corpus with a per-language
quota contract and a provenance record for every shard.

## What is in it

| property | value |
|---|---|
| tokens | **2,783,383,150** train tokens (8K byte-level BPE, **3.60 bytes/token**) |
| text | **10.02 GB**, 933,522 documents (EN 517,973 / RU 415,549) |
| languages | **EN 59.7 % / RU 40.3 % of tokens** — EN 46.5 % / RU 53.5 % of bytes |
| held-out | 46,197,636 tokens, sha256 `a45bbb03…`, **byte-identical since v3** |
| system prompts | **none inserted** (the 46,820 `<|system|>` ids are chat documents' own markup) |
| bins | uint16 (`train.bpe.bin`, `val.bpe.bin`), one epoch = 169,884 steps of 16,384 tokens |
| index | `train_index.npy`: span + token hash per document, for resumable training |
| provenance | source, licence, URL and SHA-256 for **every** shard in `manifest.jsonl` |

**Why bytes and not tokens.** The quota is stated in bytes because that is the
only quantity independent of the tokenizer, and the token share is measured and
published next to it. Russian costs **4.96 bytes/token** against **2.92** for
English, so a byte quota is not a token quota: at 53.5 % of the bytes, Russian is
40.3 % of the tokens. The 60/40 target is therefore enforced **per language**
(`QUOTAS_EN` / `QUOTAS_RU` in `kronllm/bilingual.py`), not on one shared table.

## Composition

Quotas are per language; "realised" is what the mixer actually selected after
dedup. ⚠ marks a domain where the supply cannot fill the quota.

**English**

| domain | quota | realised | main sources |
|---|---:|---:|---|
| mathematics | 26 % | 26.8 % | OpenWebMath, Wikipedia math |
| encyclopedic | 16 % | 16.5 % | Wikipedia EN (3.4 GB of shards) |
| literature | 15 % | 15.5 % | Project Gutenberg |
| code | 11 % | 11.3 % | permissively licensed code corpora, SFT code slices |
| CS / software engineering | 10 % | 10.3 % | Stack Overflow (207,646 answers), IETF RFCs |
| science | 9 % | 9.3 % | arXiv, Wikipedia physics/astronomy |
| machine learning | 7 % | 7.2 % | arXiv cs.LG/CL, Cosmopedia, framework docs |
| logic | 4 % | 2.8 % ⚠ | Wikipedia logic, Lean 4 mathlib |
| humour / identity | 2 % | 0.2 % ⚠ | Fandom, synthetic identity slice |

**Russian**

| domain | quota | realised | main sources |
|---|---:|---:|---|
| encyclopedic | 34 % | 35.5 % | Wikipedia RU |
| CS / software engineering | 21 % | 21.9 % | Habr (49,116 posts), ru Stack Overflow (291,165 answers) |
| literature | 15 % | 15.7 % | Wikisource, Gutenberg translations, FineWeb-2 |
| mathematics | 13 % | 13.6 % | FineWeb-2 math, Wikipedia |
| science | 10 % | 10.4 % | FineWeb-2 science, Wikipedia |
| logic | 2 % | 2.1 % | Wikipedia logic, FineWeb-2 |
| code | 3 % | 0.4 % ⚠ | Russian code tasks with translated questions |
| ML / identity | 1.5 % | 0.3 % ⚠ | FineWeb-2, synthetic identity slice |

Structural shortfalls, reported rather than padded: `en/logic` 131 of 192 MB,
`en/humour` 11 of 48, `en/identity` 1 of 48, `ru/code` 20 of 168, `ru/ml` 17 of
56, `ru/humour` nothing, `ru/identity` 1 of 28. There is no more Russian code
text on the Hub under a usable licence; more would have to be translated.

## How documents are selected

1. **Quality ranking inside each domain.** Every candidate gets a cheap score on
   its first 4,000 characters (alphabetic density, digit ratio, vocabulary
   diversity, word length, length bonus) and the quota takes the **best** ones,
   not a random sample. This is the change that separates this corpus from
   earlier builds, which took documents in SHA-256 order.
2. **Exact dedupe.** sha256 of the whole text plus sha256 of its 2 KB prefix.
   The MinHash/Jaccard pass it replaced cost 55 minutes and 20 GB of RAM.
3. **Two-pass streaming build.** Pass 1 streams all 87 shards and keeps only
   `(sha, lang, domain, bytes, path, offset)` plus the score — peak RSS ~1.1 GB
   instead of the 25 GB that killed the in-memory builder. Pass 2 seeks to the
   selected offsets and writes the mix.
4. **Filters before scoring.** Minimum length, symbol ratio, repetition ratio and
   a script check per language.
5. **Provenance is mandatory.** A shard without a stated licence and a URL is
   rejected by `record_shard`; `manifest.jsonl` is the attribution.

## Sources

| source | licence | role |
|---|---|---|
| Wikipedia (dump 2023-11-01, EN/RU) | CC BY-SA 3.0/4.0 | encyclopedic, domains via keywords |
| OpenWebMath | ODC-By 1.0 | mathematics |
| arXiv (abstracts + full text) | arXiv non-exclusive | science, ML |
| Project Gutenberg, Wikisource | public domain / CC BY-SA | literature |
| FineWeb-2 `rus_Cyrl` | ODC-By 1.0 | Russian domains without a primary source |
| Stack Overflow (EN, CC BY-SA 4.0) | CC BY-SA 4.0 | CS / software engineering |
| Habr, ru Stack Overflow | Habr content / Stack Exchange CC BY-SA 4.0 | Russian CS |
| Lean 4 `mathlib4` | Apache-2.0 | formal proofs (logic) |
| Cosmopedia v2 | Apache-2.0 | ML/science textbook prose |
| Code corpora (permissive) | per file, see the dataset pages | code |
| Fandom wiki API | CC BY-SA 3.0 | humour slice only |

## Decay-pass corpus (Q10A)

A second bin is published next to the epoch one: **809,120,240 tokens** (29.1 %
of an epoch, 249,868 documents) — the same domain profile selected by quality,
plus **5 % chat** spread through the pass so the four chat tokens
(`<|system|>`, `<|user|>`, `<|assistant|>`, `<|eot|>`) are not cold when SFT
starts. Same held-out bin, so both stages are scored on the same blocks.

## Loading

```python
import numpy as np

# Kaggle: the tokenised bins (uint16) live there, GitHub is not a data host
#   kaggle datasets download -d icold21/kron-corpus-2b        # epoch, v10
#   kaggle datasets download -d icold21/kron-q10a-anneal-bin  # decay pass
tokens = np.memmap("train.bpe.bin", dtype=np.uint16, mode="r")
index = np.load("train_index.npy")     # (start, n, hash) per document

# Hugging Face: card, manifest and the identity slice -- not the bins
#   ds = load_dataset("Icold/kron-corpus")
```

## Reproducing the build

```bash
# 1. fetch: append-only, sharded, one timeout per source
python scripts/build_bilingual.py fetch-hf --source all
python scripts/build_bilingual.py fetch-gutenberg-ids ...
python scripts/build_bilingual.py fetch-wikibooks ...
python scripts/gen_identity.py --samples-per-language 800

# 2. mix: quality-ranked quotas, per language (two passes, ~1 GB RSS)
python scripts/build_streaming.py --budget-en-mb 4800 --budget-ru-mb 5600

# 3. bins: reuse the tokenizer and the held-out bin so bpb stays comparable
python scripts/prepare_bilingual_bin.py --out-dir bins \
    --reuse-tokenizer data/bilingual/tokenizer.json \
    --reuse-val-dir held_out --phase epoch --no-domain-prompts
```

The decay pass adds the chat pools (`scripts/build_anneal_chat.py` in the code
repository), the same mix build at 30 % of the budget, and
`--phase anneal --interleave chat`.

## Version history

| version | what changed |
|---|---|
| v10 (current) | prose rebalance: EN literature 15 %, EN encyclopedic 16 %, RU technical up, SFT samples, quality-ranked quotas, 2.78B tokens |
| v1-v9 | earlier rebuilds of the same pipeline. The held-out bin stayed byte-identical from v3 onward; versions 4-7 carried inserted domain system prompts, which v8 removed — the trainer must never spend gradient on its own boilerplate |

## Licence

The corpus is a **collection**: each shard keeps the licence of its source, as
recorded in `manifest.jsonl`. CC BY-SA text requires attribution and share-alike;
Gutenberg texts are public domain in the US; OpenWebMath and FineWeb-2 are
ODC-By 1.0; mathlib4 is Apache-2.0. The build scripts are MIT, like KronLLM.

If you redistribute a mix, keep `manifest.jsonl` next to it — that file is the
attribution.

---

<sub>The `identity` slices are synthetic: the model calls itself *Kron* from a
fixed fact sheet (no numeric facts), 800 plain + 850 pastiche documents per
language, under 0.1 % of the corpus and marked in the manifest.</sub>
