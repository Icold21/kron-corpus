# KronCorpus

**A 2B-token bilingual English–Russian pretraining corpus with a science accent.**

KronCorpus is the training corpus behind [KronLLM](https://github.com/Icold21/kron_llm),
a compact language model whose weights are Kronecker products. It is published
separately because the data is useful on its own: a small, fully documented,
science-heavy bilingual corpus for anyone training a model in the 10M–100M
parameter range.

---

## What is in it

| property | value |
|---|---|
| tokens | **~2.0B** (target; fetched text ≈ 7.3 GB at 3.4 bytes/token) |
| languages | **60 / 40 EN / RU by bytes** — stated in bytes on purpose, see below |
| documents | ~790k, deduplicated (MinHash, Jaccard 0.8) |
| tokenizer | 8K byte-level BPE, bilingual, trained on the mix |
| held-out | 2 % per language, disjoint by document |
| provenance | source, licence, URL and SHA-256 for **every** shard in `manifest.jsonl` |

**Why bytes and not tokens.** Cyrillic costs two UTF-8 bytes per character, so
a 40 % Russian share by bytes is roughly a third of the tokens. We fix the
contract in bytes because it is the only quantity that does not depend on the
tokenizer; the token share is reported next to it, and the tokenizer's
fertility per language is measured in `tokenizer_meta.json`.

## Composition

Domain quotas are the design contract; the "fetched" column is what the
sources actually yielded before the final mix.

| domain | quota | fetched | main sources |
|---|---:|---:|---|
| mathematics | 25 % | 26 % | OpenWebMath, Wikipedia math |
| science (physics, astronomy, cosmology) | 14 % | 22 % | arXiv, Wikipedia, curated cosmology articles |
| code | 15 % | 14 % | CodeParrot Clean, CPython stdlib |
| machine learning | 12 % | 5 % | arXiv cs.LG/CS.CL abstracts, textbooks, framework docs |
| literature | 11 % | 13 % | Project Gutenberg, Wikisource, FineWeb-2 |
| logic | 10 % | 3 % | Wikipedia logic, Lean 4 mathlib proofs |
| encyclopedic | 9 % | 17 % | Wikipedia EN/RU |
| CS / software engineering / engineering | 3 % | 3 % | Wikipedia (curated categories: microprocessors, VLSI, XP, agile) |
| humour / satire | 1 % | <1 % | Project Gutenberg, Fandom transcripts |
| self-identity | 1 % | <0.1 % | synthetic, prompt-constrained |

The table is honest about the two hard domains: **logic and ML are short**.
Logic text at scale barely exists outside formal-proof repositories, and ML
prose is mostly behind publisher paywalls. The mixer reports both as
shortfalls instead of padding them with mislabelled text.

## How sources were selected (the priority order)

1. **Verifiable provenance beats volume.** A source is admitted only if its
   licence is stated and a URL can be recorded. Every shard carries
   `source / licence / url / sha256` in `manifest.jsonl`; nothing is anonymous.
2. **Primary sources over scrapes.** Wikipedia dumps, arXiv, Project Gutenberg,
   OpenWebMath and official repositories come first; web text (FineWeb-2) is
   used only where a domain has no primary source, and its shards are marked as
   keyword-routed.
3. **Science and mathematics first.** The quotas are not balanced for
   "general English": a quarter of the corpus is mathematics, and physics,
   astronomy and cosmology get an explicit slice of the science budget.
4. **Quality filters before volume.** Minimum length, symbol ratio, repetition
   ratio, script check per language, and a MinHash near-duplicate pass
   (Jaccard 0.8) that removes translated mirrors and boilerplate.
5. **Domain labels are declared as heuristics** where they are heuristics.
   Wikipedia and FineWeb shards are routed by subject keywords, and the
   manifest says so. Curated lists (cosmology, chips/XP) are explicit.
6. **Leave room for the model's own name.** 1 % is reserved for a synthetic
   self-identity slice; it is generated from a fixed fact sheet, not scraped.

## Sources

| source | licence | role |
|---|---|---|
| Wikipedia (HF dump 2023-11-01, EN/RU) | CC BY-SA | encyclopedic, domains via keywords |
| OpenWebMath | ODC-By 1.0 | mathematics |
| arXiv (abstracts + full text) | arXiv non-exclusive | science, ML |
| Project Gutenberg (pglaf mirror) | public domain | literature, satire |
| Wikisource | CC BY-SA | Russian literature |
| CodeParrot Clean | permissive, per file | code |
| FineWeb-2 `rus_Cyrl` | ODC-By 1.0 | Russian domains without a primary source |
| Lean 4 `mathlib4` | Apache-2.0 | formal proofs (logic) |
| Cosmopedia v2 | Apache-2.0 | ML/science textbook-style prose |
| Fandom wiki API | CC BY-SA 3.0 | easter-egg slice only |

## Loading

The tokenised bins (uint16, ~4 GB) are distributed outside git — GitHub is not
a data host.

```python
# Kaggle (private dataset, mount and read the memmap)
#   kaggle datasets download -d icold21/kron-corpus-2b
import numpy as np
tokens = np.memmap("train.bpe.bin", dtype=np.uint16, mode="r")

# Hugging Face (once published)
#   from datasets import load_dataset
#   ds = load_dataset("Icold21/kron-corpus-2b")
```

The raw, deduplicated text shards and the exact mix are reproducible from this
repository: `scripts/` contains the fetchers, the mixer, the tokenizer training
and the bin preparation. See `docs/composition.md` for measured numbers and
`docs/rebuild.md` for the command sequence.

## Reproducing the build

```bash
python scripts/build_bilingual.py fetch-hf --source all      # Wikipedia, arXiv, OpenWebMath, FineWeb-2
python scripts/build_bilingual.py fetch-gutenberg-ids ...    # curated public-domain books
python scripts/build_bilingual.py fetch-tarball ...          # Lean mathlib4
python scripts/gen_identity.py --samples-per-language 800    # self-identity slice
python scripts/build_bilingual.py build --budget-en-mb 4300 --budget-ru-mb 2870
python scripts/prepare_bilingual_bin.py --vocab 8192
```

Fetches are append-only and sharded; `build` is deterministic given the same
shards (dedup walks documents in SHA-256 order).

## Licence

The corpus is a **collection**: each shard keeps the licence of its source, as
recorded in `manifest.jsonl`. CC BY-SA text (Wikipedia, Wikisource, Fandom)
requires attribution and share-alike; Gutenberg texts are public domain in the
US; OpenWebMath, FineWeb-2 and peS2o are ODC-By 1.0; mathlib4 is Apache-2.0.
The build scripts themselves are MIT, like KronLLM.

If you redistribute a mix, keep `manifest.jsonl` next to it — that file is the
attribution.

---

<sub>Two slices in `data/easter_eggs/` and `data/identity/` are easter eggs: a
synthetic self-identity set where the model calls itself *Kron*, and 1.8 MB of
animated sci-fi dialogue. Both are under 0.1 % of the corpus and marked in the
manifest.</sub>
