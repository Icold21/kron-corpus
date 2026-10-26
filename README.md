# KronCorpus

**A bilingual English–Russian pretraining corpus with a science accent.**

KronCorpus is the training corpus behind [KronLLM](https://github.com/Icold21/kron-llm),
a compact language model whose weights are Kronecker products. It is published
separately because the data is useful on its own: a small, fully documented,
science-heavy bilingual corpus for anyone training a model in the 10M–100M
parameter range.

---

## What is in it

| property | value |
|---|---|
| tokens | **2,183,929,941 train tokens** (8K BPE, 3.80 bytes/token) |
| size | **8.61 GB**, 945,234 documents |
| languages | **EN 46.2 % / RU 53.8 % of bytes**, which is **EN 62.1 % / RU 37.9 % of tokens** |
| documents | 787,004 after MinHash (20,022 dropped) + 289,000 Russian bulk articles (title-deduplicated) |
| tokenizer | 8K byte-level BPE, bilingual, trained on the mix |
| held-out | 2 % per language, disjoint by document |
| provenance | source, licence, URL and SHA-256 for **every** shard in `manifest.jsonl` |

**Why bytes and not tokens.** The quota is fixed in bytes because that is the
only quantity that does not depend on the tokenizer, and the token share is
measured and published next to it. The measurement is blunt: Russian costs
**5.12 bytes/token** against **3.00** for English, so a byte quota is not a
token quota. The quota build gave Russian 15.7 % of the tokens at 29 % of the
bytes; the fix was more Russian data, not a different formula. The full Russian
Wikipedia (2.0 GB of articles) and the full Russian Wikisource (1.1 GB of
public-domain literature) were imported with curated titles skipped, taking
Russian to **37.9 % of tokens**.

## Composition

Domain quotas are the design contract; the "measured" column is the actual mix
after dedup, as a share of all documents.

| domain | quota | measured | main sources |
|---|---:|---:|---|
| mathematics | 25 % | 19.1 % | OpenWebMath, Wikipedia math |
| science (physics, astronomy, cosmology) | 14 % | 11.1 % | arXiv, Wikipedia, curated cosmology articles, theoretical-physics categories |
| code | 15 % | 7.5 % | CodeParrot Clean, CPython stdlib |
| machine learning | 12 % | 5.4 % | arXiv cs.LG/CS.CL abstracts, Cosmopedia, framework docs |
| literature | 11 % | 21.2 % | Project Gutenberg, Wikisource (1.1 GB Russian), FineWeb-2 |
| logic | 10 % | 3.2 % | Wikipedia logic, Lean 4 mathlib proofs |
| encyclopedic | 9 % | 30.3 % | Wikipedia EN/RU, full Russian Wikipedia import |
| CS / software engineering / engineering | 3 % | 2.3 % | IETF RFCs, Wikipedia (microprocessors, VLSI, XP, agile) |
| humour / satire | 1 % | <0.1 % | Project Gutenberg, Fandom transcripts |
| self-identity | 1 % | 0.09 % | synthetic: 800 plain + 850 pastiche per language, no numeric facts |

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

The tokenised bins (uint16, 4.4 GB) are distributed outside git — GitHub is not
a data host.

```python
# Kaggle (private dataset with the tokenised bins, uploaded after the build)
#   kaggle datasets download -d icold21/kron-corpus-2b
import numpy as np
tokens = np.memmap("train.bpe.bin", dtype=np.uint16, mode="r")

# Hugging Face (card, manifest, identity slice, easter eggs — no bins)
#   from datasets import load_dataset
#   ds = load_dataset("Icold/kron-corpus")
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
synthetic self-identity set where the model calls itself *Kron* (800 plain +
850 pastiche per language), and 2 MB of animated sci-fi dialogue (81 episode
transcripts plus 11 season-9 plot summaries). Both are under 0.1 % of the corpus and marked in the
manifest.</sub>
