# Rebuild

The build is deterministic given the same shards: deduplication walks documents
in SHA-256 order, and the mixer walks each bucket in the same order.

```bash
pip install -r requirements.txt      # tokenizers, datasets, numpy, tqdm, pandas

# 1. Primary sources through Hugging Face (~5.5 GB of text)
python scripts/build_bilingual.py fetch-hf --source all

# 2. Public-domain books (pglaf mirror; gutendex is unreliable from some networks)
python scripts/build_bilingual.py fetch-gutenberg-ids \
    --ids 972,1080,269,19942,849,308,3176,86,74,76,1906,2183,1026,3177,3178,1595,829 \
    --domain humour

# 3. Formal proofs (Lean 4)
python scripts/build_bilingual.py fetch-tarball \
    --repo leanprover-community/mathlib4 --ext .lean --domain logic

# 4. Self-identity slice (needs a local Ollama with qwen3.5:4b)
python scripts/gen_identity.py --samples-per-language 800
python scripts/gen_identity.py --samples-per-language 850 --style rick

# 5. Dedup, quotas, held-out, report
python scripts/build_bilingual.py build --budget-en-mb 4300 --budget-ru-mb 2870 --heldout-mb 120

# 6. 8K bilingual BPE + uint16 memmaps + fertility metadata
python scripts/prepare_bilingual_bin.py --vocab 8192
```

Outputs:

| path | content |
|---|---|
| `data/bilingual/shards/*.jsonl` | fetched, cleaned documents |
| `data/bilingual/mix/*.jsonl.gz` | the quota-selected mix |
| `data/bilingual/heldout/*.jsonl` | disjoint validation slice |
| `data/bilingual/tokenizer.json`, `tokenizer_meta.json` | 8K BPE |
| `data/bilingual/train.bpe.bin`, `val.bpe.bin` | uint16 token stream |

The build prints every shortfall (a quota missed because the pool is too small)
and writes it to `corpus_report.json`; nothing is silently padded.
