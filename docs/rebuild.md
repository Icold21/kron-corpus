# Rebuild

The build is reproducible from the shards: `manifest.jsonl` records source,
licence, URL and sha256 for every one of the 87 shards, and `kronllm/` here is a
snapshot of the package from [kron-llm](https://github.com/Icold21/kron-llm) at
the same commit, so the pipeline runs without cloning a second repository.

## What "reproducible" means here

* **Fetches are append-only and sharded.** A rerun adds shards, it does not
  rewrite history; `record_shard` refuses a shard without a licence and a URL.
* **The mixer is deterministic given the same shards.** Pass 1 streams every
  document, keeps `(sha, lang, domain, bytes, path, offset)` plus a quality
  score, and selects quota-by-quota; pass 2 seeks to those offsets and writes
  the mix. Peak memory is ~1.1 GB, not the 25 GB the in-memory builder needed.
* **The held-out bin never changes.** `--reuse-val-dir` copies the canonical
  `val.bpe.bin` (sha256 `a45bbb03…`) into the new bin, which is what keeps
  `val_bpb_mixed` comparable across corpus rebuilds and training sessions.

## Commands

```bash
# 1. fetch (or append to) the shards
python scripts/build_bilingual.py fetch-hf --source all
python scripts/build_bilingual.py fetch-gutenberg-ids ...
python scripts/build_bilingual.py fetch-wikibooks ...
python scripts/gen_identity.py --samples-per-language 800

# 2. mix: quality-ranked quotas, per language
python scripts/build_streaming.py --budget-en-mb 4800 --budget-ru-mb 5600

# 3. bins: reuse tokenizer and held-out bin, no inserted prompts
python scripts/prepare_bilingual_bin.py --out-dir bins \
    --reuse-tokenizer data/bilingual/tokenizer.json \
    --reuse-val-dir held_out --phase epoch --no-domain-prompts
```

Notes that cost time to learn:

* the mix directory is swept at the start of pass 2 — the chat pools used by the
  decay pass are kept by `sweep_stale_mix`, which matches `"_chat." in name`
  (`Path.stem` strips only one suffix, and a `stem.endswith("_chat")` guard
  silently missed them twice);
* documents shorter than 200 characters are dropped before scoring, so a quota
  is filled from the ranked remainder rather than padded;
* the decay pass is the same build at 30 % of the budget, plus the chat pools
  from the code repository, encoded with `--phase anneal --interleave chat`.
