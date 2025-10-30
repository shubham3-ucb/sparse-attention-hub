Personal README — local edits and how-to
=====================================

Purpose
-------
- Track local changes, quick run & debug instructions, and rationale for fixes we made while experimenting with sparse attention.

Key edits made in this fork
--------------------------
- Implemented chunked prefill support (dense and sparse paths) in `sparse_attention_hub/adapters/huggingface.py`.
- Added concise SPARSE_DEBUG logging to `output_test_sparse/hf_prefill.log` (markers, per-chunk entries, OK/FAIL signals).
- implemented an automated dense chunked-vs-single-check (float32) for deterministic verification.
- Added sparse chunked sanity checks (seq_lens growth, sparse_meta_keys history) and per-chunk logging.
- Fixed MagicPig internal tensor layout bug: replaced `vectors.view(...)` with `vectors.reshape(...)` to support non-contiguous tensors.

Why the MagicPig change matters
------------------------------
- The masker used `.view()` on an internal tensor that could be non-contiguous after chunking/transpose operations; this produced a RuntimeError. Replacing with `.reshape()` is a safe, minimal fix that preserves behavior and avoids brittle external hacks.

Environment & important env vars
-------------------------------
- Activate conda env: `conda activate sparse_attention_hub`
- `HF_TOKEN` or `HF_HUB_TOKEN` — (export if using private HF models)
- `PREFILL_CHUNK_SIZE` — set to an integer (e.g. 128) to enable chunked prefill for dense and/or sparse (adapter honors it)
- `SPARSE_DEBUG=1` — enable verbose prefill logging and write to `output_test_sparse/hf_prefill.log`
- `SPARSE_LOG_PATH` — optional path override for the prefill log

Quick run (1-sample sanity)
---------------------------
1. Activate env and cd to repo:

```bash
conda activate sparse_attention_hub
cd /home/nvidia/shubham/sparse/sparse-attention-hub
```

2. Set env and run (example):

```bash
export PREFILL_CHUNK_SIZE=128
export SPARSE_DEBUG=1
# export HF_TOKEN=<your_token>
python3 test_sparse.py
```

What to inspect
----------------
- `output_test_sparse/hf_prefill.log`
  - Look for `%% working on dense prefill` and `%% working on sparse prefill` markers.
  - Dense check: `[prefill] CHUNK ASSERT OK: chunk_size=...` confirms chunked==full prefill (float32 deterministic check).
  - Sparse check: `[prefill] SPARSE CHUNK SANITY OK: final_seq=...` confirms KV accumulation and sparse metadata growth.
  - Per-layer `final_mask_density=` values show whether maskers produced sparse masks (<1.0) or fell back to full (1.0).

Notes / troubleshooting
-----------------------
- If you see `view size is not compatible` from MagicPig previously, that is fixed by the internal `reshape` change (`magic_pig.py`).
- We removed the adapter-level contiguity hack after fixing Masker internals.
- For strict equality tests use `torch.float32` (set in `test_sparse.py` or model_kwargs) — mixed precision (bfloat16/float16) can show tiny numeric diffs.
- Avoid committing tokens; search for `HF_TOKEN` or `.env` before pushing.

Keeping this file updated
-------------------------
- Add a short bullet for each future change: file, one-line rationale, and whether it requires review by upstream authors.
- We should open PRs against upstream for internal fixes that are generally useful (e.g. the `reshape` change in MagicPig) and reference this README.

How to push this fork to GitHub (Plan A)
-------------------------------------
1. Fork upstream on GitHub to `github.com/<you>/sparse-attention-hub`.
2. Locally:

```bash
cd /home/nvidia/shubham/sparse/sparse-attention-hub
git add -A
git commit -m "Save local changes before push" || true
git remote rename origin upstream || true
git remote add origin git@github.com:shubham3-ucb/sparse-attention-hub.git
git push -u origin --all
git push origin --tags
```

What to report upstream
----------------------
- The MagicPig `.view()` → `.reshape()` change (bug and fix).
- The need for stable prefill/chunking behavior and suggested tests for non-contiguous tensors.

Contact / next steps
---------------------
- I will update this file as we make more changes. Keep entries short and factual.

End.


