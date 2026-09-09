# Phase 5.5 same-source CUDA runtime snapshot

This is the minimal Python import closure of the completed Phase 5.5 CUDA runner,
extracted from the fixed review payload. `SOURCE_MANIFEST.json` records the source
archive hash and both original and published file hashes. All model, tokenizer
adapter, dataset validation and training/evaluation modules are byte-identical.
Only the entry point adds `--run-root`, passes that argument to child runs, and
checks the distributed code closure instead of unrelated unpublished modules.
Training conditions, optimizer, stopping rule and checkpoint evaluation are unchanged.

## Inputs and reproducibility limits

Python with PyTorch and SentencePiece is required. The original CUDA runs used
float32; see the published hardware/software tables for measured versions.
`tokenizers` is optional and only used by the unused HF adapter.
The Python package name is `canonical_corpus`; launch from this snapshot directory
to avoid importing a different checkout.

The following input files are NOT distributed: `dataset/train.jsonl`,
`dataset/eval.jsonl`, `dataset/manifest.json`, `dataset/manifest.sha256`, and each
J32/J48/J64 `tokenizers/<candidate>/tokenizer.model` and `config.json`.
Their expected hashes are preserved verbatim in `protocol.json`. All tokenizer
inputs are checked even when selecting one candidate, matching the original runner.
Do not replace these with newly scraped text and describe the result as reproduction.
Corpus download/splitting and tokenizer retraining cannot currently be reproduced
end to end from this snapshot alone. Dataset/text rights and acquisition manifests
must be supplied separately. This snapshot supplies no corpus text or checkpoints.

## Existing workflow and command mapping

Set `RUN_ROOT` to a local directory containing the exact inputs above. Paths below
are relative examples; no particular machine or private service is required.

```bash
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN_ROOT=./local-inputs
# prepare/validate: existing loader checks manifest, shards, row hashes and split audit
python -c 'import sys; from pathlib import Path; from canonical_corpus.phase55_data import load_dataset; load_dataset(Path(sys.argv[1])/"dataset")' "$RUN_ROOT"
# optional actual full-context optimizer and short E2E smoke (writes checkpoints)
python scripts/run_phase55_cuda_same_source.py smoke --seed 1 --run-root "$RUN_ROOT"
# train + checkpoint/final evaluate: evaluation is integrated into the existing runner
python scripts/run_phase55_cuda_same_source.py run --seed 1 --candidate j-reversible-sp-unigram-48k --run-root "$RUN_ROOT"
# aggregate completed J32/J48 pair; returns without output if either run is missing
python scripts/run_phase55_cuda_same_source.py report --seed 1 --run-root "$RUN_ROOT"
```

`pair` runs both J32/J48 and invokes `report`; it is not necessary for the single
J64 exploratory run. Select `j-reversible-sp-unigram-64k` with `run --seed 1` for
that run. The archived protocol's J64 deferral text describes the Phase 5.5 snapshot,
not a new completion claim. No second J64 seed, tokenizer freeze or Phase 6 is implied.
Checkpoint fractions are 0.19, 0.20, 0.50, 0.75 and 1.00, as in the fixed runner.
There is no standalone eval command; `evaluate` is an existing runtime API and the
runner evaluates during training and after checkpoint reload. `report` aggregates
J32/J48 only; the separate public result exporter handles J64 comparison tables.

Outputs go under `RUN_ROOT/results/cuda_same_source`. Runtime output can contain
local input paths and must pass the public exporter before publication. Do not
commit checkpoint files or raw local output directories. Existing run directories
are intentionally refused, so an accidental invocation cannot overwrite a run.

## Verification scope

`verification.json` records syntax compilation, isolated CLI help, import closure,
escape/unescape and analytical parameter-count checks. These are local checks;
they do not represent a fresh CUDA training run or end-to-end data reproduction.
