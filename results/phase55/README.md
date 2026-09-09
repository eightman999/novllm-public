# Phase 5.5: J32 → J48 → J64 exploratory

## Measured

These tables export four completed CUDA runs from preserved metrics, not conversation summaries. J48 is the leading candidate; J64 is exploratory; tokenizer freeze is not decided and Phase 6 has not started.

Each run trained on the same 30,000,000 source characters (84,611,719 UTF-8 bytes), at context length 4096, batch size 1, seed 1 or 2 and approximately 150M total parameters. The tied embedding grows with vocabulary and the FFN width shrinks. See `conditions`, `parameters`, and `provenance` CSV/JSON pairs for exact per-run values and hashes. Conditions contain a maximum step safeguard; the actual stop definition is source-character budget, not equal optimization steps.

`summary`: final byte-normalized likelihood, training counters/time, throughput and peak VRAM.
`domain_bpb`: original categories without reclassification; literature remains three Aozora subsets and Web is `web_novel`.
`checkpoints`: all recorded fractions, including 19%, 20%, 50%, 75%, 100%; requested and actual fractions are separate.
`hardware`: recorded GPU, dtype, Python and PyTorch. CUDA runtime was not independently recorded; null is not a claim of no CUDA.
`compression`: content token counts for the fixed source character budget.

## Derived

BPB is checked independently as `nll_sum / source_bytes / ln(2)`. Overall BPB is byte-weighted, not an unweighted average of domain BPBs. Embedding ratio is embedding / total. Characters and bytes per token use training source counts; token ratios to J32/J48 are within seed and equal source counts only. FLOPs are an existing analytic estimate, not measured compute or energy. Its runtime formula is `6 * total_parameters * padded_positions + 12 * layers * hidden * batch * sequence_length^2`; optimizer compute is excluded.

## Interpretation

Compare tokenizer candidates within the same seed/GPU. Seed 1 used RTX 3060 and seed 2 P100; between-seed timing differences are confounded by hardware. Same total parameters does not keep non-embedding capacity fixed. The final cultural regression needs separate evaluation hardening before a tokenizer decision.

## Not yet verified / limitations

The original kanbun and kakikudashi sets are small. Known-identity/exact-hash checks in the original runs do not certify upstream near-duplicates or unknown work identities. These exports do not substitute for the Phase 5.5.1 audit. Training timing excludes tokenization, tensor staging, checkpoint I/O and verification; invocation wall time includes additional work. Peak VRAM covers the entire run, including evaluation/inference/resume verification. Missing values remain JSON null / empty CSV cells. There is no measured FLOPs, energy or independent CUDA-runtime version record here.

No corpus text, raw document identifiers, private endpoints, absolute paths, checkpoint files or arbitrary log strings are exported. Source metrics hashes preserve provenance but the original private archive is not a public download. Third parties can recompute aggregates from published NLL/counts, but exact training reproduction still requires access to the same licensed input data and tokenizer artifacts; hashes alone do not supply them.

## Code / CLI mapping

- Dataset preparation and preserved source accounting: `canonical_corpus/phase55_data.py`, `canonical_corpus/phase55_same_source_runtime.py`.
- Training and evaluation: `scripts/run_phase55_cuda_same_source.py run --seed 1 --candidate j-reversible-sp-unigram-32k` (existing runtime setup required). `pair` runs J32/J48, `smoke` is a preflight, and `report --seed 1` aggregates the seed.
- Table export: `python3 scripts/export_phase551_public.py --payload INPUT.zip --output results/phase55_public`.
- J64 export after completion: add `--extra-run RUN_DIRECTORY`, whose layout is `artifacts/metrics.json`, `artifacts/config.json`, and optional `hardware.json`. Use `results/phase551_public` for the combined comparison. No missing J64 values are fabricated.
- Boundary tests: `python3 -m unittest discover -s tests -p test_phase551_public.py`.

The archived protocol and per-run controls differ in checkpoint fractions; the tables preserve per-run controls and actual curves. Code hashes in the original runs identify the historical runtime, which may differ from current repository code.

`checkpoint_domains` retains all 260 recorded checkpoint/domain measurements. Per-checkpoint training loss was not recorded in these curve objects; validation token NLL is not training loss. `tokenizer_config.json` preserves the J32/J48/J64 recipe J training settings, actual vocabulary, recorded seed, input hash, reversible adapter version and identity normalization. Tokenizer training uses a separate 100M-character input; this is not the 30M-character LM training budget. The recorded seed is provenance, not proof that SentencePiece consumes it. Export those settings with `--tokenizer-config-root TOKENIZER_DIRECTORY`.

`provenance` includes the four historical runtime code hashes. Model activation/norm/position encoding and scheduler/regime are explicitly allowlisted in their respective tables. `--software-observation OBSERVATION.json` exports a separately dated observation, never retroactively filling old-run hardware values.
