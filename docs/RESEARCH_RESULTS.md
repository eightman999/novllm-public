# Research results publication workflow

The research loop is `experiment → machine-readable results → report → GitHub`.
Keep each measured run immutable and keep experimental candidates distinct from a
frozen tokenizer. Do not re-create missing numbers from chat history.

## Run and collect

Record input hashes, tokenizer/config hashes, exact source counts, seed, software,
GPU and the actual stopping condition. Preserve requested and actual checkpoint
fractions. Keep unsuccessful runs and missing observations explicitly labeled.
The runtime and data acquisition commands are documented in
[the fixed runtime](../research/phase55_runtime/README.md) and
[the hardening audit](../results/phase551/audit/README.md).

Store raw text, checkpoint binaries and private connection configuration outside
tracked result directories. The macOS monitor is optional, read-only and does not
invoke an AI model; it is not an experiment-completion validator.

## Export and independently verify

Use the existing exporter for the corresponding artifact type:

```bash
# Completed original cohort, optionally with a new run directory
python3 scripts/export_phase551_public.py \
  --payload LOCAL_MEASUREMENT_ARCHIVE.zip --extra-run LOCAL_RUN_DIRECTORY \
  --tokenizer-config-root LOCAL_TOKENIZER_CONFIGS \
  --output results/phase551
# Only a complete, compatible same-seed cohort may receive an A/B/C/D decision
python3 scripts/analyze_phase551.py --results results/phase551 --seed 1
# Hardened evaluation: text-free IDs, hashes, counts and NLL
python3 scripts/export_phase551_audit_public.py \
  --input LOCAL_PRIVATE_AUDIT --output results/phase551/audit
python3 scripts/verify_phase551_public.py --audit results/phase551/audit
# Historical tokenizer-only observations are a separate export
python3 scripts/export_phase551_public.py \
  --phase5-summary LOCAL_PHASE5_SUMMARY.json --output results/phase5
```

The archive and run directories are local inputs, not files to commit. Null means
unknown/unrecorded, never zero. Compare BPB on the same byte denominator. Report
measured training/invocation time separately; analytic FLOPs are not measured
compute. Preserve domain regressions even when overall BPB improves.

## Report and publish

Describe Measured, Derived, Interpretation and Not yet verified separately. Link
the machine-readable tables and immutable source hashes. For corpus text without
an established redistribution grant, publish the source URL, retrieval procedure,
hash and manifest only. Private record/work IDs are consistently hashed; public
upstream source IDs may remain explicit. No credentials, private endpoints,
personal data, raw text or model checkpoints belong in these result commits.

Use an explicit file allowlist when staging, inspect the staged diff, and run the
relevant exporter/gate tests. Make separate commits for code, results and the final
research interpretation. Push the research branch and cite its exact commit in
reports. A published branch is not a tokenizer freeze or permission to start a new
seed/model-size experiment.

Exact retraining still requires the original licensed inputs and tokenizer
artifacts, which are not distributed here. Public NLL/counts suffice to recompute
BPB and paired-work bootstrap intervals, not to reproduce an unavailable corpus.
