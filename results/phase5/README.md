# Phase 5 historical tokenizer observations

Allowlisted export of the original Phase 5 summary, with its SHA-256 in
`provenance.json`. CSV and JSON contain identical recorded values. Missing values
are JSON null / CSV empty; no missing measurements were reconstructed.

- `summary`: candidate identity and recorded round-trip rate.
- `category_metrics`: compression, reversibility and category size observations.
- `parameter_cost`: source-derived embedding/head cost under assumed hidden sizes;
  these are not measured trained-model parameter counts.
- `saturation`, `saturation_exclusions`, `web_regressions`: historical derived comparisons.
- `shortlist`: the historical four-candidate selection and its recorded metrics.

This is tokenizer-only evaluation. Small kanbun, kakikudashi, gyaru and technical
probes are historical limitations. The shortlist and saturation exclusions do not
supersede subsequent Phase 5.5 LM results or the J64 exploratory experiment.
Freeze remains undecided; no Phase 6 result is claimed. Corpus text, original
item identifiers, arbitrary source prose, private paths and checkpoints are omitted.

Recreate this directory from an authorized local copy of the source summary:

```bash
python scripts/export_phase551_public.py --phase5-summary SOURCE_SUMMARY.json --output results/phase5
```
