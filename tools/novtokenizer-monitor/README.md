# NovTokenizer Monitor

macOS native SwiftUI monitor. No model/API calls, AI tokens or recurring automation. While open, the app performs one read-only SSH snapshot about every 10 seconds. Closing the app stops monitoring only; it does not stop training. No start/stop/restart experiment controls are present.

## Configure / build / launch

Requirements: macOS 13+, Swift toolchain via Xcode command line tools, `/usr/bin/python3`, SSH with an already configured trusted host key and noninteractive authentication.

Create `.private/config.json` (ignored by Git):

```json
{"host":"YOUR_SSH_ALIAS","root":"/your/experiment/root","evaluation_root":"/your/evaluation/root"}
```

```bash
bash build.sh
open 'build/NovTokenizer Monitor.app'
python3 collector.py --once --config .private/config.json
```

`build/` is also ignored and contains a copy of private configuration. Do not distribute that app bundle without removing its `Contents/Resources/config.json`. Source code contains no private host names. No installation, login item, background daemon or SSH-key change is performed.

The collector uses BatchMode, StrictHostKeyChecking, 8-second connect timeout and an 18-second overall timeout. A lock prevents overlapping collection for the same config path; the GUI also disables overlapping refresh. Connection failures keep the last successful snapshot, with its acquisition age and a stale warning after 30 seconds. Authentication errors display an explicit status without echoing SSH secrets/logs.

J64 progress follows `results/cuda_same_source/runs/seed1/j-reversible-sp-unigram-64k/artifacts/` and `results/phase551_controller/launch.json`. Evaluation counts the eight `old-seed*.json` / `hardened-seed*.json` outputs under the evaluation root's `results/`, with `done.json` completion and `comparison.json` aggregates. Missing progress is not completion. GPU observations are point-in-time nvidia-smi values; absent values display an em dash.

ETA is a rough wall-time extrapolation from completed source characters. Speeds use cumulative synchronized training time and exclude evaluation/checkpoint work; they need not match the ETA. A progress age warning can reflect checkpoint evaluation, not a stopped job. Completed elapsed time uses `done.json` when available.

## Verify

`bash verify.sh --check` probes tools; `bash verify.sh` or `--full` builds and runs four collector contract tests. A separate live `--once` call verifies SSH/GPU/run metrics. Finally open the actual app and exercise pause/manual refresh; build/tests alone do not prove GUI operation.
