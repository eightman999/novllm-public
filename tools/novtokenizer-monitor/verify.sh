#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = --check ]; then
  command -v xcrun
  /usr/bin/python3 --version
  printf '%s\n' 'Will build native app and run collector tests; live SSH/GUI separate.'
  exit 0
fi
trap 'echo VERIFY FAIL' ERR
printf '%s\n' '[1/2] Native SwiftUI build'
bash build.sh
printf '%s\n' '[2/2] Collector contract tests'
/usr/bin/python3 -m unittest test_collector.py
printf '%s\n' 'VERIFY PASS'
