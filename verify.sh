#!/usr/bin/env bash
# novllm 検証ハーネス (AIエージェント向け)
#
# 使い方:
#   ./verify.sh          高速検証 (構文チェック + tokenizer非依存の速いユニットテストのみ)
#   ./verify.sh --check  環境確認のみ (テストは実行しない)
#   ./verify.sh --full   重い検証 (pipeline/tests 全体。HFローカルキャッシュ前提のテストを含む)
#
# 実行環境の実態 (リポジトリ実態調査に基づく):
#   - pyproject.toml / pytest.ini / conftest.py は存在しない。READMEのセットアップ手順は
#     「python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt」。
#   - よって .venv/bin/python があれば優先し、無ければ system python3 にフォールバックする。
#   - pytest は requirements.txt に含まれない (未導入環境ではテスト工程をSKIP表示する)。
#   - テストは pipeline/tests/ 配下。リポジトリ直下から実行する前提 (from pipeline.X import ...)。
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"

MODE="fast"
case "${1:-}" in
  "")      MODE="fast" ;;
  --check) MODE="check" ;;
  --full)  MODE="full" ;;
  *) echo "使い方: $0 [--check|--full]" >&2; exit 2 ;;
esac

STEP=0
TOTAL=0
step() { STEP=$((STEP + 1)); echo "[${STEP}/${TOTAL}] $1"; }
fail() { echo "VERIFY FAIL"; exit 1; }

# python実行系の決定 (.venv優先 -> venv -> system python3)
PY=""
PY_KIND=""
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"; PY_KIND="venv(.venv/bin/python)"
elif [ -x "venv/bin/python" ]; then
  PY="venv/bin/python"; PY_KIND="venv(venv/bin/python)"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"; PY_KIND="system(python3)"
fi

has_module() {
  "$PY" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$1') else 1)" >/dev/null 2>&1
}

# 高速ユニットテストの選定 (各テストのimport実態を確認して選定):
#   採用 (純粋ロジック):        test_normalize_meta.py / test_sampling.py / test_style_stats.py
#   採用 (requestsはmockのみ):  test_api_refresh.py / test_lmstudio_classify.py
# 除外理由:
#   test_chunker.py            Qwen3-8B tokenizer のHFローカルキャッシュ必須 (無いと失敗する)
FAST_TESTS_PURE="pipeline/tests/test_normalize_meta.py pipeline/tests/test_sampling.py pipeline/tests/test_style_stats.py"
FAST_TESTS_REQUESTS="pipeline/tests/test_api_refresh.py pipeline/tests/test_lmstudio_classify.py"

compile_step() {
  step "構文チェック (compileall: pipeline/ tools/ とルート直下の*.py)"
  if ! "$PY" -m compileall -q pipeline tools \
      build_dataset.py common.py control_format.py export_novel_jsonl.py \
      generate.py prune_model.py train_lora.py; then
    echo "  構文エラーあり"
    fail
  fi
}

if [ "$MODE" = "check" ]; then
  TOTAL=4
  step "python実行系の確認"
  if [ -z "$PY" ]; then
    echo "  pythonが見つからない (.venv/bin/python も python3 も無い)"
    fail
  fi
  echo "  実行系: ${PY_KIND} / $("$PY" --version 2>&1)"

  step "venv / 依存定義の確認"
  if [ -d ".venv" ]; then echo "  .venv: あり"; else echo "  .venv: なし (README手順: python -m venv .venv)"; fi
  if [ -f "requirements.txt" ]; then echo "  requirements.txt: あり"; else echo "  requirements.txt: なし"; fi
  if [ -f "requirements.frozen.txt" ]; then echo "  requirements.frozen.txt: あり"; fi

  step "pytest導入確認"
  if has_module pytest; then
    echo "  pytest: あり"
  else
    echo "  pytest: なし (無印/--fullのテスト工程はSKIPまたは失敗する。導入: $PY -m pip install pytest)"
  fi

  step "主要依存の存在確認 (参考)"
  for m in torch transformers requests; do
    if has_module "$m"; then echo "  ${m}: あり"; else echo "  ${m}: なし"; fi
  done
  echo "VERIFY PASS"
  exit 0
fi

if [ -z "$PY" ]; then
  TOTAL=1
  step "python実行系の確認"
  echo "  pythonが見つからない (.venv/bin/python も python3 も無い)"
  fail
fi

if [ "$MODE" = "fast" ]; then
  TOTAL=2
  compile_step

  step "高速ユニットテスト (tokenizer非依存のみ)"
  if ! has_module pytest; then
    echo "  SKIP (pytest未導入。./verify.sh --check で環境確認)"
  else
    TESTS="$FAST_TESTS_PURE"
    if has_module requests; then
      TESTS="$TESTS $FAST_TESTS_REQUESTS"
    else
      echo "  注: requests未導入のため test_api_refresh.py / test_lmstudio_classify.py はSKIP"
    fi
    # shellcheck disable=SC2086
    if ! "$PY" -m pytest -q $TESTS; then
      fail
    fi
  fi
  echo "VERIFY PASS"
  exit 0
fi

# --full
TOTAL=2
compile_step

step "pytest全体 (pipeline/tests)"
echo "  注意: test_chunker.py 等はQwen3-8B tokenizerのHFローカルキャッシュが無いと失敗する (HF_HUB_OFFLINE=1のため新規ダウンロードはしない)。"
if ! has_module pytest; then
  echo "  pytest未導入のため実行不能"
  fail
fi
if ! "$PY" -m pytest -q pipeline/tests; then
  fail
fi
echo "VERIFY PASS"
