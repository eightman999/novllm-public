#!/usr/bin/env python3
"""長編対応のための「作品要約」と「直近要約」をLM Studioで生成する。

**なぜ要るか（PLAN_20260730.md Phase 2）**:
現行のctx形式は prompt に **直前2,400字しか入っていない**。短編なら直前文脈だけで
続きが書けるが、長編では登場人物・既出設定・伏線を必ず忘れる。作品の中央値は
147,711字あるので、2,400字は全体の1.6%でしかない。

そこで prompt を3層にする:
    作品要約(全体)  ← このスクリプトが作る（作品ごとに1回）
    直近要約(直前)  ← このスクリプトが作る（作品×話数ごと）
    直前文脈(逐語)  ← build_context_view.py が既に作っている

**pipeline/lmstudio_classify.py との違い**:
あちらは「本文の要約・引用・再生成を一切させない」ことを契約にした**分類**専用
クライアントで、その制約は意図的なもの。ここは目的が正反対（要約そのものが成果物）
なので、あちらを流用せず別モジュールにしてある。接続先がローカルのLM Studioのみで、
外部LLMサービスへ接続しないという方針だけ共通。

**直近要約の定義**: 対象サンプルの episode_no より**前**の話だけを材料にする。
対象話の本文を混ぜると答えを漏らすことになり、評価が無意味になる。
さらに材料は直前 --recent-source-chars 字に限る（作品全体の記憶は作品要約が担当）。

**中断耐性**: 出力へ1件ずつ追記し、起動時に既存キーを読んで再開する。
28時間級のジョブはプロセスが落ちる前提で組む（7/30 に学習側で経験済み）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

DEFAULT_OUT = os.environ.get(
    "NOVLLM_SUMMARY_OUT", "./runs/summaries.jsonl")
DEFAULT_BASE_URL = os.environ.get("NOVLLM_LMSTUDIO_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = os.environ.get("NOVLLM_LMSTUDIO_MODEL", "qwen3-14b-mlx-4bit")
DEFAULT_TIMEOUT = 180

# thinkingを無効化する指示。無視するモデルがあるので _parse でも除去する。
_NO_THINK = "/no_think"

WORK_SUMMARY_SCHEMA = {
    "name": "work_summary",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "characters": {"type": "array", "items": {"type": "string"}},
            "setting_terms": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "characters", "setting_terms"],
        "additionalProperties": False,
    },
}

RECENT_SUMMARY_SCHEMA = {
    "name": "recent_summary",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "open_threads": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "open_threads"],
        "additionalProperties": False,
    },
}


def _work_system_prompt(chars: int) -> str:
    return (
        "あなたは小説の要約器です。与えられた作品の断片から、続きを書く人が読むための"
        "作品要約を作ってください。\n"
        f"- summary: {chars}字以内。あらすじ・世界設定・主要人物の関係を、続きを書くのに"
        "必要な情報だけに絞って書く。感想・評価・宣伝文句は書かない\n"
        "- characters: 本文に明示された人名のみ。推測で補わない\n"
        "- setting_terms: 作品固有の地名・組織名・道具名・能力名のみ。一般名詞は入れない\n"
        "本文をそのまま引用せず、必ず要約の形にすること。\n"
        f"{_NO_THINK}"
    )


def _recent_system_prompt(chars: int) -> str:
    return (
        "あなたは小説の要約器です。与えられた直近の本文から、続きを書く人が読むための"
        "「ここまでの流れ」を作ってください。\n"
        f"- summary: {chars}字以内。誰が・どこで・何をして・今どうなっているかを時系列で書く\n"
        "- open_threads: まだ解決していない問題・約束・謎を短く列挙する。無ければ空配列\n"
        "本文をそのまま引用せず、必ず要約の形にすること。\n"
        f"{_NO_THINK}"
    )


def _post(session, base_url, model, system_prompt, user_content, schema, timeout):
    response = session.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_schema", "json_schema": schema},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _parse(content: str) -> dict:
    text = content.strip()
    if "<think>" in text and "</think>" in text:
        text = text[text.rindex("</think>") + len("</think>"):].strip()
    return json.loads(text)


def call_llm(session, args, system_prompt, user_content, schema) -> dict | None:
    """失敗したら None を返す。**既定値で埋めない** — 捏造した要約が学習データに
    混ざる方が、欠損しているより有害なため。"""
    for _ in range(args.max_retries):
        try:
            return _parse(_post(session, args.base_url, args.model,
                                system_prompt, user_content, schema, args.timeout))
        except (requests.RequestException, json.JSONDecodeError, KeyError,
                IndexError, ValueError):
            continue
    return None


def load_targets(path: str) -> tuple[set[str], set[tuple[str, int]]]:
    """学習に使うサンプルの meta から、要約が必要な作品と(作品,話数)を拾う。
    view全体(393,740件)ではなく**実際に学習に使うサブセット**だけを対象にする。"""
    works: set[str] = set()
    episodes: set[tuple[str, int]] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line).get("meta") or {}
        ncode, episode_no = meta.get("ncode"), meta.get("episode_no")
        if ncode is None or episode_no is None:
            continue
        works.add(ncode)
        episodes.add((ncode, int(episode_no)))
    return works, episodes


def load_done(path: Path) -> set[str]:
    """既に生成済みのキーを読む（再開用）。壊れた行は飛ばす。"""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as source:
        for line in source:
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="chunks_v2 のglob")
    parser.add_argument("--targets", required=True,
                        help="要約が必要なサンプルのJSONL(meta.ncode/meta.episode_no を見る)。"
                             "train_ctx_subset/balanced-*.jsonl をそのまま渡してよい")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"出力JSONL(追記・再開可)。既定は {DEFAULT_OUT}")
    parser.add_argument("--kind", choices=["work", "recent", "both"], default="both")
    parser.add_argument("--work-summary-chars", type=int, default=600)
    parser.add_argument("--recent-summary-chars", type=int, default=600)
    parser.add_argument("--work-source-chars", type=int, default=5000,
                        help="作品要約の材料。作品全体から等間隔に抜いてこの字数まで")
    parser.add_argument("--recent-source-chars", type=int, default=5000,
                        help="直近要約の材料。対象話の直前からこの字数まで")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, help="先頭N件で打ち切る（スループット計測用）")
    parser.add_argument("--shard", help="複数マシンで分担する。i/n 形式（例: Mac側 0/2 / 実機側 1/2）。"
                                        "**--out もマシンごとに分けること**（同一ファイルへの"
                                        "並行追記は壊れる）。生成後に結合する")
    args = parser.parse_args()

    works, episodes = load_targets(args.targets)
    print(f"[build_summaries] 対象: 作品{len(works):,}件 / (作品,話){len(episodes):,}件",
          file=sys.stderr)

    # 本文は作品ごとに読み順で保持する。対象作品だけに絞ってメモリを節約する。
    by_work: dict[str, list[dict]] = defaultdict(list)
    paths = sorted(glob.glob(args.chunks))
    if not paths:
        raise SystemExit(f"見つかりません: {args.chunks}")
    for path in paths:
        with open(path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                ncode = row["meta"]["ncode"]
                if ncode in works:
                    by_work[ncode].append(row)
    for rows in by_work.values():
        rows.sort(key=lambda r: (r["meta"]["episode_no"], r["meta"]["chunk_index"]))
    print(f"[build_summaries] 本文を読み込んだ作品: {len(by_work):,}件", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    if done:
        print(f"[build_summaries] 生成済みを再開: {len(done):,}件", file=sys.stderr)

    jobs: list[tuple[str, str, tuple]] = []
    if args.kind in ("work", "both"):
        jobs += [("work", f"work:{n}", (n,)) for n in sorted(works)]
    if args.kind in ("recent", "both"):
        jobs += [("recent", f"recent:{n}:{e}", (n, e)) for n, e in sorted(episodes)]
    # --- 複数マシンで分担するための分割（2026-08-01追加）------------------
    # 15,855件を1台で回すと実測18.5秒/件＝81時間かかる。Mac(LM Studio)と
    # 実機(Ollama)で分担できるようにする。
    #
    # **done を引く前に分割する。** 後に分割すると、各ワーカーの進捗状況によって
    # 担当が変わってしまい、同じ仕事を2台で二重生成する／どちらも作らない、が起きる。
    # jobs は sorted() 由来で決定的なので、同じ引数なら常に同じ担当になる。
    if args.shard:
        try:
            index_text, total_text = args.shard.split("/")
            shard_index, shard_total = int(index_text), int(total_text)
        except ValueError:
            raise SystemExit(f"--shard は i/n の形式で指定してください（例: 0/2）: {args.shard}")
        if not (shard_total >= 1 and 0 <= shard_index < shard_total):
            raise SystemExit(f"--shard の範囲が不正です（0 <= i < n, n >= 1）: {args.shard}")
        before = len(jobs)
        jobs = [j for i, j in enumerate(jobs) if i % shard_total == shard_index]
        print(f"[build_summaries] 分担 {shard_index}/{shard_total}: "
              f"{before:,}件中 {len(jobs):,}件を担当", file=sys.stderr)

    jobs = [j for j in jobs if j[1] not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[build_summaries] 生成対象: {len(jobs):,}件", file=sys.stderr)

    session = requests.Session()
    started = time.perf_counter()
    written = failed = skipped = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for index, (kind, key, ident) in enumerate(jobs, 1):
            if kind == "work":
                (ncode,) = ident
                rows = by_work.get(ncode) or []
                # 作品全体から等間隔に抜く。冒頭だけだと後半の設定が落ちる。
                texts = [str(r.get("text") or "") for r in rows]
                if not texts:
                    skipped += 1
                    continue
                stride = max(1, len(texts) // 12)
                material = "\n\n---\n\n".join(texts[::stride])[: args.work_source_chars]
                result = call_llm(
                    session, args, _work_system_prompt(args.work_summary_chars),
                    f"以下は同一作品からの断片です。\n\n{material}", WORK_SUMMARY_SCHEMA)
            else:
                ncode, episode_no = ident
                rows = by_work.get(ncode) or []
                # **対象話より前だけ**を材料にする。対象話を含めると答えの漏洩になる。
                prior = [str(r.get("text") or "") for r in rows
                         if r["meta"]["episode_no"] < episode_no]
                material = "".join(prior)[-args.recent_source_chars:]
                if len(material) < 400:
                    # 第1話近辺は直近要約が作れない。欠損として記録し捏造しない。
                    out.write(json.dumps({"key": key, "kind": kind, "ncode": ncode,
                                          "episode_no": episode_no, "summary": None,
                                          "reason": "直前の本文が不足"},
                                         ensure_ascii=False) + "\n")
                    out.flush()
                    skipped += 1
                    continue
                result = call_llm(
                    session, args, _recent_system_prompt(args.recent_summary_chars),
                    f"以下はここまでの本文です。\n\n{material}", RECENT_SUMMARY_SCHEMA)

            if result is None:
                failed += 1
                continue
            record = {"key": key, "kind": kind, "ncode": ident[0],
                      "summary": result.get("summary")}
            if kind == "work":
                record["characters"] = result.get("characters") or []
                record["setting_terms"] = result.get("setting_terms") or []
            else:
                record["episode_no"] = ident[1]
                record["open_threads"] = result.get("open_threads") or []
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            written += 1

            if index % 20 == 0:
                elapsed = time.perf_counter() - started
                rate = index / elapsed
                remaining = (len(jobs) - index) / rate if rate else 0
                print(f"  {index:,}/{len(jobs):,}  {rate:.2f}件/秒  "
                      f"残り約{remaining / 3600:.1f}時間  失敗{failed}  欠損{skipped}",
                      file=sys.stderr)

    elapsed = time.perf_counter() - started
    print(f"\n[build_summaries] 書き出し {written:,}件 / 失敗 {failed:,}件 / "
          f"欠損 {skipped:,}件 / {elapsed / 60:.1f}分 "
          f"({written / elapsed if elapsed else 0:.2f}件/秒)", file=sys.stderr)
    if failed:
        print("  失敗分は記録していない。同じコマンドを再実行すれば再開する。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
