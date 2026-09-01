#!/usr/bin/env python3
# tools/eval_tagger.py
"""ローカルLLMによるタグ付けの実用性をncode単位で評価する。

目的は「どぎつい性表現・暴力表現を含むチャンクでも、モデルが拒否せず、かつ本文を
読んだ上でまともなcontent強度を返せるか」を実測すること。判定値のみを返させ、
本文の再生成・要約・引用は要求しない(pipeline/lmstudio_classify.py と同じ方針)。

評価する軸:
  1. 疎通・スキーマ準拠率 : JSONが返るか、スキーマに従うか
  2. 拒否率               : 明示的な拒否や全ゼロ既定値への逃げが出るか
  3. 定値化(constancy)    : 入力が違うのに同じcontent辞書を返していないか
                            (これが起きているとモデルは本文を読んでいない)
  4. 弁別力(discrimination): 露骨チャンクと一般チャンクでスコアが分離するか
  5. 遅延                 : median / p90

使い方:
    # R18作品8件・一般作品8件を自動選定し、2モデルを比較
    python tools/eval_tagger.py \\
        --shards ./data/chunks \\
        --base-url http://127.0.0.1:11434/v1 \\
        --models qwen3-4b:latest,general-qwen3-30b:latest \\
        --auto-r18 8 --auto-general 8 --chunks-per-work 3 \\
        --out runs/tagger_eval

    # 特定ncodeだけ
    python tools/eval_tagger.py --ncodes n1234ab,n5678cd ...

    # 人手ラベル用テンプレートを出す → 埋める → 一致率を測る
    python tools/eval_tagger.py ... --emit-gold gold_template.tsv
    python tools/eval_tagger.py ... --gold gold_filled.tsv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.schemas import (  # noqa: E402
    CHUNK_CLASSIFICATION_JSON_SCHEMA,
    WORK_CLASSIFICATION_JSON_SCHEMA,
)

CONTENT_AXES = [
    "violence", "gore", "sexual_content", "sexual_violence",
    "coercion", "self_harm", "abuse",
]

# 露骨さで章を並べ替えるためだけの検出用語彙。判定そのものには使わず、
# 「モデルにとって最も難しいチャンク」を選ぶためのランカーとして使う。
# --lexicon で外部ファイル(1行1語)に差し替え可能。
DEFAULT_LEXICON_SEXUAL = [
    "喘", "嬌声", "絶頂", "愛液", "陰茎", "陰部", "性器", "膣", "乳首", "乳房",
    "挿入", "射精", "自慰", "性交", "情事", "裸体", "全裸", "痴態", "淫",
]
DEFAULT_LEXICON_VIOLENCE = [
    "斬り", "刺し", "殴", "血飛沫", "鮮血", "臓物", "断末魔", "絶叫", "拷問",
    "惨殺", "虐殺", "切断", "骨が折れ", "internal", "呻き",
]

REFUSAL_PATTERNS = re.compile(
    r"(申し訳|お答えでき|できません|不適切|ポリシー|I can'?t|I'm sorry|cannot assist|as an AI)",
    re.IGNORECASE,
)

SINGLE_AXIS_RUBRIC = {
    "sexual_content": (
        "0 = 性的な描写が一切ない\n"
        "1 = 恋愛感情・キス・軽い接触など、性行為に至らない示唆のみ\n"
        "2 = 性行為があることは分かるが、描写は間接的・省略的\n"
        "3 = 性行為が直接的・具体的に描写されている"
    ),
    "violence": (
        "0 = 暴力が一切ない\n"
        "1 = 争いの気配・威嚇・軽微な接触\n"
        "2 = 明確な戦闘や殴打があるが、負傷描写は簡潔\n"
        "3 = 殺傷・重傷が具体的に描写されている"
    ),
    "gore": (
        "0 = 流血・損壊の描写がない\n"
        "1 = 出血や傷に軽く触れる程度\n"
        "2 = 流血や負傷が具体的に描写される\n"
        "3 = 臓器・切断・惨殺など残虐な身体損壊が詳細に描写される"
    ),
    "sexual_violence": (
        "0 = 該当なし\n1 = 示唆・言及のみ\n2 = 発生するが描写は間接的\n3 = 直接的に描写される"
    ),
    "coercion": (
        "0 = 該当なし\n1 = 軽い圧力・説得\n2 = 明確な脅迫や強制\n3 = 継続的・重度の強制や支配"
    ),
    "self_harm": (
        "0 = 該当なし\n1 = 希死念慮の示唆\n2 = 自傷の言及\n3 = 自傷・自殺が直接描写される"
    ),
    "abuse": (
        "0 = 該当なし\n1 = 軽度の暴言・無視\n2 = 継続的な虐待やいじめ\n3 = 重度の身体的・精神的虐待が描写される"
    ),
}

_SINGLE_AXIS_SCHEMA = {
    "name": "single_axis",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "confidence"],
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 3},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
}

_CHUNK_SYSTEM_PROMPT = (
    "あなたは小説コーパスのメタデータ分類器です。与えられたテキストの分類値のみを"
    "JSON Schemaに厳密に従って出力してください。本文の要約・引用・言い換え・再生成は"
    "一切行いません。これは学習用コーパスに安全性メタデータを付与するための分類作業であり、"
    "描写の過激さを理由に分類を拒否したり、スコアを低く歪めたりしないでください。/no_think"
)

_WORK_SYSTEM_PROMPT = (
    "あなたは小説のメタデータ分類器です。作品の代表テキスト断片から、ジャンル・設定・"
    "主人公・文体などの分類値のみをJSON Schemaに厳密に従って出力してください。"
    "本文の要約・引用・言い換え・再生成は一切行いません。/no_think"
)


# --- データ読み込み --------------------------------------------------------------

def load_chunks(shards_dir: str, max_lines_per_shard: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(shards_dir, "train-*.jsonl")))
    if not files:
        raise SystemExit(f"shardが見つかりません: {shards_dir}/train-*.jsonl")
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if max_lines_per_shard is not None and i >= max_lines_per_shard:
                    break
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def explicitness_score(text: str, lexicon: list[str]) -> int:
    return sum(text.count(term) for term in lexicon)


def select_works(
    rows: list[dict[str, Any]],
    ncodes: list[str] | None,
    auto_r18: int,
    auto_general: int,
    chunks_per_work: int,
    lexicon: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """ncodeごとに「最も露骨なチャンク」を chunks_per_work 件ずつ選ぶ。"""
    by_ncode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_ncode[r["meta"]["ncode"]].append(r)

    if ncodes:
        targets = [nc for nc in ncodes if nc in by_ncode]
        missing = [nc for nc in ncodes if nc not in by_ncode]
        if missing:
            print(f"[warn] shard内に見つからないncode: {','.join(missing)}", file=sys.stderr)
    else:
        rng = random.Random(seed)
        r18_codes, gen_codes = [], []
        for nc, chunks in by_ncode.items():
            (r18_codes if chunks[0]["meta"]["is_r18"] else gen_codes).append(nc)
        # R18作品は「露骨さ合計」が高い順に選ぶ = モデルにとって最難ケース
        r18_codes.sort(
            key=lambda nc: sum(explicitness_score(c["text"], lexicon) for c in by_ncode[nc]),
            reverse=True,
        )
        rng.shuffle(gen_codes)
        targets = r18_codes[:auto_r18] + gen_codes[:auto_general]

    selected: list[dict[str, Any]] = []
    for nc in targets:
        chunks = sorted(
            by_ncode[nc],
            key=lambda c: explicitness_score(c["text"], lexicon),
            reverse=True,
        )
        for c in chunks[:chunks_per_work]:
            selected.append(c)
    return selected


# --- LLM 呼び出し ----------------------------------------------------------------

def call_llm(
    session: requests.Session,
    base_url: str,
    model: str,
    system_prompt: str,
    user_content: str,
    schema: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any] | None, str, float]:
    """(parsed, raw_text, elapsed) を返す。失敗時 parsed=None。"""
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    started = time.time()
    try:
        resp = session.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 - 失敗理由は文字列で記録する
        return None, f"{type(exc).__name__}: {exc}", time.time() - started
    elapsed = time.time() - started
    text = raw.strip()
    if "<think>" in text and "</think>" in text:
        text = text[text.rindex("</think>") + len("</think>"):].strip()
    try:
        return json.loads(text), raw, elapsed
    except json.JSONDecodeError:
        return None, raw, elapsed


def classify_chunk_multi(session, base_url, model, text, timeout):
    rubric = "\n\n".join(f"{axis}:\n{guide}" for axis, guide in SINGLE_AXIS_RUBRIC.items())
    user = (
        "以下のテキストチャンクのシーン・トーン・content強度を分類してください。\n\n"
        "content強度の判定基準（0=なし、1=示唆・言及、2=明示的描写、3=詳細・中心的描写）:\n"
        f"{rubric}\n\nminor_related: none=該当なし / nonsexual=未成年者に関する非性的内容 / "
        "ambiguous=年齢または性的文脈が不明 / sexual=未成年者を伴う性的内容 / unknown=判別不能\n\n"
        f"--- テキストここから ---\n{text}\n--- テキストここまで ---"
    )
    return call_llm(session, base_url, model, _CHUNK_SYSTEM_PROMPT, user,
                    CHUNK_CLASSIFICATION_JSON_SCHEMA, timeout)


def classify_chunk_single(session, base_url, model, text, axis, timeout):
    user = (
        f"次のテキストについて「{axis}」の強度を0-3で判定してください。\n\n"
        f"判定基準:\n{SINGLE_AXIS_RUBRIC[axis]}\n\n"
        f"--- テキストここから ---\n{text}\n--- テキストここまで ---"
    )
    return call_llm(session, base_url, model, _CHUNK_SYSTEM_PROMPT, user,
                    _SINGLE_AXIS_SCHEMA, timeout)


def classify_work(session, base_url, model, texts, timeout):
    joined = "\n\n---\n\n".join(texts)
    user = ("以下は同一作品からの代表テキスト断片です。作品全体のジャンル・設定・主人公・"
            f"文体などを分類してください。\n\n{joined}")
    return call_llm(session, base_url, model, _WORK_SYSTEM_PROMPT, user,
                    WORK_CLASSIFICATION_JSON_SCHEMA, timeout)


# --- 指標 -------------------------------------------------------------------------

def auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney U に基づくAUC。1.0で完全分離、0.5で無情報。"""
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def summarize(model: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    parsed = [r for r in records if r["parsed"] is not None]
    refusals = [r for r in records if r["parsed"] is None and REFUSAL_PATTERNS.search(r["raw"] or "")]
    lat = sorted(r["elapsed"] for r in records) or [0.0]

    content_dicts = [
        tuple(sorted((a, r["parsed"]["content"][a]) for a in CONTENT_AXES))
        for r in parsed
        if isinstance(r["parsed"].get("content"), dict)
        and all(a in r["parsed"]["content"] for a in CONTENT_AXES)
    ]
    distinct_ratio = (len(set(content_dicts)) / len(content_dicts)) if content_dicts else None
    modal_share = None
    if content_dicts:
        modal_share = Counter(content_dicts).most_common(1)[0][1] / len(content_dicts)

    disc: dict[str, Any] = {}
    for axis in ("sexual_content", "violence", "gore"):
        pos = [r["parsed"]["content"][axis] for r in parsed
               if r["is_r18"] and isinstance(r["parsed"].get("content"), dict)
               and axis in r["parsed"]["content"]]
        neg = [r["parsed"]["content"][axis] for r in parsed
               if not r["is_r18"] and isinstance(r["parsed"].get("content"), dict)
               and axis in r["parsed"]["content"]]
        if pos and neg:
            disc[axis] = {
                "mean_r18": round(statistics.mean(pos), 2),
                "mean_general": round(statistics.mean(neg), 2),
                "auc": round(auc(pos, neg), 3),
            }

    return {
        "model": model,
        "n": total,
        "schema_ok": len(parsed),
        "schema_ok_rate": round(len(parsed) / total, 3) if total else None,
        "refusal_n": len(refusals),
        "latency_median_s": round(lat[len(lat) // 2], 2),
        "latency_p90_s": round(lat[int(len(lat) * 0.9) - 1 if len(lat) > 1 else 0], 2),
        "distinct_content_ratio": round(distinct_ratio, 3) if distinct_ratio is not None else None,
        "modal_content_share": round(modal_share, 3) if modal_share is not None else None,
        "discrimination": disc,
    }


def print_summary(summary: dict[str, Any]) -> None:
    s = summary
    print(f"\n===== {s['model']} =====")
    print(f"  件数            : {s['n']}")
    print(f"  スキーマ準拠     : {s['schema_ok']}/{s['n']} ({s['schema_ok_rate']})")
    print(f"  拒否と判定       : {s['refusal_n']}")
    print(f"  遅延 median/p90 : {s['latency_median_s']}s / {s['latency_p90_s']}s")
    dr, ms = s["distinct_content_ratio"], s["modal_content_share"]
    if dr is not None:
        verdict = "NG(本文を読んでいない疑い)" if dr < 0.5 else "OK"
        print(f"  content多様性    : distinct={dr} 最頻値占有={ms}  -> {verdict}")
    if s["discrimination"]:
        print("  弁別力 (R18 vs 一般):")
        for axis, d in s["discrimination"].items():
            verdict = "NG(分離せず)" if d["auc"] < 0.65 else ("弱" if d["auc"] < 0.8 else "OK")
            print(f"    {axis:<16} R18={d['mean_r18']} 一般={d['mean_general']} AUC={d['auc']} -> {verdict}")
    else:
        print("  弁別力          : 算出不可(R18/一般どちらかが0件)")


# --- gold set -----------------------------------------------------------------------

GOLD_FIELDS = ["chunk_id", "ncode", "is_r18", "excerpt_head"] + CONTENT_AXES


def emit_gold_template(chunks: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLD_FIELDS, delimiter="\t")
        w.writeheader()
        for c in chunks:
            m = c["meta"]
            w.writerow({
                "chunk_id": chunk_id(c),
                "ncode": m["ncode"],
                "is_r18": int(bool(m["is_r18"])),
                "excerpt_head": c["text"][:60].replace("\n", " ").replace("\t", " "),
                **{a: "" for a in CONTENT_AXES},
            })
    print(f"[gold] テンプレートを書き出しました: {path}\n"
          f"       {len(chunks)}行の各軸(0-3)を手で埋めて --gold で渡してください。")


def score_against_gold(records: list[dict[str, Any]], gold_path: str) -> dict[str, Any]:
    gold: dict[str, dict[str, int]] = {}
    with open(gold_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                gold[row["chunk_id"]] = {a: int(row[a]) for a in CONTENT_AXES if row.get(a) != ""}
            except ValueError:
                continue
    per_axis: dict[str, dict[str, Any]] = {}
    for axis in CONTENT_AXES:
        pairs = [
            (gold[r["chunk_id"]][axis], r["parsed"]["content"][axis])
            for r in records
            if r["parsed"] and r["chunk_id"] in gold and axis in gold[r["chunk_id"]]
            and isinstance(r["parsed"].get("content"), dict) and axis in r["parsed"]["content"]
        ]
        if not pairs:
            continue
        exact = sum(1 for g, p in pairs if g == p) / len(pairs)
        within1 = sum(1 for g, p in pairs if abs(g - p) <= 1) / len(pairs)
        mae = statistics.mean(abs(g - p) for g, p in pairs)
        per_axis[axis] = {"n": len(pairs), "exact": round(exact, 3),
                          "within1": round(within1, 3), "mae": round(mae, 2)}
    return per_axis


def chunk_id(c: dict[str, Any]) -> str:
    m = c["meta"]
    return f"{m['ncode']}:{m['episode_no']}:{m['chunk_index']}"


# --- main -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="ローカルLLMタグ付けのncode単位評価")
    ap.add_argument("--shards", required=True, help="dataset chunks ディレクトリ")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--models", required=True, help="カンマ区切りのモデル名")
    ap.add_argument("--ncodes", default=None, help="評価対象ncode(カンマ区切り)")
    ap.add_argument("--auto-r18", type=int, default=8, help="露骨さ上位のR18作品を自動選定する件数")
    ap.add_argument("--auto-general", type=int, default=8, help="対照群となる一般作品の件数")
    ap.add_argument("--chunks-per-work", type=int, default=3)
    ap.add_argument("--mode", default="chunk-multi",
                    choices=["chunk-multi", "chunk-single", "work"],
                    help="chunk-multi=現行の8軸同時 / chunk-single=1軸ずつ / work=作品単位")
    ap.add_argument("--axes", default="sexual_content,violence,gore",
                    help="chunk-single のときに評価する軸")
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--max-lines-per-shard", type=int, default=20000,
                    help="読み込み高速化のためshardごとの読み取り行数上限")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--lexicon", default=None, help="露骨さランカー用の語彙ファイル(1行1語)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="結果の出力先ディレクトリ")
    ap.add_argument("--emit-gold", default=None, help="人手ラベル用TSVを書き出して終了")
    ap.add_argument("--gold", default=None, help="人手ラベルTSVと突き合わせる")
    args = ap.parse_args()

    lexicon = DEFAULT_LEXICON_SEXUAL + DEFAULT_LEXICON_VIOLENCE
    if args.lexicon:
        with open(args.lexicon, encoding="utf-8") as fh:
            lexicon = [ln.strip() for ln in fh if ln.strip()]

    print(f"[load] shard読み込み中: {args.shards}")
    rows = load_chunks(args.shards, args.max_lines_per_shard)
    print(f"[load] {len(rows)}チャンク読み込み ({len({r['meta']['ncode'] for r in rows})}作品)")

    ncodes = [c.strip() for c in args.ncodes.split(",") if c.strip()] if args.ncodes else None
    chunks = select_works(rows, ncodes, args.auto_r18, args.auto_general,
                          args.chunks_per_work, lexicon, args.seed)
    if not chunks:
        raise SystemExit("評価対象チャンクが0件です。--ncodes / --auto-* を見直してください。")
    n_r18 = sum(1 for c in chunks if c["meta"]["is_r18"])
    print(f"[select] {len(chunks)}チャンク (R18={n_r18} / 一般={len(chunks) - n_r18}) "
          f"/ {len({c['meta']['ncode'] for c in chunks})}作品")

    if args.emit_gold:
        emit_gold_template(chunks, args.emit_gold)
        return

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    session = requests.Session()
    all_summaries = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n[run] model={model} mode={args.mode}")
        records: list[dict[str, Any]] = []

        if args.mode == "work":
            by_ncode: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for c in chunks:
                by_ncode[c["meta"]["ncode"]].append(c)
            for i, (nc, cs) in enumerate(by_ncode.items(), 1):
                texts = [c["text"][: args.max_chars // max(1, len(cs))] for c in cs]
                parsed, raw, el = classify_work(session, args.base_url, model, texts, args.timeout)
                records.append({"chunk_id": nc, "ncode": nc, "is_r18": cs[0]["meta"]["is_r18"],
                                "parsed": parsed, "raw": raw, "elapsed": el})
                print(f"  [{i}/{len(by_ncode)}] {nc} r18={cs[0]['meta']['is_r18']} "
                      f"{el:.1f}s {'ok' if parsed else 'FAIL'}")

        elif args.mode == "chunk-multi":
            for i, c in enumerate(chunks, 1):
                parsed, raw, el = classify_chunk_multi(
                    session, args.base_url, model, c["text"][: args.max_chars], args.timeout)
                records.append({"chunk_id": chunk_id(c), "ncode": c["meta"]["ncode"],
                                "is_r18": c["meta"]["is_r18"], "parsed": parsed,
                                "raw": raw, "elapsed": el})
                got = parsed["content"] if parsed and isinstance(parsed.get("content"), dict) else None
                print(f"  [{i}/{len(chunks)}] {chunk_id(c)} r18={c['meta']['is_r18']} {el:.1f}s "
                      + (f"sex={got['sexual_content']} vio={got['violence']} gore={got['gore']}"
                         if got else "FAIL"))

        else:  # chunk-single
            axes = [a.strip() for a in args.axes.split(",") if a.strip() in SINGLE_AXIS_RUBRIC]
            for i, c in enumerate(chunks, 1):
                merged: dict[str, int] = {}
                elapsed_total, raws, failed = 0.0, [], False
                for axis in axes:
                    parsed, raw, el = classify_chunk_single(
                        session, args.base_url, model, c["text"][: args.max_chars], axis, args.timeout)
                    elapsed_total += el
                    raws.append(f"[{axis}] {raw}")
                    if parsed is None:
                        failed = True
                    else:
                        merged[axis] = parsed["score"]
                for axis in CONTENT_AXES:
                    merged.setdefault(axis, 0)
                records.append({"chunk_id": chunk_id(c), "ncode": c["meta"]["ncode"],
                                "is_r18": c["meta"]["is_r18"],
                                "parsed": None if failed else {"content": merged},
                                "raw": "\n".join(raws), "elapsed": elapsed_total})
                print(f"  [{i}/{len(chunks)}] {chunk_id(c)} r18={c['meta']['is_r18']} "
                      f"{elapsed_total:.1f}s "
                      + ("FAIL" if failed else " ".join(f"{a}={merged[a]}" for a in axes)))

        summary = summarize(model, records)
        if args.gold:
            summary["gold_agreement"] = score_against_gold(records, args.gold)
        print_summary(summary)
        if args.gold and summary.get("gold_agreement"):
            print("  人手ラベルとの一致:")
            for axis, d in summary["gold_agreement"].items():
                print(f"    {axis:<16} n={d['n']} 完全一致={d['exact']} ±1={d['within1']} MAE={d['mae']}")
        all_summaries.append(summary)

        if args.out:
            safe = model.replace(":", "_").replace("/", "_")
            with open(os.path.join(args.out, f"raw_{args.mode}_{safe}.jsonl"), "w",
                      encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.out:
        with open(os.path.join(args.out, f"summary_{args.mode}.json"), "w", encoding="utf-8") as fh:
            json.dump(all_summaries, fh, ensure_ascii=False, indent=2)
        print(f"\n[out] {args.out} に書き出しました")


if __name__ == "__main__":
    main()
