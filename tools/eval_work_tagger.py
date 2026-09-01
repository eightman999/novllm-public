#!/usr/bin/env python3
# tools/eval_work_tagger.py
"""作品(ncode)単位のジャンル分類をローカルLLMで行い、精度を実測する。

評価の考え方:
  作者がDBに登録したキーワード(control_flags)を silver label として使い、
  「キーワードがある作品」でLLMがそれを再現できるかを測る。人手ラベルを
  作らずに精度が出せる。再現できるなら、キーワードが0件の作品(実測21.4%)を
  LLMで埋める根拠になる。

指標の読み方:
  recall    : 作者が申告したタグをLLMが拾えた割合。こちらが主指標。
  precision : 参考値。作者の申告漏れ(戦記なのに「オリジナル戦記」を付けない等)が
              多いため、LLMの「誤検出」が実際は正しい場合がある。低くても即NGでは
              なく、--show-fp で中身を見て判断すること。

使い方:
    # キーワード有り作品60件で精度測定
    python tools/eval_work_tagger.py \\
        --shards ./data/chunks \\
        --manifest ./data/manifests/works.jsonl \\
        --models qwen3-4b:latest --eval-n 60 --out runs/work_tag

    # 精度に納得したら、キーワード0件の作品を埋める
    python tools/eval_work_tagger.py ... --fill --fill-n 233 --out runs/work_fill
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# build_dataset.py と同一の語彙を正本として読み込む(二重定義を避ける)
def _load_genre_vocab() -> dict[str, str]:
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build_dataset.py")
    spec = importlib.util.spec_from_file_location("_bd", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return dict(mod._GENRE_FLAG_TO_TAG)


GENRE_FLAG_TO_TAG = _load_genre_vocab()
TAG_TO_FLAG = {v: k for k, v in GENRE_FLAG_TO_TAG.items()}
GENRE_TAGS = sorted(TAG_TO_FLAG)

# 各タグの判定基準。曖昧なタグ(戦記4種の区別など)は明示しないと当たらない。
TAG_RUBRIC = {
    "original_war_chronicle": "架空の世界・国家を舞台に、軍事作戦や戦争の推移そのものを描く",
    "fictional_war_chronicle": "現実の歴史・兵器をベースに、史実とは異なる戦争を描く",
    "if_war_chronicle": "実在の戦争の特定局面を改変した「もしも」を描く",
    "military": "軍隊・兵器・軍事組織の描写が中心にある",
    "war": "戦争が物語の主要な背景または主題である",
    "harem": "主人公が複数の異性(同性)から好意を寄せられる関係が主軸",
    "romance": "恋愛関係の進展が物語の主軸",
    "school": "学校・学園が主要な舞台",
    "contemporary": "現代の現実世界が主要な舞台",
    "historical": "実在の歴史時代が舞台",
    "sf": "科学技術・宇宙・未来社会を扱う",
    "horror": "恐怖・戦慄を主要な効果として狙う",
    "mystery": "謎解き・推理が物語の主軸",
    "bl": "男性同士の恋愛が主題",
    "gl": "女性同士の恋愛が主題",
    "ts": "性別が転換する(性転換・TSF)要素がある",
    "ensemble_cast": "単一の主人公ではなく複数の視点人物を並列に描く",
}

_WORK_GENRE_SCHEMA = {
    "name": "work_genre",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["genre", "confidence"],
        "properties": {
            "genre": {
                "type": "array",
                "items": {"type": "string", "enum": GENRE_TAGS},
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
}

_SYSTEM_PROMPT = (
    "あなたは小説コーパスのジャンル分類器です。与えられた作品の代表テキスト断片から、"
    "指定された語彙の中から当てはまるジャンルタグだけを選んでJSON Schemaに厳密に従って"
    "出力してください。本文の要約・引用・言い換え・再生成は一切行いません。"
    "該当するものが無ければ空配列を返してください。当てはまらないタグを推測で足さないでください。"
    "これは学習用コーパスへのメタデータ付与であり、描写の過激さを理由に分類を拒否しないでください。"
    "/no_think"
)


def build_user_prompt(texts: list[str]) -> str:
    vocab = "\n".join(f"- {t}: {TAG_RUBRIC[t]}" for t in GENRE_TAGS)
    joined = "\n\n---\n\n".join(texts)
    return (
        f"選択可能なジャンルタグ:\n{vocab}\n\n"
        "以下は同一作品からの代表テキスト断片です。この作品に当てはまるタグを選んでください。\n\n"
        f"{joined}"
    )


_SINGLE_TAG_SCHEMA = {
    "name": "single_tag",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["applies", "confidence"],
        "properties": {
            "applies": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
}


def build_single_tag_prompt(texts: list[str], tag: str, title: str | None = None) -> str:
    joined = "\n\n---\n\n".join(texts)
    head = f"作品タイトル: {title}\n\n" if title else ""
    return (
        f"判定対象のタグ: {tag}\n"
        f"このタグの定義: {TAG_RUBRIC[tag]}\n\n"
        f"{head}"
        "以下は同一作品からの代表テキスト断片です。この作品にこのタグが当てはまるかを"
        "true/falseで判定してください。断片に根拠が無ければ false にしてください。\n\n"
        f"{joined}"
    )


def call_llm(session, base_url, model, user_content, timeout, schema=None):
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_schema", "json_schema": schema or _WORK_GENRE_SCHEMA},
    }
    started = time.time()
    try:
        resp = session.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}", time.time() - started
    elapsed = time.time() - started
    text = raw.strip()
    if "<think>" in text and "</think>" in text:
        text = text[text.rindex("</think>") + len("</think>"):].strip()
    try:
        return json.loads(text), raw, elapsed
    except json.JSONDecodeError:
        return None, raw, elapsed


def silver_tags(work: dict[str, Any]) -> set[str]:
    flags = work.get("control_flags") or {}
    return {GENRE_FLAG_TO_TAG[f] for f, on in flags.items() if on and f in GENRE_FLAG_TO_TAG}


def load_texts_by_ncode(shards_dir: str, wanted: set[str], per_work: int,
                        max_lines_per_shard: int | None) -> dict[str, list[str]]:
    """作品ごとに冒頭・中盤・終盤へ散らしたチャンクを集める。"""
    buckets: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(shards_dir, "train-*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if max_lines_per_shard is not None and i >= max_lines_per_shard:
                    break
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                nc = r["meta"]["ncode"]
                if nc in wanted:
                    buckets[nc].append((r["meta"]["episode_no"], r["text"]))
    out: dict[str, list[str]] = {}
    for nc, items in buckets.items():
        items.sort(key=lambda x: x[0])
        if len(items) <= per_work:
            picked = [t for _, t in items]
        else:
            step = len(items) / per_work
            picked = [items[min(int(k * step), len(items) - 1)][1] for k in range(per_work)]
        out[nc] = picked
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser(description="作品単位ジャンル分類の精度実測")
    ap.add_argument("--shards", required=True)
    ap.add_argument("--manifest", required=True, help="manifests/works.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--models", required=True)
    ap.add_argument("--eval-n", type=int, default=40, help="精度測定に使う「キーワード有り」作品数")
    ap.add_argument("--min-silver-tags", type=int, default=1,
                    help="silver labelがこの件数以上ある作品だけ評価に使う")
    ap.add_argument("--fill", action="store_true", help="キーワード0件の作品を推論して埋める")
    ap.add_argument("--fill-n", type=int, default=0, help="0で全件")
    ap.add_argument("--chunks-per-work", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--max-lines-per-shard", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single-tag", default=None,
                    help="17タグ同時ではなくタグごとにtrue/falseで個別に聞く(カンマ区切り)。"
                         "指定したタグを持つ作品と持たない作品を同数ずつ選んで評価する")
    ap.add_argument("--use-title", action="store_true",
                    help="作品タイトルもプロンプトに含める(構造的タグの検出率が上がるか検証用)")
    ap.add_argument("--show-fp", type=int, default=8, help="誤検出タグの実例をこの件数まで表示")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    works = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    rng = random.Random(args.seed)

    single_tags = None
    if args.single_tag:
        single_tags = [t.strip() for t in args.single_tag.split(",") if t.strip() in TAG_RUBRIC]
        unknown = [t.strip() for t in args.single_tag.split(",") if t.strip() not in TAG_RUBRIC]
        if unknown:
            raise SystemExit(f"未知のタグ: {','.join(unknown)}\n選択可能: {','.join(GENRE_TAGS)}")

    if args.fill:
        targets = [w for w in works if not w.get("normalized_keywords")]
        rng.shuffle(targets)
        if args.fill_n:
            targets = targets[: args.fill_n]
        print(f"[fill] キーワード0件の作品 {len(targets)}件を対象にします")
    elif single_tags:
        want = set(single_tags)
        pos = [w for w in works if silver_tags(w) & want]
        neg = [w for w in works if silver_tags(w) and not (silver_tags(w) & want)]
        rng.shuffle(pos); rng.shuffle(neg)
        half = max(1, args.eval_n // 2)
        targets = pos[:half] + neg[:half]
        rng.shuffle(targets)
        print(f"[eval] 個別判定モード tags={','.join(single_tags)} / "
              f"該当作品{len(pos)}件中{len(pos[:half])}件 + 対照群{len(neg[:half])}件")
    else:
        pool = [w for w in works if len(silver_tags(w)) >= args.min_silver_tags]
        rng.shuffle(pool)
        targets = pool[: args.eval_n]
        print(f"[eval] silver label {args.min_silver_tags}件以上の作品 {len(pool)}件から "
              f"{len(targets)}件を評価に使用")

    wanted = {w["ncode"] for w in targets}
    print(f"[load] 本文チャンク収集中 ({len(wanted)}作品)...")
    texts_by_ncode = load_texts_by_ncode(args.shards, wanted, args.chunks_per_work,
                                         args.max_lines_per_shard)
    targets = [w for w in targets if w["ncode"] in texts_by_ncode]
    print(f"[load] 本文が取れた作品: {len(targets)}件")
    if not targets:
        raise SystemExit("本文が取得できませんでした。--shards を確認してください。")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
    session = requests.Session()

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n[run] model={model}")
        records = []
        per_chunk_budget = args.max_chars // args.chunks_per_work
        for i, w in enumerate(targets, 1):
            nc = w["ncode"]
            texts = [t[:per_chunk_budget] for t in texts_by_ncode[nc]]
            raw = None
            if single_tags:
                pred, el, failed = set(), 0.0, False
                for tag in single_tags:
                    p, raw, e = call_llm(session, args.base_url, model,
                                         build_single_tag_prompt(
                                             texts, tag,
                                             w.get("title") if args.use_title else None),
                                         args.timeout, _SINGLE_TAG_SCHEMA)
                    el += e
                    if p is None:
                        failed = True
                    elif p.get("applies"):
                        pred.add(tag)
                if failed:
                    pred = None
            else:
                parsed, raw, el = call_llm(session, args.base_url, model,
                                           build_user_prompt(texts), args.timeout)
                pred = set(parsed["genre"]) if parsed and isinstance(parsed.get("genre"), list) else None
            gold = silver_tags(w)
            if single_tags:
                gold = gold & set(single_tags)
            records.append({"ncode": nc, "title": w.get("title"), "is_r18": w.get("is_r18"),
                            "gold": sorted(gold), "pred": sorted(pred) if pred is not None else None,
                            "raw": raw if pred is None else None, "elapsed": el})
            if args.fill:
                print(f"  [{i}/{len(targets)}] {nc} {el:.1f}s -> "
                      + (",".join(sorted(pred)) if pred else "FAIL/空"))
            else:
                mark = "" if pred is None else (
                    "○" if pred == gold else ("△" if pred & gold else "×"))
                print(f"  [{i}/{len(targets)}] {nc} {el:.1f}s {mark} "
                      f"gold={','.join(sorted(gold))} pred="
                      + (",".join(sorted(pred)) if pred is not None else "FAIL"))

        ok = [r for r in records if r["pred"] is not None]
        lat = sorted(r["elapsed"] for r in records) or [0.0]
        print(f"\n===== {model} =====")
        print(f"  スキーマ準拠 : {len(ok)}/{len(records)}")
        print(f"  遅延 median  : {lat[len(lat)//2]:.2f}s  (全{len(records)}件 "
              f"{sum(r['elapsed'] for r in records)/60:.1f}分)")

        if args.fill:
            c = Counter(t for r in ok for t in r["pred"])
            empty = sum(1 for r in ok if not r["pred"])
            print(f"  タグ0件だった作品: {empty}/{len(ok)}")
            print(f"  付与タグ分布: {c.most_common()}")
        else:
            tp = fp = fn = 0
            per_tag = defaultdict(lambda: [0, 0, 0])
            fp_examples = []
            for r in ok:
                g, p = set(r["gold"]), set(r["pred"])
                tp += len(g & p); fp += len(p - g); fn += len(g - p)
                for t in g & p: per_tag[t][0] += 1
                for t in p - g:
                    per_tag[t][1] += 1
                    if len(fp_examples) < args.show_fp:
                        fp_examples.append((r["ncode"], r["title"], t))
                for t in g - p: per_tag[t][2] += 1
            P, R, F = prf(tp, fp, fn)
            exact = sum(1 for r in ok if set(r["gold"]) == set(r["pred"])) / len(ok) if ok else 0
            partial = sum(1 for r in ok if set(r["gold"]) & set(r["pred"])) / len(ok) if ok else 0
            print(f"  micro  precision={P:.3f} recall={R:.3f} F1={F:.3f}  (tp={tp} fp={fp} fn={fn})")
            print(f"  完全一致={exact:.3f}  部分一致(1つ以上重なる)={partial:.3f}")
            print("  タグ別 (tp/fp/fn, recall):")
            for t in sorted(per_tag, key=lambda x: -(per_tag[x][0] + per_tag[x][2])):
                a, b, c = per_tag[t]
                rec = a / (a + c) if a + c else 0.0
                print(f"    {t:<26} {a:>3}/{b:>3}/{c:>3}  recall={rec:.2f}")
            if fp_examples:
                print("  誤検出の実例(作者の申告漏れの可能性あり。中身を見て判断すること):")
                for nc, title, t in fp_examples:
                    print(f"    {nc} 「{title}」 -> {t}")

        if args.out:
            safe = model.replace(":", "_").replace("/", "_")
            mode = "fill" if args.fill else "eval"
            with open(os.path.join(args.out, f"{mode}_{safe}.jsonl"), "w", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"\n[out] {args.out} に書き出しました")


if __name__ == "__main__":
    main()
