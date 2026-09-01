#!/usr/bin/env python3
"""あらすじを前置きしたら、生成がそれに従うかを客観的に測る。

**なぜ客観指標が要るか**: 生成文を読んで「あらすじに沿っている気がする」で判断すると、
読み手が勝手に辻褄を合わせて読んでしまう。C-4の視点判定で目視を誤った前例がある。

方法: あらすじに**固有名詞**（人名・地名・道具名）を仕込み、生成文に何個現れるかを数える。
あらすじ無し（対照群）と比べて出現率が上がるかを見る。固有名詞は珍しい語を選ぶので、
偶然一致はほぼ起きない。

注意: このLoRAは「条件行＋物語の途中の断片」だけで学習しており、
**プロンプトにあらすじが入った例を一度も見ていない**。つまりこれは分布外の使い方で、
効かなくても学習の失敗ではない。ここで効かないなら「あらすじを学習時の条件に入れる」
という次の一手の必要性が示される、という位置づけの実験。

**固有名詞の一貫性（2026-07-30 追加、PLAN_20260730.md Phase 2 の前提）**:
再現率（与えた固有名詞が現れたか）だけでは長編の破綻を捉えられない。実際に困るのは
「勝手に知らない人名を作る」「表記が揺れる（ミオ→ミヲ）」「後半で設定を忘れる」の3つ。
そこで次の3指標を足した。

  novel_rate  生成に現れたカタカナ固有名詞のうち、与えた設定に無いものの割合。
              低いほど良い。汚染の指標
  drift       与えた名前に編集距離1で近いが一致しない語の数。表記揺れの指標
  late_rate   生成の**後半だけ**での再現率。前半と比べて落ちるなら、長い生成で
              設定を保てていない

固有名詞の抽出は**カタカナ2文字以上の連続**というヒューリスティックで、形態素解析は
使わない（依存を増やさないため）。したがって漢字表記の人名・地名は拾えず、
novel_rate と drift は**カタカナ名に限った近似**である。この限界を承知の上で、
条件間の相対比較にのみ使うこと。絶対値には意味を持たせない。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import urllib.request

# カタカナ2文字以上の連続。長音符・中黒を含む（例: ヴァルド・グレイ、シャーロット）
KATAKANA_RUN = re.compile(r"[ァ-ヴー]{2,}(?:・[ァ-ヴー]{2,})*")

# 固有名詞ではないが頻出するカタカナ語。除外しないと novel_rate が飽和する。
COMMON_KATAKANA = {
    "ドア", "テーブル", "ベッド", "カーテン", "スカート", "シャツ", "コート", "ポケット",
    "ボタン", "ガラス", "コンクリート", "アスファルト", "エレベーター", "ロビー", "ホール",
    "ソファ", "カップ", "コーヒー", "パン", "ナイフ", "フォーク", "スプーン", "ライト",
    "ランプ", "カメラ", "テレビ", "ラジオ", "メール", "スマホ", "パソコン", "ページ",
    "スーツ", "ネクタイ", "リボン", "ベル", "ドレス", "マント", "ローブ", "ブーツ",
    "レベル", "スキル", "ステータス", "アイテム", "モンスター", "ダンジョン", "ギルド",
    "パーティ", "クエスト", "ポイント", "エネルギー", "イメージ", "リズム", "テンポ",
    "タイミング", "チャンス", "スピード", "パターン", "バランス", "コントロール",
}


def edit_distance_one(a: str, b: str) -> bool:
    """編集距離が1以下か。表記揺れ（ミオ/ミヲ、ガルド/ガルト）の検出用。
    長さ差が2以上なら計算せず False。"""
    if abs(len(a) - len(b)) > 1:
        return False
    if a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(longer)):
        if longer[:i] + longer[i + 1:] == shorter:
            return True
    return False


def extract_katakana_entities(text: str) -> set[str]:
    """生成文からカタカナの固有名詞候補を拾う。一般名詞は除外する。"""
    return {m for m in KATAKANA_RUN.findall(text)
            if len(m) >= 2 and m not in COMMON_KATAKANA}


def entity_consistency(text: str, given: list[str]) -> dict:
    """与えた固有名詞に対する生成側の汚染・表記揺れ・後半での持続を測る。"""
    given_katakana = {e for e in given if KATAKANA_RUN.fullmatch(e)}
    found = extract_katakana_entities(text)
    novel = {f for f in found if f not in given_katakana}
    drift = sorted({f for f in novel
                    if any(edit_distance_one(f, g) for g in given_katakana)})
    half = len(text) // 2
    late_hits = [e for e in given if e in text[half:]]
    return {
        "novel_rate": round(len(novel) / len(found), 3) if found else 0.0,
        "novel_entities": sorted(novel)[:10],
        "drift": drift,
        "late_rate": round(len(late_hits) / len(given), 3) if given else 0.0,
    }


CASES = [
    {
        "synopsis": "灯は旧校舎の地下で銀の鍵を拾い、蒼硝子の扉を開けてしまう。",
        "entities": ["灯", "旧校舎", "銀の鍵", "蒼硝子"],
        "opening": "放課後の廊下は、いつもより静かだった。",
    },
    {
        "synopsis": "傭兵ガルドは、砂の都ネフィリムで盗まれた竜骨剣を追っている。",
        "entities": ["ガルド", "ネフィリム", "竜骨剣"],
        "opening": "市場の喧騒を抜けると、風が砂を運んできた。",
    },
    {
        "synopsis": "星見の少女ミオは、七番目の月が沈む前に岬の灯台へ辿り着かねばならない。",
        "entities": ["ミオ", "七番目の月", "灯台", "岬"],
        "opening": "空を見上げると、雲の切れ間が広がっていた。",
    },
]


def generate(host: str, payload: dict) -> str:
    request = urllib.request.Request(
        f"http://{host}/generate", json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"})
    pieces = []
    with urllib.request.urlopen(request, timeout=1200) as response:
        for raw in response:
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                pieces.append(event["text"])
            elif event["type"] == "error":
                raise RuntimeError(event["message"])
    return "".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1:8760")
    parser.add_argument("--samples", type=int, default=2, help="条件あたり・ケースあたりの本数")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--out")
    args = parser.parse_args()

    modes = [
        ("あらすじ無し", lambda c: c["opening"]),
        ("あらすじ有り", lambda c: f"【あらすじ】{c['synopsis']}\n\n【本文】{c['opening']}"),
    ]
    scores: dict[str, list[float]] = {name: [] for name, _ in modes}
    consistency_scores: dict[str, dict[str, list[float]]] = {
        name: {"novel_rate": [], "drift": [], "late_rate": []} for name, _ in modes}
    records = []

    for case in CASES:
        for name, make_body in modes:
            for i in range(args.samples):
                text = generate(args.host, {
                    "viewpoint": "third_person", "protagonist_gender": "",
                    "setting": [], "genre": [], "r18": "no",
                    "body": make_body(case), "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                })
                hits = [e for e in case["entities"] if e in text]
                rate = len(hits) / len(case["entities"])
                scores[name].append(rate)
                consistency = entity_consistency(text, case["entities"])
                for key in ("novel_rate", "late_rate"):
                    consistency_scores[name][key].append(consistency[key])
                consistency_scores[name]["drift"].append(len(consistency["drift"]))
                records.append({"mode": name, "synopsis": case["synopsis"],
                                "hits": hits, "rate": round(rate, 3),
                                **consistency, "text": text})
                print(f"  [{name}] {case['entities'][0]}… {i + 1}/{args.samples} "
                      f"一致 {len(hits)}/{len(case['entities'])} {hits} "
                      f"新規{consistency['novel_rate']:.2f} 揺れ{len(consistency['drift'])}",
                      file=sys.stderr)

    print("\n=== あらすじの固有名詞が生成文に現れた割合 ===")
    print(f"{'条件':<14}{'平均':>8}{'中央値':>9}{'本数':>6}")
    for name, _ in modes:
        s = scores[name]
        print(f"{name:<14}{statistics.mean(s):>8.3f}{statistics.median(s):>9.3f}{len(s):>6}")
    lift = statistics.mean(scores["あらすじ有り"]) - statistics.mean(scores["あらすじ無し"])
    print(f"\n差（有り - 無し）= {lift:+.3f}")

    # --- 固有名詞の一貫性（カタカナ名に限った近似。絶対値ではなく条件間の差を見る）---
    print("\n=== 固有名詞の一貫性 ===")
    print(f"{'条件':<14}{'新規率↓':>10}{'表記揺れ↓':>11}{'後半再現率↑':>13}")
    for name, _ in modes:
        c = consistency_scores[name]
        print(f"{name:<14}{statistics.mean(c['novel_rate']):>10.3f}"
              f"{statistics.mean(c['drift']):>11.2f}"
              f"{statistics.mean(c['late_rate']):>13.3f}")
    late_gap = (statistics.mean(consistency_scores["あらすじ有り"]["late_rate"])
                - statistics.mean(scores["あらすじ有り"]))
    print(f"\n後半再現率 - 全体再現率（あらすじ有り）= {late_gap:+.3f}")
    if late_gap <= -0.3:
        print("判定: 生成の**後半で設定を落としている**。長編では直前文脈だけでは保たない。"
              "作品要約・直近要約を prompt に入れる（PLAN Phase 2）必要性の裏付けになる")
    else:
        print("判定: 後半でも設定を保てている。この生成長では破綻していない")
    novel_gap = (statistics.mean(consistency_scores["あらすじ有り"]["novel_rate"])
                 - statistics.mean(consistency_scores["あらすじ無し"]["novel_rate"]))
    print(f"新規カタカナ名の割合の差（有り - 無し）= {novel_gap:+.3f}"
          "（正なら、あらすじを与えても知らない名前を作り続けている）")
    if lift >= 0.5:
        print("判定: **あらすじに従っている**。推論時に足すだけで効くので、"
              "パイプライン化の価値がある")
    elif lift >= 0.2:
        print("判定: **部分的に従う**。パイプラインは作れるが、"
              "あらすじを学習時の条件にも入れた方が確実")
    else:
        print("判定: **従っていない**。推論時に足すだけでは効かない。"
              "あらすじを学習時の条件に入れて学習し直す必要がある")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n生成文: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
