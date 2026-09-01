#!/usr/bin/env python3
"""条件付けが実際に効いているかを客観指標で測る（PLAN_20260727.md C-4）。

**目視で判定してはいけない**。日本語は主語を落とすので、生成文を読んで
「一人称っぽい/三人称っぽい」を人が判定すると、読み手の期待に引きずられる。

代わりに A-0 で作ったルール（地の文＝鉤括弧の外の一人称代名詞の比率）を
そのまま生成文に当てる。学習データの視点ラベルを作ったのと同じ物差しなので、
「学習した条件が出力に出ているか」を同じ尺度で測れる。

条件ごとに N 本生成し、指標の分布を比較する。1本だけ見て判断しない
（サンプリングありの生成は本質的にばらつくため）。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from tools.detect_viewpoint_rule import PRONOUN_PATTERN, strip_dialogue  # noqa: E402

PROMPTS = [
    "朝の教室に足を踏み入れると、",
    "森の奥に光る扉があった。",
    "その報せが届いたのは、雨の降る夕方だった。",
]


def generate(host: str, payload: dict) -> str:
    request = urllib.request.Request(
        f"http://{host}/generate", json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"})
    text = []
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                text.append(event["text"])
            elif event["type"] == "error":
                raise RuntimeError(event["message"])
    return "".join(text)


def first_person_rate(text: str) -> float:
    """地の文1000字あたりの一人称代名詞の数。学習データのラベル付けと同じ尺度。"""
    narration = strip_dialogue(text)
    if not narration:
        return 0.0
    return len(PRONOUN_PATTERN.findall(narration)) / len(narration) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1:8760")
    parser.add_argument("--samples", type=int, default=6, help="条件あたりの生成本数")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--out", help="生成文を保存するJSONL")
    args = parser.parse_args()

    conditions = [("first_person", "一人称を指定"), ("third_person", "三人称を指定")]
    results: dict[str, list[float]] = {}
    records = []

    for value, label in conditions:
        rates = []
        for i in range(args.samples):
            body = PROMPTS[i % len(PROMPTS)]
            text = generate(args.host, {
                "viewpoint": value, "protagonist_gender": "female",
                "setting": ["school"], "genre": ["school"], "r18": "no",
                "body": body, "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            })
            rate = first_person_rate(text)
            rates.append(rate)
            records.append({"viewpoint": value, "prompt": body,
                            "rate_per_1000": round(rate, 3), "text": text})
            print(f"  [{value}] {i + 1}/{args.samples} 一人称率={rate:6.2f}", file=sys.stderr)
        results[value] = rates
        print(f"{label}: 平均 {statistics.mean(rates):.2f} "
              f"中央値 {statistics.median(rates):.2f}", file=sys.stderr)

    print("\n=== 結果: 地の文1000字あたりの一人称代名詞 ===")
    print(f"{'指定':<16}{'平均':>8}{'中央値':>9}{'最小':>8}{'最大':>8}")
    for value, _ in conditions:
        r = results[value]
        print(f"{value:<16}{statistics.mean(r):>8.2f}{statistics.median(r):>9.2f}"
              f"{min(r):>8.2f}{max(r):>8.2f}")

    first, third = results["first_person"], results["third_person"]
    gap = statistics.mean(first) - statistics.mean(third)
    print(f"\n差（first - third）= {gap:+.2f}")
    # 学習データ側の分離は 三人称 0.09〜0.31 / 一人称 1.70〜8.71 だった（A-0実測）。
    # 条件が効いているなら、生成側にも同じ向きの差が出るはず。
    print("学習データでの分離: 三人称 0.09〜0.31 / 一人称 1.70〜8.71（A-0実測）")
    if gap > 1.0:
        print("判定: **条件が効いている**（向きも大きさも学習データと整合）")
    elif gap > 0.3:
        print("判定: 弱いが効いている兆候あり。サンプル数を増やして再確認する")
    else:
        print("判定: **効いていない**。ステップを増やしても直らない可能性が高く、"
              "LoRAのrank・target_modules・条件行の与え方を見直す必要がある")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\n生成文: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
