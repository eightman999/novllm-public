# control_format.py
"""control_tags を学習テキスト／生成プロンプトの条件行に変換する（PLAN_20260727.md B-4）。

**学習時と推論時で書式が1文字でもずれると条件付けは効かない**ので、
train_lora.py と generate.py の両方がこのモジュールだけを使う。書式を変えるときは
ここだけを直すこと。ここを直したら学習し直しになる点に注意する。

書式:
    [viewpoint=first_person][protagonist_gender=female][setting=isekai,school][genre=romance][r18=no][dialogue=mid][sentence=short]
    本文...

設計判断:
- `unknown` を捨てず1つの値として学習させる。捨てると推論時に「指定しない」が
  表現できなくなる。実測でも viewpoint 4.0% / protagonist_gender 2.3% が unknown で残る。
- キーは常に全部出す。値が空のリストは `none` にして欠落させない。
  フィールドが出たり出なかったりすると、モデルは書式の揺れの方を学んでしまう。
- **`tone` は含めない**。全作品で空であり、定数はトークンを食うだけで情報を持たない
  （Phase L1 で protagonist_count を κ=0.000 で落としたのと同じ理由）。
- **`r18` は含める**（2026-07-27追加）。サイト側のレーティングで
  LLM推定ではない source-of-truth であり、コーパスの17.2%と量も十分。
  ここに入れないと R18 は永久に制御できない。値は yes/no の2値。
- **`dialogue` / `sentence` は文体軸**（2026-07-29追加）。`meta.style_metrics` に
  チャンクごとの連続値が既に計算済みなので、生成コストゼロで足せる。
  ただし連続値のままでは条件にならないので、**コーパスの三分位で3段階に離散化**する。
  境界は build_context_view.py が実データから算出し、
  `style_buckets.json` に残す（決め打ちにすると分布とずれる）。
"""
from __future__ import annotations

# 出力順は固定する。並びが揺れると別の書式として学習される。
FIELDS = ("viewpoint", "protagonist_gender", "setting", "genre", "r18",
          "dialogue", "sentence")
LIST_FIELDS = {"setting", "genre"}
EMPTY = "none"

# 新形式（2026-07-29）の区切り。**この文字列の直後から損失をかける**。
# 目的は「直前の文脈と計画を踏まえて続きを書く」動作の学習なので、
# 文脈側を損失に含めると文脈の丸暗記を学んでしまう。
CONTINUATION_MARKER = "【続き】\n"


def bucket(value: float | None, edges: tuple[float, float],
           labels: tuple[str, str, str] = ("low", "mid", "high")) -> str:
    """連続値を3段階に離散化する。edges は下側・上側の境界（三分位を想定）。"""
    if value is None:
        return "unknown"
    if value < edges[0]:
        return labels[0]
    if value < edges[1]:
        return labels[1]
    return labels[2]


def format_control_prefix(control_tags: dict | None) -> str:
    """control_tags から条件行を作る。末尾に改行を含む。"""
    tags = control_tags or {}
    parts = []
    for field in FIELDS:
        value = tags.get(field)
        if field in LIST_FIELDS:
            # 並びを決定的にする。作品ごとに順序が違うと同じ条件が別物に見える。
            rendered = ",".join(sorted(str(item) for item in (value or []))) or EMPTY
        else:
            rendered = str(value) if value else "unknown"
        parts.append(f"[{field}={rendered}]")
    return "".join(parts) + "\n"


def build_plan_header(work_title: str | None = None, episode_title: str | None = None,
                      terms: list[str] | None = None) -> str:
    """作品・話・固有語の見出しを作る（新形式の「計画」部分）。

    episode_title は 10.8% が裸のナンバリング（`第107話` 等）で情報を持たない。
    そういうものは呼び出し側で None にして渡し、ここでは unknown と出す
    （欠けたまま出すとフィールドの出没をモデルが学んでしまう）。
    """
    lines = [
        f"【作品】{(work_title or 'unknown').strip()}",
        f"【話】{(episode_title or 'unknown').strip()}",
        f"【語】{'、'.join(terms) if terms else 'none'}",
    ]
    return "\n".join(lines) + "\n"


def build_sample(control_tags: dict, work_title: str | None, episode_title: str | None,
                 terms: list[str] | None, context: str, continuation: str) -> tuple[str, str]:
    """新形式の (prompt, completion) を作る。

    prompt 側は損失をかけない。TRL の prompt-completion データセットとして渡すと
    completion 側だけに損失がかかる（completion_only_loss）。
    """
    prompt = (format_control_prefix(control_tags)
              + build_plan_header(work_title, episode_title, terms)
              + f"【ここまで】\n{context}\n"
              + CONTINUATION_MARKER)
    return prompt, continuation


def build_training_text(record: dict) -> str:
    """旧形式（条件行＋本文）。チャンク単位の学習に使う。

    r18 は control_tags ではなく meta.is_r18 にあるのでここで合流させる。
    """
    tags = dict(record.get("control_tags") or {})
    tags["r18"] = "yes" if (record.get("meta") or {}).get("is_r18") else "no"
    return format_control_prefix(tags) + str(record.get("text") or "")


def build_prompt(viewpoint: str | None = None, protagonist_gender: str | None = None,
                 setting: list[str] | None = None, genre: list[str] | None = None,
                 r18: str | None = None, dialogue: str | None = None,
                 sentence: str | None = None, work_title: str | None = None,
                 episode_title: str | None = None, terms: list[str] | None = None,
                 context: str | None = None, body: str = "") -> str:
    """推論用。学習時と同じ関数を通すので書式ずれが起きない。

    context を渡すと新形式（計画＋直前文脈＋続き）、渡さなければ旧形式になる。
    """
    tags = {"viewpoint": viewpoint, "protagonist_gender": protagonist_gender,
            "setting": setting, "genre": genre, "r18": r18,
            "dialogue": dialogue, "sentence": sentence}
    if context is None and work_title is None and episode_title is None and not terms:
        return format_control_prefix(tags) + body
    prompt, _ = build_sample(tags, work_title, episode_title, terms, context or "", "")
    return prompt + body
