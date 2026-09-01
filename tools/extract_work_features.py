#!/usr/bin/env python3
"""短編作品の全文をLLMに読ませて構造化特徴を抽出する（PLAN_20260726.md Phase 1-3〜1-6）。

要点:
- タグを直接答えさせない。物語の構造を項目として出させ、それを特徴量／中間表現にする。
- 長編でLLM分類が失敗した原因は「断片しか見せられない」ことだったので、
  **全文が文脈長に収まる作品だけ**を対象にする。収まらないものは処理しない（黙って切らない）。
- 中断前提。1作品ごとに追記保存し、--resume で済みを飛ばす。

ollama の structured outputs（format にJSON Schemaを渡す）を使うので、
出力は文法レベルでスキーマに従う。パース失敗は基本的に起きないが、
項目の欠落やenum外の値は起こりうるので呼び出し側で検証する。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, time as dtime
from pathlib import Path

PROMPT_TEMPLATE = """あなたは日本語の小説を読んで、その構造を客観的に記述する分析者です。

以下に小説の全文を示します。全文を読んだうえで、指定されたJSON形式で分析結果だけを出力してください。

制約:
- ジャンルタグやキーワード（「異世界転生」「ハーレム」等）を答えるのではありません。
  物語の構造そのものを、指定された項目に沿って記述してください。
- 本文に書かれていないことは推測で埋めず、unclear を使ってください。
- setting_era は「架空世界」「中世風」「1940年代」のように短く書いてください。
- confidence は、全文を読んだうえでの判定の確からしさです。

各項目の選択肢の意味（重要: 出力フォーマットには選択肢名だけが見えるが、
判定はここに書いた定義に従うこと。選択肢名の字面の印象で選ばない）:

- world_transfer（異世界との関わり方）:
  reincarnation = 死亡・消滅等により元の肉体を失い、別人・別存在として
    異世界に生まれ変わった（例: 交通事故で死んで赤ん坊として異世界に生まれる）
  transfer = 元の肉体・自我を保ったまま、生きている状態で異世界へ移動・召喚された
    （例: 気づいたら異世界にいた、召喚陣に巻き込まれた）
  returned = 異世界から元の世界へ戻った、またはその途中
  none = 異世界との関わりがない（現代物・現実世界のみが舞台）
  この区別は「死んだかどうか」という本文の描写で判定する。
- ending_type: happy=幸福な結末 / bittersweet=ほろ苦い結末 / tragic=悲劇的結末 /
  open=結末が明示されず読者の想像に委ねられる（連載継続中でこの先が続く場合も含む）/
  cyclical=同じ状況が繰り返される・振り出しに戻る / unresolved=話が完結せず
  尻切れになっている（open との違いは「意図的な余韻」か「単に終わっていない」か）
- protagonist_gender: male/female=明記されている性別 / multiple=主人公が複数いる /
  non_human_or_none=人間でない、または主人公という概念が薄い / unclear=不明
- viewpoint_person: first=一人称 / third_limited=三人称一視点 /
  third_omniscient=三人称神視点（複数人物の内心が描かれる）/ second=二人称 /
  mixed=混在
- protagonist_role（物語上の役割。性格の話ではない）: chosen_hero=選ばれし者・
  運命づけられた英雄 / reluctant_hero=乗り気でないまま巻き込まれた主人公 /
  ordinary_person=特別な使命のない一般人 / villain_or_antihero=悪役・反英雄視点 /
  noble_or_ruler=貴族・支配者 / soldier_or_knight=軍人・騎士 /
  merchant_or_artisan=商人・職人 / servant_or_slave=使用人・奴隷 /
  scholar_or_mage=学者・魔術師 / observer_narrator=物語の傍観者・語り手に徹する役
- story_shape（物語の型。最大3件、重要な順）: coming_of_age=成長物語 /
  revenge=復讐が主軸 / quest_adventure=探索・冒険 / romance_courtship=恋愛・求愛が主軸 /
  slice_of_life=日常物 / mystery_investigation=謎解き・調査 / war_campaign=戦争・軍事作戦 /
  political_intrigue=権謀術数 / survival=生存・サバイバル / tragedy_downfall=没落・破滅 /
  rise_to_power=成り上がり / workplace_or_craft=仕事・職人技が主軸 /
  horror_confrontation=恐怖との対峙 / comedy=コメディ主体 / other=上記に当てはまらない

--- 小説の全文ここから ---
{text}
--- 小説の全文ここまで ---

上記の小説について、JSONで分析結果を出力してください。
"""


def load_schema(path: Path) -> dict:
    schema = json.loads(path.read_text(encoding="utf-8"))
    # ollama/llama.cpp の GBNF 変換に不要なメタキーを落とす
    for k in ("$schema", "$id", "title"):
        schema.pop(k, None)
    return schema


def parse_deadline(s: str | None) -> dtime | None:
    if not s:
        return None
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if not m:
        raise SystemExit(f"--deadline の形式が不正です: {s}")
    return dtime(int(m.group(1)), int(m.group(2)))


def past_deadline(dl: dtime | None) -> bool:
    return dl is not None and datetime.now().time() >= dl


def call_ollama(host: str, model: str, prompt: str, schema: dict, num_ctx: int,
                think: bool, timeout: int) -> tuple[dict, dict]:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "format": schema,
            "options": {"num_ctx": num_ctx, "temperature": 0.0},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}/api/generate", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    stats = {
        "prompt_eval_count": d.get("prompt_eval_count"),
        "prompt_eval_duration_s": round((d.get("prompt_eval_duration") or 0) / 1e9, 2),
        "eval_count": d.get("eval_count"),
        "eval_duration_s": round((d.get("eval_duration") or 0) / 1e9, 2),
        "total_duration_s": round((d.get("total_duration") or 0) / 1e9, 2),
    }
    return json.loads(d["response"]), stats


def validate(obj: dict, schema: dict) -> list[str]:
    """必須項目とenumを再帰検証する（jsonschema依存を避ける）。"""
    problems: list[str] = []

    def check(value: object, spec: dict, path: str) -> None:
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"enum:{path}={value!r}")
        if isinstance(value, dict):
            props = spec.get("properties", {})
            for key in spec.get("required", []):
                if key not in value:
                    problems.append(f"missing:{path}.{key}" if path else f"missing:{key}")
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                child_spec = props.get(key)
                if child_spec is None:
                    problems.append(f"unknown:{child_path}")
                    continue
                check(child, child_spec, child_path)
        elif isinstance(value, list) and "items" in spec:
            for index, child in enumerate(value):
                check(child, spec["items"], f"{path}[{index}]")

    check(obj, schema, "")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="work_texts_{N}.jsonl")
    ap.add_argument("--manifest", required=True, help="works.jsonl")
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", required=True, help="1作品1行のJSONLを追記する")
    ap.add_argument("--host", default="127.0.0.1:11434")
    ap.add_argument("--model", default="general-qwen3-30b")
    ap.add_argument("--num-ctx", type=int, default=65536)
    ap.add_argument("--max-chars", type=int, default=30000, help="この文字数以下の作品だけを対象にする")
    ap.add_argument("--min-chars", type=int, default=0, help="この文字数以上の作品だけを対象にする（長い側の実測用）")
    ap.add_argument("--limit", type=int, default=0, help="0で無制限。試作時に使う")
    ap.add_argument("--think", action="store_true", help="思考トークンを有効にする（遅い）")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--deadline", help="HH:MM。この時刻を過ぎたら新規作品に着手しない")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    schema = load_schema(Path(args.schema))
    deadline = parse_deadline(args.deadline)

    meta: dict[str, dict] = {}
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                meta[rec["ncode"]] = rec

    targets: list[tuple[str, str]] = []
    with open(args.cache, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if args.min_chars <= len(rec["text"]) <= args.max_chars:
                targets.append((rec["ncode"], rec["text"]))
    targets.sort(key=lambda t: len(t[1]))

    done: set[str] = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["ncode"])
                    except (json.JSONDecodeError, KeyError):
                        pass

    todo = [t for t in targets if t[0] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(
        f"対象={len(targets)}件 (<= {args.max_chars}字) / 済={len(done)} / 今回={len(todo)}",
        file=sys.stderr,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for i, (ncode, text) in enumerate(todo, 1):
            if past_deadline(deadline):
                print(f"締切 {args.deadline} に達したので停止します", file=sys.stderr)
                break
            t0 = time.time()
            rec: dict = {
                "ncode": ncode,
                "n_chars": len(text),
                "model": args.model,
                "num_ctx": args.num_ctx,
                "think": args.think,
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                obj, stats = call_ollama(
                    args.host, args.model, PROMPT_TEMPLATE.format(text=text),
                    schema, args.num_ctx, args.think, args.timeout,
                )
                problems = validate(obj, schema)
                rec.update({"ok": True, "features": obj, "problems": problems, "stats": stats})
                n_ok += 1
                status = "OK" if not problems else f"OK(問題{len(problems)}件)"
            except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as e:
                rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
                n_fail += 1
                status = "FAIL"
                stats = {}
            # 1作品ごとに書き出して flush する（中断されても失わない）
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            pe = stats.get("prompt_eval_count")
            print(
                f"[{i}/{len(todo)}] {ncode} {len(text):>6,}字 {status} "
                f"{time.time() - t0:.1f}s prompt_tokens={pe}",
                file=sys.stderr,
            )

    print(f"完了: ok={n_ok} fail={n_fail} -> {out_path}", file=sys.stderr)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
