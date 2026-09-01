#!/usr/bin/env python3
# tools/classify_works.py
"""作品単位のジャンル分類を TF-IDF(文字n-gram) + ロジスティック回帰 で行う。

LLMに作品全体の構造を尋ねる方式は実測で harem recall=0.14 / 戦記 recall=0.00 と
機能しなかった。断片しか見せられないのが原因なので、こちらは作品全文の統計量を
特徴量にする。作者が申告したキーワード(control_flags)が教師ラベルになる。

PU learning について:
  「タグが付いていない = そのジャンルではない」は偽であることが実測で判明している
  (TS作品にTSタグが無い実例を確認済み)。よって負例は「負例」ではなく「未ラベル」で
  あり、素朴な二値分類は系統的に過小評価する。Elkan-Noto法で観測確率 g(x) を
  真の確率 g(x)/c に補正し、cは交差検証した正例上の平均から推定する。
  補正後スコアが高い未ラベル作品は「作者の付け忘れ候補」として出力する。

中断と再開:
  処理は3段階でそれぞれディスクにチェックポイントを書く。
    1. 本文抽出   -> cache/work_texts.jsonl   (最も重い。2回目以降はスキップ)
    2. ベクトル化 -> cache/tfidf.npz
    3. タグ別評価 -> <out>/results.jsonl      (1タグごとに追記)
  --resume を付けると既に終わったタグを飛ばす。--deadline HH:MM を過ぎたら
  次のタグに入らず正常終了するので、電源を落とす時刻を渡しておけば安全に止まる。

使い方:
    python tools/classify_works.py \\
        --shards ./data/chunks \\
        --manifest ./data/manifests/works.jsonl \\
        --cache ./data/cache \\
        --out runs/work_clf --deadline 21:45 --resume
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_genre_vocab() -> dict[str, str]:
    """build_dataset.py の語彙を正本として読む(二重定義を避ける)。"""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build_dataset.py")
    spec = importlib.util.spec_from_file_location("_bd", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return dict(mod._GENRE_FLAG_TO_TAG)


GENRE_FLAG_TO_TAG = load_genre_vocab()


def silver_tags(work: dict[str, Any]) -> set[str]:
    """build_dataset.py の17ジャンル語彙に写像した正解タグ。"""
    flags = work.get("control_flags") or {}
    return {GENRE_FLAG_TO_TAG[f] for f, on in flags.items() if on and f in GENRE_FLAG_TO_TAG}


def raw_keyword_tags(work: dict[str, Any]) -> set[str]:
    """作者申告キーワードをそのまま正解タグとして使う(--vocab auto)。

    control_flags の23語彙は延べキーワードの22.6%しか拾っておらず、
    「ふたなり」「調教」「中出し」「ファンタジー」「チート」等の性癖・
    設定タグを全部捨てている。これらも作者の自己申告なので教師ラベルに
    使えるが、17ジャンル語彙とは異なり人手のスラッグ定義が無いため、
    キーワード文字列をそのままタグ名として扱う。
    """
    return set(work.get("normalized_keywords") or [])


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


# --- 段階1: 本文抽出 ------------------------------------------------------------

def extract_work_texts(shards_dir: str, cache_path: str, chars_per_work: int) -> dict[str, str]:
    """作品ごとに全話から均等サンプリングして chars_per_work 文字まで連結する。"""
    if os.path.exists(cache_path):
        log(f"段階1: キャッシュを使用 {cache_path}")
        out = {}
        with open(cache_path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[r["ncode"]] = r["text"]
        log(f"段階1: {len(out)}作品を読み込み")
        return out

    log("段階1: shardから本文抽出中(初回のみ・数分かかります)")
    buckets: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    files = sorted(glob.glob(os.path.join(shards_dir, "*.jsonl")))
    for n, path in enumerate(files, 1):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = r["meta"]
                buckets[m["ncode"]].append((m["episode_no"], m["chunk_index"], r["text"]))
        if n % 20 == 0:
            log(f"  {n}/{len(files)} shard  作品{len(buckets)}件")

    out: dict[str, str] = {}
    for nc, items in buckets.items():
        items.sort(key=lambda x: (x[0], x[1]))
        total = sum(len(t) for _, _, t in items)
        if total <= chars_per_work:
            picked = [t for _, _, t in items]
        else:
            # 作品全体へ均等に散らす。冒頭だけ見ると構造タグを取り逃す。
            keep = max(1, chars_per_work // max(1, (total // len(items))))
            step = len(items) / keep
            picked = [items[min(int(k * step), len(items) - 1)][2] for k in range(keep)]
        out[nc] = "\n".join(picked)[:chars_per_work]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for nc, text in out.items():
            fh.write(json.dumps({"ncode": nc, "text": text}, ensure_ascii=False) + "\n")
    os.replace(tmp, cache_path)
    log(f"段階1: {len(out)}作品を抽出 -> {cache_path}")
    return out


# --- 段階2: ベクトル化 ----------------------------------------------------------

def vectorize(texts: list[str], cache_dir: str, ngram: tuple[int, int],
              max_features: int, min_df: int, chars_per_work: int):
    # chars_per_work をキーに含めないと、本文量を変えたつもりで古い行列を
    # 再利用してしまう(実際に踏んだ。15万字の実験が6万字の結果と完全一致した)
    npz = os.path.join(
        cache_dir, f"tfidf_{ngram[0]}{ngram[1]}_{max_features}_c{chars_per_work}.npz")
    if os.path.exists(npz):
        log(f"段階2: キャッシュを使用 {npz}")
        return sp.load_npz(npz)
    log(f"段階2: TF-IDF計算中 char {ngram} max_features={max_features}")
    t = time.time()
    vec = TfidfVectorizer(analyzer="char", ngram_range=ngram, max_features=max_features,
                          min_df=min_df, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(texts)
    os.makedirs(cache_dir, exist_ok=True)
    sp.save_npz(npz, X)
    log(f"段階2: shape={X.shape} nnz={X.nnz} {time.time()-t:.0f}秒 -> {npz}")
    return X


# --- 段階3: タグ別の評価 --------------------------------------------------------

def eval_tag(X, y: np.ndarray, folds: int, seed: int, C: float,
             calibrate: bool = True) -> dict[str, Any]:
    """交差検証で out-of-fold 予測を作り、通常評価とPU補正の両方を返す。

    順位付け(AUC/F1)には class_weight="balanced" のモデルを使う。候補抽出の順位は
    較正済み確率(oof_cal)側を使う。
    PU補正の c 推定には較正済み確率が必要で、class_weight を付けると確率が
    歪んで有病率が過大に出るため、別途 class_weight なし + Platt scaling
    (CalibratedClassifierCV) のモデルを回して確率だけそちらから取る。
    """
    n_pos = int(y.sum())
    skf = StratifiedKFold(n_splits=min(folds, n_pos), shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    oof_cal = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="liblinear")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        if calibrate:
            base = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
            # 内側3-foldでPlatt scaling。正例が少ないタグでは分割数を落とす
            inner = min(3, max(2, int(y[tr].sum())))
            cal = CalibratedClassifierCV(base, method="sigmoid", cv=inner)
            cal.fit(X[tr], y[tr])
            oof_cal[te] = cal.predict_proba(X[te])[:, 1]
    if not calibrate:
        oof_cal = oof

    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)

    # 閾値をF1最大で選ぶ
    order = np.argsort(-oof)
    tp = fp = 0
    best = (0.0, 0.5, 0, 0, 0)
    for rank, idx in enumerate(order, 1):
        if y[idx]:
            tp += 1
        else:
            fp += 1
        prec = tp / rank
        rec = tp / n_pos
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[0]:
            best = (f1, float(oof[idx]), tp, fp, n_pos - tp)

    # Elkan-Noto: c = P(観測ラベル=1 | 真に正) を較正済み確率の正例平均で推定
    c = float(oof_cal[y == 1].mean())
    corrected = np.clip(oof_cal / c, 0.0, 1.0) if c > 0 else oof_cal
    est_prevalence = float(corrected.mean())
    # 較正の妥当性チェック: 較正済み確率の全体平均は申告率に近いはず。
    # 大きく外れていたら c 推定＝有病率推定は信用できない。
    calib_mean = float(oof_cal.mean())
    silver_rate = n_pos / len(y)
    calib_ok = bool(0.5 * silver_rate <= calib_mean <= 3.0 * silver_rate)

    # 未ラベルのうち補正スコアが高いもの = 作者の付け忘れ候補。
    # 並べ替えは clip 前の較正済み確率 oof_cal で行う。corrected = clip(oof_cal/c, 0, 1)
    # は c が小さいタグ(実測 c=0.02 等)で大半が 1.0 に飽和し、argsort が同値の塊を
    # インデックス順で返すため順位情報が失われていた(実測 98.3% の候補が 1.0 に飽和)。
    # oof_cal は corrected の clip 前と単調同値なので、順位はこちらが正しい。
    unlabeled = np.where(y == 0)[0]
    cand = unlabeled[np.argsort(-oof_cal[unlabeled])]

    return {
        "n_pos": n_pos, "n_total": int(len(y)),
        "auc": round(float(auc), 4), "ap": round(float(ap), 4),
        "best_f1": round(best[0], 4), "threshold": round(best[1], 4),
        "tp": best[2], "fp": best[3], "fn": best[4],
        "precision": round(best[2] / (best[2] + best[3]), 4) if best[2] + best[3] else 0.0,
        "recall": round(best[2] / n_pos, 4),
        "pu_c": round(c, 4),
        "silver_prevalence": round(silver_rate, 4),
        "pu_estimated_prevalence": round(est_prevalence, 4),
        "calibrated_mean": round(calib_mean, 4),
        "calibration_ok": calib_ok,
        "_oof": oof, "_oof_cal": oof_cal, "_corrected": corrected, "_candidates": cand,
    }


def main() -> None:
    ap_ = argparse.ArgumentParser(description="TF-IDF + PU learning による作品ジャンル分類")
    ap_.add_argument("--shards", required=True)
    ap_.add_argument("--manifest", required=True)
    ap_.add_argument("--cache", default="./cache")
    ap_.add_argument("--out", default="./runs/work_clf")
    ap_.add_argument("--chars-per-work", type=int, default=60000,
                     help="1作品あたりに使う本文文字数の上限(作品全体から均等サンプリング)")
    ap_.add_argument("--ngram-min", type=int, default=2)
    ap_.add_argument("--ngram-max", type=int, default=3)
    ap_.add_argument("--max-features", type=int, default=200000)
    ap_.add_argument("--min-df", type=int, default=3)
    ap_.add_argument("--folds", type=int, default=5)
    ap_.add_argument("--C", type=float, default=1.0)
    ap_.add_argument("--min-pos", type=int, default=15, help="正例がこの数未満のタグは評価しない")
    ap_.add_argument("--tags", default=None, help="評価するタグ(カンマ区切り、既定は全部)")
    ap_.add_argument("--top-candidates", type=int, default=20,
                     help="タグごとに出力する「付け忘れ候補」の件数")
    ap_.add_argument("--seed", type=int, default=42)
    ap_.add_argument("--vocab", default="genre", choices=["genre", "auto", "both"],
                     help="genre=build_dataset.pyの17ジャンル語彙 / "
                          "auto=作者申告キーワードをそのままタグにする / both=両方")
    ap_.add_argument("--vocab-min-works", type=int, default=40,
                     help="--vocab auto のとき、この件数以上の作品に付いたキーワードだけ対象にする")
    ap_.add_argument("--grid", action="store_true",
                     help="--tags のタグについて ngram/max_features/C を総当たりしてAUCを比較する")
    ap_.add_argument("--grid-vecs", default="2-3-200000,2-4-300000,1-3-300000",
                     help="grid用のベクトル化設定 ngramMin-ngramMax-maxFeatures をカンマ区切り")
    ap_.add_argument("--grid-cs", default="0.3,1.0,3.0")
    ap_.add_argument("--no-calibrate", action="store_true",
                     help="Platt scalingを行わない(高速だがPU有病率は信用できない)")
    ap_.add_argument("--resume", action="store_true", help="results.jsonl に済のタグを飛ばす")
    ap_.add_argument("--deadline", default=None,
                     help="HH:MM。この時刻を過ぎたら次のタグに入らず正常終了する")
    args = ap_.parse_args()

    deadline = None
    if args.deadline:
        hh, mm = (int(v) for v in args.deadline.split(":"))
        now = datetime.datetime.now()
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= now:
            deadline += datetime.timedelta(days=1)
        log(f"締切: {deadline:%Y-%m-%d %H:%M}（残り{(deadline-now).total_seconds()/60:.0f}分）")

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.jsonl")
    done: set[str] = set()
    if args.resume and os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as fh:
            done = {json.loads(l)["tag"] for l in fh}
        log(f"再開: 済{len(done)}タグをスキップ ({','.join(sorted(done))})")

    works = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    texts_by_ncode = extract_work_texts(
        args.shards,
        os.path.join(args.cache, f"work_texts_{args.chars_per_work}.jsonl"),
        args.chars_per_work)
    works = [w for w in works if w["ncode"] in texts_by_ncode]
    log(f"対象作品: {len(works)}件  平均{sum(len(texts_by_ncode[w['ncode']]) for w in works)//len(works)}文字")

    if args.vocab == "genre":
        silver = [silver_tags(w) for w in works]
    elif args.vocab == "auto":
        silver = [raw_keyword_tags(w) for w in works]
    else:
        silver = [silver_tags(w) | raw_keyword_tags(w) for w in works]

    counts: dict[str, int] = defaultdict(int)
    for s_ in silver:
        for t in s_:
            counts[t] += 1
    if args.vocab == "genre":
        all_tags = sorted(set(GENRE_FLAG_TO_TAG.values()))
    else:
        all_tags = sorted((t for t, n in counts.items() if n >= args.vocab_min_works),
                          key=lambda t: -counts[t])
        log(f"語彙: {len(counts)}種のうち {args.vocab_min_works}作品以上に付いた "
            f"{len(all_tags)}種を対象にします")
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else all_tags
    corpus = [texts_by_ncode[w["ncode"]] for w in works]

    if args.grid:
        if not args.tags:
            raise SystemExit("--grid には --tags で対象タグを指定してください")
        vecs = []
        for spec in args.grid_vecs.split(","):
            a, b, mf = (int(v) for v in spec.strip().split("-"))
            vecs.append(((a, b), mf))
        cs = [float(v) for v in args.grid_cs.split(",")]
        log(f"grid: ベクトル{len(vecs)}種 × C{len(cs)}種 × タグ{len(tags)}個 "
            f"= {len(vecs)*len(cs)*len(tags)}回のCV  chars_per_work={args.chars_per_work}")
        grid_rows = []
        for ngram, mf in vecs:
            if deadline and datetime.datetime.now() >= deadline:
                log("締切に到達。gridを打ち切ります")
                break
            Xg = vectorize(corpus, args.cache, ngram, mf, args.min_df,
                           args.chars_per_work)
            for C in cs:
                for tag in tags:
                    y = np.array([1 if tag in s else 0 for s in silver])
                    if y.sum() < args.min_pos:
                        continue
                    t0 = time.time()
                    r = eval_tag(Xg, y, args.folds, args.seed, C,
                                 calibrate=not args.no_calibrate)
                    row = {"tag": tag, "ngram": f"{ngram[0]}-{ngram[1]}", "max_features": mf,
                           "C": C, "chars_per_work": args.chars_per_work,
                           "auc": r["auc"], "ap": r["ap"], "best_f1": r["best_f1"],
                           "seconds": round(time.time() - t0, 1)}
                    grid_rows.append(row)
                    log(f"  {tag:<8} char{ngram[0]}-{ngram[1]} mf={mf:<7} C={C:<4} "
                        f"AUC={r['auc']:.3f} AP={r['ap']:.3f} F1={r['best_f1']:.3f} "
                        f"[{row['seconds']}s]")
        grid_path = os.path.join(args.out, "grid.jsonl")
        with open(grid_path, "a", encoding="utf-8") as fh:
            for row in grid_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("\n===== grid結果 (タグ別・AUC降順) =====")
        for tag in tags:
            sub = sorted([g for g in grid_rows if g["tag"] == tag], key=lambda x: -x["auc"])
            if not sub:
                continue
            print(f"\n[{tag}]  最良 AUC={sub[0]['auc']:.3f} "
                  f"(char{sub[0]['ngram']} mf={sub[0]['max_features']} C={sub[0]['C']} "
                  f"chars={sub[0]['chars_per_work']})")
            for g in sub:
                print(f"    char{g['ngram']:<5} mf={g['max_features']:<7} C={g['C']:<5} "
                      f"AUC={g['auc']:.3f} AP={g['ap']:.3f} F1={g['best_f1']:.3f}")
        log(f"grid完了 -> {grid_path}")
        return

    X = vectorize(corpus, args.cache, (args.ngram_min, args.ngram_max),
                  args.max_features, args.min_df, args.chars_per_work)

    log(f"段階3: {len(tags)}タグを評価")
    rows = []
    for tag in tags:
        if tag in done:
            continue
        if deadline and datetime.datetime.now() >= deadline:
            log(f"締切に到達。{tag} 以降は未処理のまま正常終了します（--resume で続行可）")
            break
        y = np.array([1 if tag in s else 0 for s in silver])
        if y.sum() < args.min_pos:
            log(f"  {tag}: 正例{int(y.sum())}件で不足のためスキップ")
            continue
        t0 = time.time()
        res = eval_tag(X, y, args.folds, args.seed, args.C, calibrate=not args.no_calibrate)
        cand_idx = res.pop("_candidates")[: args.top_candidates]
        corrected = res.pop("_corrected")
        oof_cal = res.pop("_oof_cal")
        res.pop("_oof")
        res["tag"] = tag
        res["seconds"] = round(time.time() - t0, 1)
        res["candidates"] = [
            {"ncode": works[i]["ncode"], "title": works[i].get("title"),
             # score: PU補正後(clipで1.0に飽和しうる)。score_cal: 順位の根拠の較正済み確率
             "score": round(float(corrected[i]), 4),
             "score_cal": round(float(oof_cal[i]), 4)}
            for i in cand_idx
        ]
        with open(results_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        rows.append(res)
        log(f"  {tag:<26} pos={res['n_pos']:>4} AUC={res['auc']:.3f} AP={res['ap']:.3f} "
            f"F1={res['best_f1']:.3f} (P={res['precision']:.3f} R={res['recall']:.3f}) "
            f"PU推定有病率={res['pu_estimated_prevalence']:.3f} vs 申告={res['silver_prevalence']:.3f} "
            f"[{res['seconds']}s]")

    if rows:
        print("\n===== まとめ (AUC降順) =====")
        print(f"{'tag':<26}{'pos':>5}{'AUC':>8}{'AP':>8}{'F1':>8}{'prec':>8}{'recall':>8}"
              f"{'申告率':>9}{'PU推定':>9}{'較正':>6}")
        for r in sorted(rows, key=lambda x: -x["auc"]):
            print(f"{r['tag']:<26}{r['n_pos']:>5}{r['auc']:>8.3f}{r['ap']:>8.3f}{r['best_f1']:>8.3f}"
                  f"{r['precision']:>8.3f}{r['recall']:>8.3f}"
                  f"{r['silver_prevalence']:>9.3f}{r['pu_estimated_prevalence']:>9.3f}"
                  f"{'ok' if r.get('calibration_ok') else 'NG':>6}")
        print("\nAUC 0.5=無情報 / 0.7=弱い / 0.8以上=実用圏")
        print("PU推定 > 申告率 なら、作者の付け忘れがその差分だけ存在すると推定される")
        print("較正=NG のタグはPU推定が信用できない(AUC・順位・候補リストは有効)")
    log(f"完了。結果: {results_path}")


if __name__ == "__main__":
    main()
