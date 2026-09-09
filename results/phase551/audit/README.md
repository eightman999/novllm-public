# Phase 5.5.1 漢文・書き下し評価の監査

## 条件と出典

既存Phase 5.5の行分割を維持する。漢文原文 (`hakubun`) と書き下し (`kakikudashi`) を分離し、原文行の連結、表記変更、訓読文への自動変換は行わない。BOSを各行に置く既存の `phase55_same_source_runtime.evaluate()` で、保存済みのJ32/J48 final checkpointを再評価する。新しい学習は行わない。

[Kanbun-LM公式リポジトリ](https://github.com/nlp-waseda/Kanbun-LM) は `poetry_id` を詩の識別子と説明し、詩がsplitを跨がない分割を採用している。既存690行×2カテゴリの出典レコードを全文一致でjoinし、欠けていた作品IDを復元した。旧評価の漢文は4,292字、書き下しは7,344字、各93作品。

増量候補は既存canonicalに保存済みの全3,421対・465詩。upstreamのtrain/val/testラベルはそのまま残す。upstreamのtrainはNovTokenizerのLM学習・tokenizer学習とは別の集合であり、実際に使った30M字LM inputとJ tokenizerの100M字入力を別々に監査して選別する。

2026-09-09に公式リポジトリのREADMEと固定commitの完全recursive tree（LICENSE/COPYING候補なし）を確認したが、本文再配布の明示的許諾を確認できなかった。既存registryの `redistribution=not_allowed` を維持し、本文はGitへ入れない。公開対象はID、SHA-256、統計、機械可読監査、実測NLL/BPBのみ。詩名・作者名は今回のcanonicalにないため推測しない。異なるcorpusとの作品名照合は `unknown` のまま残す。

## 固定した監査方法

実測前に閾値を固定した。

- 原文SHA-256によるdocument全体exact match。
- NFKC後にUnicode whitespaceを除いた監査専用viewのexact match。評価本文自体は変更しない。
- 正規化viewで50文字以上の連続substring一致。50文字未満の評価行は `null`。
- 5文字以上の評価行がtrain recordに含まれる短文substringも別途保存。短文評価で長文thresholdだけを使った見逃しを避ける。
- 文字5-gramの方向付きcontainment: `|eval_grams ∩ train_grams| / |eval_grams| >= 0.8`。全train recordを走査し、最大値と一致record IDを保存。
- 5文字未満ではnear/substringは `null`。低い検出力を「汚染なし」に置き換えない。
- 評価集合内部も5-gram overlap coefficient 0.8でnear duplicate pairを列挙する。

containmentは引用を含む長いtrain recordを検出できるが、離れた位置の共通句でも陽性になり得る。意味的な言い換え、翻訳、異体字の網羅は保証しない。作品IDのnamespaceが異なるため、ID集合にないだけでは作品不一致と断定しない。

hardening集合では、LMまたはtokenizerに一行でも検出された詩を作品単位で除外する。その後、カテゴリ内のexact duplicate行を入力順の先頭一件に限定する。近似一致の閾値を結果を見て緩める処理はない。すべての除外はmanifestに記録する。

## 再現手順

取得は `run_phase551_audit.py --prepare-source` から既存 `scale_special.audit_kanbun()` を再利用する。source manifestのcommit `9f9edf0d17072d31ede9d7d9f0904967c9dbf69d` に固定した3個のCSVを取得し、それぞれSHA-256を照合する。全カラムを保持してcanonicalを再生成し、そのバイトhashまで照合する。2026-09-09の独立した新規ディレクトリへの取得で、3 CSVとcanonicalのhash一致を確認した。取得結果の本文はprivateのまま保持する。追加依存は `scale_special.py`、`cultural_store.py`、`cultural_registry.py`（標準ライブラリのみ）。

```bash
python research/phase55_runtime/scripts/run_phase551_audit.py \
  --prepare-source /path/to/public/source_manifest.json \
  --source-output /path/to/new/private/source

python research/phase55_runtime/scripts/run_phase551_audit.py \
  --dataset /path/to/frozen/phase55/dataset \
  --kanbun /path/to/canonical/kanbun_lm/raw-00000.jsonl \
  --tokenizer-input /path/to/recipes/J/train.escaped.txt \
  --output results/phase551_eval_private

CUDA_VISIBLE_DEVICES=1 python research/phase55_runtime/scripts/eval_phase551_checkpoints.py \
  --bundle /path/to/phase55-same-source-bundle \
  --eval-dir /path/to/phase551_eval_private \
  --output /path/to/new/phase551-evaluation-results

cd research/phase55_runtime && python -m pytest tests/test_phase551_audit.py -q
```

再評価CLIはP100を確認し、checkpoint sidecar hashとtokenizer hashを照合する。原文完全復元とunknown数を保存する。NLLを作品単位に合算し、同一作品をJ32/J48で対応させたpaired bootstrap 2,000回（seed=20260909）の95% percentile区間を出す。これは学習済みの一seedを条件とした作品samplingの不確実性であり、学習seed間の不確実性ではない。

## Limitation

今回の候補は漢詩に限られ、新たな散文文体や独立した訓読文カテゴリは加わらない。再配布可能な複数文体の評価本文を確保したという完了主張はしない。短文のBOS頻度が高い既存条件を維持しているため、長文漢文への一般化は未検証。作品間の文学的な引用関係と未知のidentityは残る。tokenizer freezeは行わない。

## Measured: 集合と汚染

| 集合 | カテゴリ | 行数 | 作品数 | 文字数 | UTF-8 bytes | 最大作品の文字比率 |
|---|---|---:|---:|---:|---:|---:|
| 旧 | 漢文 | 690 | 93 | 4,292 | 12,876 | 11.09% |
| 旧 | 書き下し | 690 | 93 | 7,344 | 22,032 | 10.12% |
| hardened | 漢文 | 3,384 | 457 | 20,492 | 61,476 | 3.00% |
| hardened | 書き下し | 3,384 | 457 | 35,069 | 105,207 | 2.76% |

旧漢文のうちLM30M inputに4行、J tokenizer100M inputに1行の短文substring一致を検出した。旧書き下しではこの検出は0行。whole-document exact/normalized exactは両カテゴリ0件だった。短文が中心なので50文字substring非検出だけをcleanの根拠にできない。

増量候補ではLM inputに漢文5行・書き下し2行、tokenizer inputに漢文3行・書き下し2行を検出した。重複する検出を作品単位でまとめて8作品を除外した。候補集合内near duplicateは185 pairを列挙し、原文exact duplicateの余剰行は各カテゴリ1件だった。作品名のcross-corpus identityは未確定のままで、残した集合も完全な非汚染証明ではない。

## Measured: checkpoint再評価

| 集合 | seed | 分野 | J32 BPB | J48 BPB | J48相対差 | 作品bootstrap 95% CI |
|---|---:|---|---:|---:|---:|---|
| 旧 | 1 | 漢文 | 5.245909 | 5.325431 | +1.516% | +1.328〜+1.757% |
| 旧 | 1 | 書き下し | 3.816686 | 3.845733 | +0.761% | +0.541〜+0.998% |
| 旧 | 2 | 漢文 | 5.258579 | 5.365920 | +2.041% | +1.865〜+2.229% |
| 旧 | 2 | 書き下し | 3.823654 | 3.855089 | +0.822% | +0.586〜+1.051% |
| hardened | 1 | 漢文 | 5.285414 | 5.366070 | +1.526% | +1.444〜+1.620% |
| hardened | 1 | 書き下し | 3.833037 | 3.863311 | +0.790% | +0.689〜+0.891% |
| hardened | 2 | 漢文 | 5.299373 | 5.406127 | +2.014% | +1.938〜+2.095% |
| hardened | 2 | 書き下し | 3.840712 | 3.873218 | +0.846% | +0.743〜+0.943% |

P100で4 checkpoint×2集合の8評価を完了。旧metricsとの最大absolute BPB差は6.42e-9未満で、seed2は完全一致した。全評価で原文完全復元率1.0、unknown token 0。checkpointの評価前後SHA-256はすべて一致し、評価runtime/model/tokenizer adapterのhashも固定Phase55 bundleに一致した。回収した18ファイルはremoteとのSHA-256一致を確認した。

機械可読結果は `checkpoint_evaluation/comparison.json`、各 `*-seed*-*.json` の作品単位NLL/BPBと `*-items.jsonl` の行単位NLL/BPBに保存する。これらは本文を含まない。公開先への配置・ID処理は公開exporterが担当する。

## Interpretation

この短行・漢詩評価条件では、J48の漢文・書き下し回帰は、評価作品増量と検出作品の除外後にも両seedで維持された。各seed条件付きの作品bootstrap区間も0を含まない。旧集合の少数作品への偏りだけで観測差が生じたという説明は支持されない。

この結論は長文漢文・散文・別の訓読体系に一般化しない。また、同一総パラメータ予算下のarchitecture差を含む比較であり、語彙サイズだけの因果効果とは断定しない。全体BPBでJ48が優位だった事実と、文化分野の回帰が残る事実を別々に保持する。

## Public reanalysis

The private eval/train/work IDs are consistently replaced with SHA-256 of their UTF-8 identifiers. `unknown` stays unknown. Public upstream record IDs are retained in `hardened_manifest.csv`. Large item audits and manifests use CSV without redundant JSON copies. Evaluation text is never published.

`work_scores.csv` retains per-work NLL/counts and the original sorted sampling order as `bootstrap_work_index`, so the paired work bootstrap can be replayed without text or private identities:

```bash
python3 scripts/verify_phase551_public.py --audit results/phase551/audit
```

Independent public-data-only verification reproduced all 16 domain totals and 8 bootstrap intervals. The 185 near-duplicate pairs refer to the candidate collection (including cross-category pairs); old cultural evaluation has 36 pairs. Residual similarities between different poems are not modeled by the within-poem bootstrap. CI is conditional on each fitted seed and this source collection.
