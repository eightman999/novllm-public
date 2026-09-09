# novllm

手持ちのSQLite小説データベースから日本語小説コーパスを構築し、LoRAで小説生成モデルを学習・評価するための研究用ツール群です。任意で、ShortGPTのBlock Influence指標を用いたベースモデルのレイヤー剪定（depth pruning）にも対応します。

> [!IMPORTANT]
> このリポジトリには、小説本文、データベース、モデル重み、LoRAアダプター、チェックポイントは含まれません。利用するデータとモデルの権利・利用条件は、利用者が個別に確認してください。

## 主な機能

- SQLite小説データベースからJSONLコーパスを構築
- 本文の正規化、チャンク化、メタデータ整形、品質監査
- Qwen系モデルを中心としたLoRA / QLoRA学習
- ShortGPT方式のレイヤー重要度測定と任意の剪定
- コンテキスト条件、固有名詞、プロット追従、生成品質の評価
- LM Studio / Ollamaを使った要約生成と複数マシンへの分割実行
- CUDA、Apple Silicon（MPS）、CPUの実行環境を自動選択

## 構成

```text
novllm/
├── build_dataset.py          # SQLite DBから学習用データセットを構築
├── export_novel_jsonl.py     # シンプルな本文JSONLエクスポート
├── prune_model.py            # レイヤー重要度の測定とモデル剪定
├── train_lora.py             # LoRA / QLoRA学習
├── generate.py               # ベースモデルまたはLoRAから文章生成
├── pipeline/                 # 前処理パイプラインとテスト
├── tools/                    # データ構築、要約生成、評価ツール
├── schemas/                  # 構造化データのJSON Schema
└── verify.sh                 # 構文・テスト検証ハーネス
```

## 必要環境

- Python 3.12推奨
- CUDA対応GPU、Apple Silicon、またはCPU
- ベースモデルと学習データを保存できる十分な空き容量

CUDA環境ではQLoRA（4bit量子化）を利用できます。Apple Siliconでは`bitsandbytes`が使えないため、通常のLoRA（fp16）へフォールバックします。デバイスは既定でCUDA、MPS、CPUの順に選択されます。

## セットアップ

```bash
git clone https://github.com/eightman999/novllm-public.git
cd novllm-public
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 基本的な使い方

### 1. データセットを構築する

現在の前処理パイプラインを使う場合:

```bash
python build_dataset.py \
  --db /path/to/novels.db \
  --out-dir ./dataset_v2.tmp
```

本文だけをシンプルなJSONLとして取り出す場合は、`export_novel_jsonl.py`内の`SRC_DB`を設定して実行します。

```bash
python export_novel_jsonl.py
```

生成データ、DB、モデル成果物は`.gitignore`により追跡対象外です。

### 2. ベースモデルを剪定する（任意）

まず重要度だけを確認します。

```bash
python prune_model.py \
  --base-model Qwen/Qwen3-8B-Base \
  --num-layers-to-prune 4 \
  --dry-run
```

確認後に剪定モデルを保存します。

```bash
python prune_model.py \
  --base-model Qwen/Qwen3-8B-Base \
  --num-layers-to-prune 4 \
  --output-dir ./pruned_out-qwen3-8b
```

剪定直後は表現がずれるため、単体での利用ではなく、`train_lora.py`で追加学習（healing）してから使うことを想定しています。

### 3. LoRAを学習する

```bash
python train_lora.py \
  --base-model ./pruned_out-qwen3-8b \
  --data-glob './data/*.jsonl' \
  --output-dir ./lora_out-novel
```

`--base-model`を省略した場合は`Qwen/Qwen3-8B-Base`を使用します。環境変数`NOVLLM_BASE_MODEL`でも上書きできます。

### 4. 文章を生成する

```bash
python generate.py \
  --base-model ./pruned_out-qwen3-8b \
  --adapter ./lora_out-novel/adapter \
  --prompt '森の奥に光る扉があった。'
```

## 検証

高速検証ではPythonの構文チェックと、外部DB・tokenizerに依存しないテストを実行します。

```bash
./verify.sh
```

追加の検証モード:

```bash
./verify.sh --check  # Python・依存パッケージの環境確認
./verify.sh --full   # pipeline/tests全体を実行
```

`--full`にはHugging Face tokenizerキャッシュを必要とするテストがあります。依存物がない場合、一部テストは失敗します。

## 複数マシンでの要約生成

`tools/build_summaries.py`は`--shard i/n`で決定的にジョブを分割できます。出力ファイルはマシンまたはshardごとに必ず分けてください。同じJSONLへの並行追記には対応していません。

```bash
python tools/build_summaries.py \
  --chunks '/path/to/chunks/*' \
  --targets /path/to/targets.jsonl \
  --out ./summaries/worker_0of2.jsonl \
  --shard 0/2
```

各ツールのデータパス、モデル名、推論サーバーの接続先はコマンドライン引数または`NOVLLM_*`環境変数で指定してください。公開版には特定マシン向けのSSH、WOL、NAS運用設定を含めていません。

## 公開範囲とデータ

この公開リポジトリは、履歴を含めてソースコードと文書だけで構成しています。次のものは収録していません。

- 小説本文、作品名、作者名、作品IDを含むコーパス
- SQLiteデータベース、JSONL、評価時の本文プレビュー
- モデル重み、LoRAアダプター、チェックポイント
- 生成結果、実験ログ、ローカルキャッシュ
- 個人環境のIPアドレス、MACアドレス、SSH alias、認証情報

`.gitignore`は主要なデータ・モデル形式を既定で除外します。第三者データを使う場合は、そのデータをGitへ追加しないでください。

## 注意事項

- `prune_model.py`はQwen系の`model.model.layers`構造を前提とします。他のモデルでは互換性を確認してください。
- 学習データには性描写、暴力表現、戦記表現などが含まれる可能性があります。
- 学習・生成には大きな計算資源とストレージを使用します。短いスモークテストから始めてください。

## ライセンス

このリポジトリ内のコードと文書は[MIT License](LICENSE)で提供します。ただし、学習データ、入力データベース、ベースモデル、tokenizer、学習済み重み、生成物、その他の第三者素材には、それぞれの権利者が定める別のライセンスや利用条件が適用されます。

## NovTokenizer: published measurements

| Lineage | Protocol | Status |
|---|---|---|
| J32 | 150M total parameters, same 30M source characters, two seeds | completed reference |
| J48 | same source/total as J32 | leading candidate; cultural regressions remain |
| J64 | same protocol, representative seed 1 only | exploratory; running |

Tokenizer freeze is not decided. Phase 6 has not started.

- [Phase 5 tokenizer-only observations](results/phase5/README.md): 18 candidates and 324 category measurements.
- [Phase 5.5 measured tables](results/phase55/README.md): final/domain/checkpoint BPB, compression, timing, memory, model size and provenance.
- [Hardened kanbun/kakikudashi evaluation](results/phase551/audit/README.md): 457 works per category, text-match audit, paired work-bootstrap intervals, and public-source retrieval hashes.
- [Fixed runtime and input requirements](research/phase55_runtime/README.md).
- [Native macOS live monitor](tools/novtokenizer-monitor/README.md): local read-only SSH refresh; no AI calls.

Published results exclude corpus text, checkpoint binaries, private service addresses and original private identifiers. Work-level NLL/counts permit independent BPB/bootstrap reanalysis without those inputs.
