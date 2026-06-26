# FlexMM

[English](../README.md) | [简体中文](README_zh-CN.md) | **日本語**

**FlexMM** は、系列構造を考慮したマルチモーダルデータ準備と再現可能な実験オーケストレーションのための、モデル非依存 Python フレームワークです。

順序付きサンプル辞書リストから、次の処理までを一貫した形で扱います。

- 整列済みのマルチモーダル入力・目的変数配列
- 設定可能な時間コンテキストウィンドウ
- 分類・回帰ターゲットのメタ情報
- グループ独立、グループ依存、またはグループ制約なしの train/validation/test 分割
- 入力モダリティ組合せ × fold の実験
- 生の辞書、PyTorch Dataset、または PyTorch DataLoader
- 各実行のコンテキスト、正規化統計量、評価指標、結果保存

FlexMM は、モデルアーキテクチャや学習ループをあえて規定しません。モデル構築、最適化、early stopping、ログ記録はユーザーが制御し、FlexMM はその周辺にある反復的でミスが起きやすい実験準備を担当します。

---

## FlexMM を使う理由

マルチモーダル実験では、次のような基盤処理を何度も実装することになります。

1. モダリティとターゲットの整列
2. 時系列ウィンドウの構築
3. 話者、参加者、グループ、セッション単位の分割
4. 分割間の系列リーク防止
5. すべてのモダリティ組合せと fold に対する実験反復
6. fold 固有情報の学習・評価処理への受け渡し
7. 再現に必要な情報の保存

FlexMM はこれらを、責務の明確な 3 モジュールにまとめます。

| モジュール | 役割 |
|---|---|
| `flexmm.info_utils` | 軽量なサンプル情報、turn、隣接関係、区間の照会と整理。 |
| `flexmm.data_prep` | 整列済み系列データの構築、ターゲット処理、データ分割、準備結果の保存。 |
| `flexmm.experiment` | `ExperimentManager`、`ExperimentUnit`、`RunContext` による入力組合せ・fold の反復。 |

---

## 主な機能

- **1 サンプル 1 辞書の順序付きデータ契約**：`data_dicts` の各要素は、サンプル、フレーム、発話、イベントのいずれかを表します。
- **マルチモーダル入力**：任意数の入力キーを個別に設定できます。
- **系列を考慮したデータ準備**：過去/未来コンテキスト、stride、offset、端点除外、constant/edge padding を利用できます。
- **分類・回帰**：ラベル写像、ターゲット統計、回帰ビン、多次元回帰ターゲットに対応します。
- **3 種類の分割意味論**：
  - 話者/グループ/セッション独立
  - 各 reference group 内での依存分割
  - reference group を考慮しないサンプル単位分割
- **Holdout、K-fold、Leave-one-out** のテスト方式
- **任意の層化**：スカラー分類ターゲットとスカラー回帰ターゲットに対応
- **系列重複の防止**：test/train、および必要に応じて train/validation 間のウィンドウ重複を除去
- **入力モダリティ組合せ実験**：全組合せの自動生成または明示指定
- **リークを避けた標準化**：各 fold の学習データで統計量を推定し、validation/test に再利用
- **任意の PyTorch 連携**：データ準備段階では PyTorch を必須にしません
- **再反復可能な実験管理**：一度しか使えない generator ではなく、同じ manager を複数回反復可能
- **再現可能な成果物**：データ準備設定、実験設定、分割情報、ターゲット写像、インデックス写像を保存

---

## インストール

FlexMM は Python 3.9 以降を必要とします。

リポジトリのルートで次を実行します。

```bash
pip install -e .
```

コア依存関係：

```bash
pip install numpy scipy scikit-learn
```

PyTorch は任意です。`data_level="dataset"`、`data_level="dataloader"`、または明示的な tensor 変換を利用する場合のみ必要です。

```bash
pip install torch
```

> 公開前に `pyproject.toml` などへ依存関係を記述し、通常は `pip install -e .` または `pip install flexmm` だけで導入できるようにしてください。

---

## データモデル

FlexMM は、**順序付き辞書リスト**を入力として受け取ります。リスト上の位置は、整列、系列構築、分割記録で用いる元サンプルインデックスです。

```python
import numpy as np

sample_infos = [
    {
        "sample_id": "S0001",
        "speaker": "P01",
        "session": "session_1",
        "audio": np.random.randn(32).astype(np.float32),
        "text": np.random.randn(64).astype(np.float32),
        "label": "positive",
    },
    {
        "sample_id": "S0002",
        "speaker": "P01",
        "session": "session_1",
        "audio": np.random.randn(32).astype(np.float32),
        "text": np.random.randn(64).astype(np.float32),
        "label": "negative",
    },
]
```

フレームワーク内部の整列は元のリストインデックスに基づきますが、監査や外部参照のために安定した `sample_id` を付与することを強く推奨します。

辞書に含めることが多いフィールド：

- 識別子：`sample_id`、`speaker`、`participant`、`group`、`session`
- ターゲット：`label`、`score`、`valence`、`arousal`
- モデル入力：`audio`、`video`、`text`、センサーベクトル、手設計特徴量
- 補足情報：タイムスタンプ、条件、元ファイルパス、trial ID

設定対象の数値は、サンプル間で互換性のある shape を持つ必要があります。規則的な数値データは可能な限り NumPy 配列として収集され、異種データや非数値データは Python リストのまま保持されます。

> 現在のデータ準備 API では、設定した値を準備時に `data_dicts` から参照できる必要があります。メモリに収まらない大規模データでは、`data_dicts` に軽量な参照を保存してプロジェクト固有のローダーを追加するか、feature-store 抽象を拡張してください。

---

## クイックスタート

### 1. データ準備を設定して実行する

```python
from flexmm.data_prep import (
    ClassificationTargetConfig,
    DataPrepConfig,
    DataPreparator,
    InputConfig,
)

prep_config = DataPrepConfig(
    focused_target_key="label",
    split_ref_key="speaker",
    split_dependency="independent",
    independent_split_valid_by="ref_key",
    split_mode="kfold",
    folds=5,
    train_valid_ratio=0.8,
    use_stratified_split=False,
    remove_test_train_overlap_range=True,
    data_configs=[
        InputConfig(
            keys=["audio", "text"],
            seq_len_before=2,
            seq_len_after=2,
            stride=1,
            seq_padding=True,
            seq_padding_mode="edge",
            standardize_data=True,
            standardize_scope="split",
            dtype="float32",
        ),
        ClassificationTargetConfig(
            keys="label",
            convert_target_to_id=True,
        ),
    ],
    save_prepared_data=True,
    overwrite_data=True,
    store_dir="./ExperimentStore/demo",
)

preparator = DataPreparator(sample_infos, prep_config)
collected_data, info_dict = preparator.run()
```

整列済みデータとメタ情報が生成されます。保存を有効にした場合、出力ディレクトリは次のようになります。

```text
ExperimentStore/demo/
├── Data.pkl
├── Info.pkl
└── DataPrepConfig.json
```

### 2. 実験マネージャーを作成する

```python
from flexmm.experiment import ExperimentManager, TorchExperimentConfig

exp_config = TorchExperimentConfig(
    experiment_input_keys=["audio", "text"],
    experiment_target_keys="label",
    generate_input_comb=True,
    input_key_abbr={"audio": "A", "text": "T"},
    data_level="dataloader",
    data_representation="pt",
    load_prepared_data=True,
    store_dir="./ExperimentStore/demo",
    random_seed=42,
    train_batch_size=32,
    valid_batch_size=64,
    test_batch_size=64,
)

manager = ExperimentManager(exp_config).setup()
```

入力キーが 2 つで `generate_input_comb=True` の場合、各 fold に対して次を生成します。

```text
[audio]
[text]
[audio, text]
```

### 3. 独自の学習ループを実行する

```python
import torch

for unit in manager:
    train_loader = unit.data["train"]
    valid_loader = unit.data["valid"]
    test_loader = unit.data["test"]
    context = unit.context

    print(context.input_comb)
    print(context.fold)
    print(context.output_dir)

    model = build_model(
        input_keys=context.input_comb,
        target_keys=context.target_keys,
        info_dict=context.info_dict,
    )

    result = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        context=context,
    )

    manager.save_result(result, context=context)
```

batch は学習ループ内でモデルの device に移動してください。

```python
batch = {
    key: value.to(device) if isinstance(value, torch.Tensor) else value
    for key, value in batch.items()
}
```

DataLoader 作成時、manager は dataset tensor を CPU 上に保持します。これは GPU メモリ使用量やマルチプロセス読み込みの点で安全です。

---

## データ準備を別途呼び出さない一体型モード

manager から内部的にデータ準備を実行できます。

```python
exp_config = TorchExperimentConfig(
    experiment_input_keys=["audio", "text"],
    experiment_target_keys="label",
    data_level="dataloader",
    data_representation="pt",
    load_prepared_data=False,
    store_dir="./ExperimentStore/demo",
)

manager = ExperimentManager(
    exp_config=exp_config,
    data_dicts=sample_infos,
    data_prep_config=prep_config,
    user_extras={"project": "demo"},
).setup()

for unit in manager:
    run_experiment(unit)
```

同じ準備済みデータを複数モデルやハイパーパラメータ実験で再利用する場合は 2 段階方式を推奨します。短いスクリプトや一度限りの実験では一体型モードが便利です。

---

## 実験オブジェクト

### `ExperimentManager`

共有実験状態を管理します。データの読み込みまたは準備、乱数シードの初期化、必要キーの検証、`ExpConfig.json` の保存を行い、すべての「入力組合せ × fold」条件に対する新しい iterator を生成します。

### `ExperimentUnit`

1 つの実行可能な条件を表します。

```python
unit.data       # {"train": ..., "valid": ..., "test": ...}
unit.context    # RunContext
unit.input_comb # 互換用プロパティ
unit.fold       # 互換用プロパティ
```

### `RunContext`

モデル構築、学習、評価、結果保存へ実行固有情報を渡します。

```python
context.fold
context.comb_index
context.comb_name
context.input_comb
context.target_keys
context.split_indexes
context.prepared_split_indexes
context.ref_value_splits
context.standardization_info
context.info_dict
context.exp_config
context.data_prep_config
context.output_dir
context.seed
context.user_extras
```

`split_indexes` は元の `data_dicts` インデックス、`prepared_split_indexes` は系列フィルタリング・準備後の `collected_data` 上の位置です。

---

## 分割方式

### Independent split

```python
split_dependency="independent"
```

`split_ref_key` で識別される reference value は、test と非 test の間で完全に分離されます。話者独立、参加者独立、グループ独立、セッション独立評価に適しています。

validation の選択方式：

```python
independent_split_valid_by="ref_key"  # validation group も train から独立
independent_split_valid_by="index"    # validation sample は train と同じ group を含み得る
```

### Dependent split

```python
split_dependency="dependent"
independent_split_valid_by=None
```

各 reference group から train、validation、test にサンプルを割り当てます。同じ参加者やグループがすべての集合に存在してよい評価でのみ利用してください。

### Unconstrained split

```python
split_dependency="none"
independent_split_valid_by=None
```

reference group を無視し、対象サンプルインデックスのみで分割します。

### テストモード

すべての分割方式が次に対応します。

```python
split_mode="holdout"
split_mode="kfold"
split_mode="leave_one_out"
```

分割は決定的で、入力/reference の順序を保持します。FlexMM は分割前に暗黙の shuffle を行いません。ランダム分割が必要な場合は、データ順序を意図的に整える、明示的な split override を使う、または準備前に再現可能な並べ替えを行ってください。

---

## 系列構築

各入力・ターゲット設定で次を指定できます。

```python
seq_len_before
seq_len_after
stride
step_offset
seq_pos_from_start
seq_pos_from_end
seq_padding
seq_padding_mode  # "constant" or "edge"
seq_padding_value
```

例：

```python
InputConfig(
    keys="audio",
    seq_len_before=2,
    seq_len_after=1,
    stride=1,
    seq_padding=True,
)
```

概念的には次のウィンドウを構築します。

```text
[t-2, t-1, t, t+1]
```

系列境界は `DataPrepConfig` で制御します。

```python
seq_group_mode="ref_key"  # seq_group_key/split_ref_key の連続区間で分割
seq_group_key="speaker"
```

または：

```python
seq_group_mode="index"    # 順序付きデータ全体を 1 つの範囲として扱う
```

半開区間によるカスタム範囲も利用できます。

```python
seq_ranges_custom=[(0, 100), (150, 220)]
```

異なる split 間で系列ウィンドウが重なる場合、優先度の低い anchor を除去できます。

```python
remove_test_train_overlap_range=True
remove_train_valid_overlap_range=False
remove_overlap_priority=["test", "train", "valid"]
```

既定では test、train、validation の順に優先して保持します。

---

## ターゲット

### 分類

```python
ClassificationTargetConfig(
    keys="label",
    convert_target_to_id=True,
)
```

準備時に次を記録します。

```python
info_dict["target_info"]["label"]["target2id"]
info_dict["target_info"]["label"]["id2target"]
info_dict["target_info"]["label"]["target_stats"]
info_dict["target_info"]["label"]["target2indexes"]
```

分類ターゲットはスカラー、またはスカラー相当の値である必要があります。

### 回帰

```python
from flexmm.data_prep import RegressionTargetConfig

RegressionTargetConfig(
    keys="score",
    stratified_bin_num=10,
    convert_target_to_bin=False,
)
```

スカラー回帰ターゲットは、層化分割のためにビン化できます。多次元回帰ターゲットは次のように設定します。

```python
RegressionTargetConfig(
    keys="trajectory",
    is_multi_dim=True,
)
```

多次元ターゲットは層化分割には使用できません。

---

## 標準化とデータリーク

入力設定で標準化を有効にします。

```python
InputConfig(
    keys="audio",
    standardize_data=True,
    standardize_scope="split",
)
```

- `standardize_scope="split"`：現在の fold の**学習データ**で平均・標準偏差を推定し、同じ統計量を train、validation、test に適用します。通常の評価ではこちらを推奨します。
- `standardize_scope="all"`：すべての準備済みサンプルで統計量を推定します。validation/test 情報を意図的に利用するため、一般的な評価ではリークとなり得ます。

実行ごとの統計量は次から参照できます。

```python
unit.context.standardization_info
```

現在の実験処理で実装されているのは z-score 標準化です。`standardize_method` は将来拡張用であり、現時点では `minmax` は適用されません。

---

## データ出力レベル

`ExperimentConfig.data_level` は各 split のオブジェクト形式を指定します。

| 値 | Split オブジェクト |
|---|---|
| `"raw"` | 配列/リストを含む辞書 |
| `"dataset"` | `TorchDataset` |
| `"dataloader"` | PyTorch `DataLoader` |

`data_representation` は Dataset 内の値の形式を指定します。

| 値 | 動作 |
|---|---|
| `"original"` | NumPy 配列、リスト、tensor を元の形式で保持 |
| `"pt"` | 対応する数値データを PyTorch tensor に変換 |

---

## 評価指標

組み込み helper：

### 分類

```python
from flexmm.experiment import compute_cls_metrics

metrics = compute_cls_metrics(predictions, targets)
```

戻り値：

- accuracy
- macro F1
- weighted F1
- macro precision
- macro recall
- 定義可能な場合の Pearson correlation
- confusion matrix
- prediction/target 配列

### 回帰

```python
from flexmm.experiment import compute_regression_metrics

metrics = compute_regression_metrics(predictions, targets)
```

戻り値：

- MAE
- MSE
- RMSE
- 定義可能な場合の Pearson correlation
- prediction/target 配列

`ExperimentManager.get_result()` は NumPy 配列と PyTorch tensor の両方を受け取り、分類予測が score/logit 配列の場合は自動的に `argmax` を適用します。

---

## 情報ユーティリティ

`flexmm.info_utils` はデータ準備前に `data_dicts` を軽量に分析できます。

```python
from flexmm import info_utils

speaker_to_indexes = info_utils.get_ref_value2indexes(
    sample_infos,
    ref_key="speaker",
)

turns = info_utils.get_turn2ref_value_and_indexes(
    sample_infos,
    ref_key="speaker",
)

speaker_to_labels = info_utils.get_ref_value2another(
    sample_infos,
    ref_key="speaker",
    another_key="label",
    unique_values=True,
)
```

turn index、隣接 reference value、区間ベースのインデックス分割も扱えます。

---

## 保存される成果物

典型的な実験ディレクトリ：

```text
ExperimentStore/demo/
├── Data.pkl
├── Info.pkl
├── DataPrepConfig.json
├── ExpConfig.json
├── Comb_A/
│   ├── fold_0/
│   │   └── ExpResult.pkl
│   └── fold_1/
│       └── ExpResult.pkl
├── Comb_T/
│   └── ...
└── Comb_A-T/
    └── ...
```

`Data.pkl` と `Info.pkl` は Python pickle を利用します。信頼できない提供元の pickle ファイルは読み込まないでください。

---

## プロジェクト構成

最小構成：

```text
flexmm/
├── __init__.py
├── info_utils.py
├── data_prep.py
└── experiment.py
docs/
└── WORKFLOW.md
README.md
pyproject.toml
LICENSE
```

---

## 現在の対象範囲

FlexMM は、現在は実験準備とオーケストレーションに重点を置いています。

- モデル定義・最適化はユーザーが実装します
- 設定した特徴値はデータ準備時に参照可能である必要があります
- 分類指標は単一ラベル分類を対象とします
- 層化にはスカラーターゲットが必要です
- 分割は入力順を保持し、自動 shuffle は行いません
- 現在実装されている標準化は z-score のみです
- PyTorch 連携は任意で、DataLoader の全データを自動的に GPU へ移しません

これらは隠れた挙動ではなく、明示的な拡張ポイントです。

---

## 詳細ドキュメント

[`docs/WORKFLOW_ja.md`](docs/WORKFLOW_ja.md) には次の内容を記載しています。

- データ準備から実験までの完全なライフサイクル
- インデックス空間の変換
- 詳細な分割意味論
- 系列ウィンドウの挙動
- ターゲット処理
- 標準化ルール
- 拡張ポイントとよくある落とし穴

---

## コントリビューション

コントリビューションを歓迎します。pull request を作成する前に：

1. 挙動変更に対するテストを追加または更新してください
2. 公開 docstring と実装を同期してください
3. validation/test リークを導入しないでください
4. 互換性やシリアライズ形式の変更を文書化してください
5. 明示的かつ seed 固定の乱数を導入しない限り、決定的な挙動を維持してください

---

## 引用

***
