# FlexMM API リファレンス

[English](API_Reference.md) | [简体中文](API_Reference_zh-CN.md) | **日本語**

本ドキュメントは、FlexMM のソースコード内の docstring から生成されています。設定クラス、実験オブジェクト、データ準備用ヘルパー、評価関数、情報ユーティリティを確認するユーザー向けに、GitHub で読みやすい形式にまとめています。

- **パッケージ：** `flexmm`
- **リポジトリ：** https://github.com/Xia-code/flexmm
- **生成日：** 2026-06-26
- **Python：** 3.9 以降

> 公開クラス、メソッド、関数を先に掲載しています。
> アンダースコアで始まる名前は実装上の詳細であり、
> 折りたたみセクションに配置しています。これらの挙動はバージョン間で変更される可能性が比較的高くなります。

## 目次

1. [パッケージ概要](#パッケージ概要)
2. [`flexmm.data_prep`](#flexmmdata_prep)
3. [`flexmm.experiment`](#flexmmexperiment)
4. [`flexmm.info_utils`](#flexmminfo_utils)

## パッケージ概要

FlexMM は、マルチモーダルデータの準備と実験フローの管理に使用する3つの主要モジュールを提供します。

| モジュール | 主な役割 |
| --- | --- |
| `flexmm.data_prep` | シーケンス対応のマルチモーダルデータ収集、ターゲット処理、データ分割、重複防止、シリアライズ。 |
| `flexmm.experiment` | 実験設定、モダリティ組み合わせの反復、PyTorch 変換、指標計算、コンテキスト管理、結果読み込み。 |
| `flexmm.info_utils` | サンプル単位のメタデータ、ターン、参照値、隣接関係、区間に対する検索・グループ化操作。 |

### 一般的なインポート

```python
from flexmm import data_prep, experiment, info_utils
from flexmm.data_prep import DataPrepConfig, DataPreparator, InputConfig
from flexmm.experiment import ExperimentConfig, ExperimentManager
```

### パッケージ初期化ファイルの docstring

```text
__init__ file of flexmm
```

---

## `flexmm.data_prep`

[GitHub でソースを表示](https://github.com/Xia-code/flexmm/blob/main/flexmm/data_prep.py)

シーケンス対応のマルチモーダルデータ、ターゲット、および学習・検証・テスト分割を準備します。

本モジュールは、設定オブジェクト、データ収集ユーティリティ、ターゲットの
統計・変換ヘルパー、決定的な分割戦略、シーケンスの
重複除去、シリアライズ用ヘルパーを定義します。サンプル単位の記述データは、
元サンプルを識別するインデックスを持つ辞書のリストとして与えることを想定しています。

### API 概要

#### クラス

| クラス | 概要 |
| --- | --- |
| [`BaseConfig`](#baseconfig) | 1つの入力またはターゲットのデータキーグループに共通する設定フィールド。 |
| [`ClassificationTargetConfig`](#classificationtargetconfig) | 1つ以上のスカラー分類ターゲットを設定します。 |
| [`RegressionTargetConfig`](#regressiontargetconfig) | 1つ以上の回帰ターゲットと、オプションの層化用ビンを設定します。 |
| [`InputConfig`](#inputconfig) | 1つ以上のモデル入力データキーを設定します。 |
| [`DataPrepConfig`](#dataprepconfig) | シーケンス構築、ターゲット処理、データ分割、永続化を設定します。 |
| [`DataPreparator`](#datapreparator) | サンプル単位の辞書に対して、データ準備パイプライン全体を実行します。 |

#### 公開関数

| 関数 | 概要 |
| --- | --- |
| `pick_data_by_indexes()` | 元インデックスに基づいてサンプル辞書を選択し、各元インデックスを記録します。 |
| `gather_data_single_key()` | 1つのデータキーに対するすべてのシーケンスウィンドウを収集します。 |
| `gather_data_by_indexes()` | 定数パディングまたはエッジパディングを適用して、1つの値シーケンスを収集します。 |
| `shift_get_seq_indexes()` | フィルタ後のアンカーインデックスを中心に、ストライド付きコンテキストウィンドウを構築します。 |
| `get_strided_seq()` | インデックスリスト内の1つの位置を中心にストライド付きウィンドウを抽出します。 |
| `data_split_independent()` | 参照グループを独立に分割し、テストグループが学習または検証に現れないようにします。 |
| `data_split_dependent()` | 各参照グループ内でサンプルを分割するため、同じグループがすべてのデータセットに現れる場合があります。 |
| `data_split_unconstrained()` | 参照グループの独立性を強制せずに、サンプルインデックスを分割します。 |
| `remove_overlapped_seq_split()` | シーケンス内容が保護対象の分割と重複する、優先度の低いアンカーを削除します。 |
| `get_target_info_cls()` | クラス統計、クラス ID マッピング、元インデックスのグループを構築します。 |
| `get_target_info_regression()` | スカラー回帰ターゲットをビン分けし、ビン統計とインデックスを収集します。 |
| `get_target2indexes()` | 各クラスまたは回帰ビンを元のサンプルインデックスへ対応付けます。 |
| `get_stratified_bin_info()` | 実際の回帰ビン幅とビン数を計算します。 |
| `generate_config_template()` | 指定されたデータキー用の JSON 設定テンプレートを書き出します。 |
| `load_config()` | JSON 設定ファイルを設定オブジェクトとして読み込みます。 |
| `make_config_from_json()` | 解析済み JSON データからネストした設定オブジェクトを再構築します。 |
| `save_data()` | 明示的な引数または設定値を使って、準備済みデータとメタデータを保存します。 |
| `load_data()` | ディレクトリから準備済みデータ、メタデータ、設定を読み込みます。 |
| `get_target_list()` | 選択された元サンプルインデックスからターゲット値を収集します。 |
| `get_scalar_target_list()` | 選択されたターゲット値を収集し、Python スカラーへ正規化します。 |
| `convert_to_python_scalar()` | スカラー相当の Python、NumPy、または PyTorch 値を Python スカラーへ変換します。 |
| `calculate_miu_sigma()` | 選択されたサンプルについて、特徴量ごとの平均と標準偏差を計算します。 |

### クラス

### `BaseConfig`

```python
class BaseConfig
```

1つの入力またはターゲットのデータキーグループに共通する設定フィールド。

#### 注記

インスタンスには、検証済みの設定または本モジュールで使用するパイプライン状態が保存されます。

#### 宣言フィールド

| フィールド | 型 | デフォルト |
| --- | --- | --- |
| `keys` | `Union[List, Any]` | `_required_` |
| `seq_len_before` | `int` | `0` |
| `seq_len_after` | `int` | `0` |
| `step_offset` | `int` | `0` |
| `stride` | `int` | `1` |
| `seq_pos_from_start` | `int` | `0` |
| `seq_pos_from_end` | `int` | `0` |
| `seq_padding` | `bool` | `True` |
| `seq_padding_mode` | `Literal['edge', 'constant']` | `'constant'` |
| `seq_padding_value` | `Any` | `0` |
| `keep_batch_seq_dims` | `bool` | `True` |
| `squeeze_singleton_dims` | `bool` | `True` |
| `standardize_data` | `bool` | `False` |
| `standardize_method` | `Literal['zscore', 'minmax']` | `'zscore'` |
| `standardize_scope` | `Literal['all', 'split']` | `'split'` |
| `dtype` | `Union[None, str]` | `None` |

<details>
<summary><strong>内部メソッド（1）</strong></summary>

##### `BaseConfig.__post_init__`

```python
def BaseConfig.__post_init__(self)
```

派生フィールドを正規化し、初期化後に設定を検証します。

</details>

### `ClassificationTargetConfig`

```python
class ClassificationTargetConfig(BaseConfig)
```

1つ以上のスカラー分類ターゲットを設定します。

#### 注記

インスタンスには、検証済みの設定または本モジュールで使用するパイプライン状態が保存されます。

#### 宣言フィールド

| フィールド | 型 | デフォルト |
| --- | --- | --- |
| `convert_target_to_id` | `bool` | `False` |

<details>
<summary><strong>内部メソッド（1）</strong></summary>

##### `ClassificationTargetConfig.__post_init__`

```python
def ClassificationTargetConfig.__post_init__(self)
```

派生フィールドを正規化し、初期化後に設定を検証します。

</details>

### `RegressionTargetConfig`

```python
class RegressionTargetConfig(BaseConfig)
```

1つ以上の回帰ターゲットと、オプションの層化用ビンを設定します。

#### 注記

インスタンスには、検証済みの設定または本モジュールで使用するパイプライン状態が保存されます。

#### 宣言フィールド

| フィールド | 型 | デフォルト |
| --- | --- | --- |
| `is_multi_dim` | `bool` | `False` |
| `convert_target_to_bin` | `bool` | `False` |
| `stratified_bin_size` | `Union[float, int, None]` | `None` |
| `stratified_bin_num` | `Optional[int]` | `10` |
| `bin_closed_side` | `Literal['upper', 'lower']` | `'lower'` |

<details>
<summary><strong>内部メソッド（1）</strong></summary>

##### `RegressionTargetConfig.__post_init__`

```python
def RegressionTargetConfig.__post_init__(self)
```

派生フィールドを正規化し、初期化後に設定を検証します。

</details>

### `InputConfig`

```python
class InputConfig(BaseConfig)
```

1つ以上のモデル入力データキーを設定します。

#### 注記

インスタンスには、検証済みの設定または本モジュールで使用するパイプライン状態が保存されます。

#### 宣言フィールド

| フィールド | 型 | デフォルト |
| --- | --- | --- |
| `is_non_numeric` | `bool` | `False` |

<details>
<summary><strong>内部メソッド（1）</strong></summary>

##### `InputConfig.__post_init__`

```python
def InputConfig.__post_init__(self)
```

派生フィールドを正規化し、初期化後に設定を検証します。

</details>

### `DataPrepConfig`

```python
class DataPrepConfig
```

シーケンス構築、ターゲット処理、データ分割、永続化を設定します。

#### 注記

インスタンスには、検証済みの設定または本モジュールで使用するパイプライン状態が保存されます。

#### 宣言フィールド

| フィールド | 型 | デフォルト |
| --- | --- | --- |
| `focused_target_key` | `Union[Any, None]` | `None` |
| `split_ref_key` | `Union[Any, None]` | `'speaker'` |
| `split_dependency` | `Literal['independent', 'dependent', 'none']` | `'independent'` |
| `independent_split_valid_by` | `Union[None, Literal['index', 'ref_key']]` | `'index'` |
| `split_mode` | `Literal['holdout', 'kfold', 'leave_one_out']` | `'kfold'` |
| `folds` | `int` | `5` |
| `train_valid_ratio` | `float` | `0.9` |
| `holdout_test_ratio` | `float` | `0.2` |
| `use_stratified_split` | `bool` | `False` |
| `seq_group_mode` | `Literal['ref_key', 'index']` | `'ref_key'` |
| `seq_group_key` | `Union[Any, None]` | `None` |
| `seq_ranges_custom` | `Union[List[tuple[int, int]], None]` | `None` |
| `include_seq_inter_ranges` | `bool` | `False` |
| `seq_padding_index` | `int` | `-1` |
| `remove_test_train_overlap_range` | `bool` | `True` |
| `remove_train_valid_overlap_range` | `bool` | `False` |
| `remove_overlap_priority` | `Union[List, None]` | `field(default_factory=lambda: ['test', 'train', 'valid'])` |
| `data_configs` | `List` | `field(default_factory=list)` |
| `save_prepared_data` | `bool` | `True` |
| `overwrite_data` | `bool` | `True` |
| `store_dir` | `str` | `'./ExperimentStore'` |
| `index_split_dict_override` | `Union[None, Dict]` | `None` |
| `train_ref_values_override` | `Union[None, Dict]` | `None` |
| `valid_ref_values_override` | `Union[None, Dict]` | `None` |
| `test_ref_values_override` | `Union[None, Dict]` | `None` |

#### 公開メソッド

##### `DataPrepConfig.assert_config`

```python
def DataPrepConfig.assert_config(self)
```

データ分割関連の設定値を検証し、互換性のあるデフォルトを解決します。

##### `DataPrepConfig.to_json`

```python
def DataPrepConfig.to_json(self)
```

ネストした設定オブジェクトを JSON シリアライズ可能な構造へ変換します。

###### 戻り値

list
    クラス名と dataclass フィールドを含むシリアライズ済み設定エントリ。

<details>
<summary><strong>内部メソッド（3）</strong></summary>

##### `DataPrepConfig.__post_init__`

```python
def DataPrepConfig.__post_init__(self)
```

派生フィールドを正規化し、初期化後に設定を検証します。

##### `DataPrepConfig._init_keys_and_check`

```python
def DataPrepConfig._init_keys_and_check(
    self,
    configs,
    config_name='input',
)
```

1つの設定カテゴリのキーを収集し、重複キーを拒否します。

###### パラメータ

- **`configs`** (`Any`): 検査対象のデータ設定オブジェクト。
- **`config_name`** (`Any`): 収集する設定カテゴリ。通常は ``"input"`` または ``"target"``。

##### `DataPrepConfig._make_seq_info`

```python
def DataPrepConfig._make_seq_info(self) -> (Dict, int, int)
```

キーごとのシーケンスウィンドウ設定を構築します。

</details>

### `DataPreparator`

```python
class DataPreparator
```

サンプル単位の辞書に対して、データ準備パイプライン全体を実行します。

#### 注記

インスタンスには、検証済みの設定または本モジュールで使用するパイプライン状態が保存されます。

#### コンストラクタとプロトコルメソッド

##### `DataPreparator.__init__`

```python
def DataPreparator.__init__(
    self,
    data_dicts,
    data_prep_config,
    split_postprocess_fn=None,
)
```

データ準備クラスを初期化し、シーケンス関連のインデックスメタデータを導出します。

###### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`data_prep_config`** (`Any`): 準備済みデータとともに保存される設定オブジェクト。
- **`split_postprocess_fn`** (`Any`): 組み込み処理後にインデックス fold と参照値 fold を受け取る、オプションの callable。

#### 公開メソッド

##### `DataPreparator.run`

```python
def DataPreparator.run(self)
```

シーケンス収集、ターゲット処理、データ分割、オプションの保存を実行します。

###### 戻り値

tuple
    パイプライン全体で生成される ``(collected_data, info_dict)``。

##### `DataPreparator.get_seq_indexes`

```python
def DataPreparator.get_seq_indexes(self)
```

設定済みの各入力キーとターゲットキーについてシーケンスウィンドウを構築します。

##### `DataPreparator.gather_data`

```python
def DataPreparator.gather_data(self)
```

設定済みのデータフィールドを、整列した配列または Python リストとして収集します。

##### `DataPreparator.process_target`

```python
def DataPreparator.process_target(self)
```

ターゲット統計を作成し、指定されたターゲット変換を適用します。

##### `DataPreparator.convert_target_to_scalar`

```python
def DataPreparator.convert_target_to_scalar(self)
```

統計処理とデータ分割に使用するスカラーターゲットのリストを作成します。

##### `DataPreparator.make_target_info_dict`

```python
def DataPreparator.make_target_info_dict(self)
```

ラベルマッピング、回帰ビン、統計情報、インデックスグループを構築します。

###### 戻り値

dict
    ターゲット名をキーとするターゲットメタデータ。

##### `DataPreparator.convert_target_form`

```python
def DataPreparator.convert_target_form(self)
```

収集したターゲットをクラス ID または回帰ビンの代表値へ変換します。

##### `DataPreparator.split_data`

```python
def DataPreparator.split_data(self)
```

設定された戦略に従って、学習・検証・テストの fold を作成します。

##### `DataPreparator.post_split_process`

```python
def DataPreparator.post_split_process(self)
```

禁止されている分割間の重複を引き起こすシーケンスアンカーを削除します。

##### `DataPreparator.make_info_dict`

```python
def DataPreparator.make_info_dict(self)
```

保存および後続の実験に使用するデータ準備メタデータをまとめます。

##### `DataPreparator.get_zscore_miu_sigma`

```python
def DataPreparator.get_zscore_miu_sigma(self)
```

設定済み入力キーについて、fold ごとの正規化統計量を計算します。

###### 戻り値

list[dict]
    各 fold と分割に対応する正規化統計量。

##### `DataPreparator.get_input_shapes`

```python
def DataPreparator.get_input_shapes(self)
```

収集済みデータからモデル入力形状を推定します。

###### 戻り値

dict
    入力キーから推定された特徴量形状へのマッピング。

##### `DataPreparator.save_data`

```python
def DataPreparator.save_data(
    self,
    save_dir=None,
    overwrite_data=None,
)
```

明示的な引数または設定値を使って、準備済みデータとメタデータを保存します。

###### パラメータ

- **`save_dir`** (`Any`): 保存先ディレクトリ。``None`` の場合は設定済み保存ディレクトリを使用します。
- **`overwrite_data`** (`Any`): 既存のシリアライズ済みデータを置き換えてよいかどうか。

<details>
<summary><strong>内部メソッド（6）</strong></summary>

##### `DataPreparator._init_seq_indexes_info`

```python
def DataPreparator._init_seq_indexes_info(self)
```

シーケンス範囲、使用インデックス、元インデックスのマッピングを初期化します。

##### `DataPreparator._init_seq_ranges`

```python
def DataPreparator._init_seq_ranges(self)
```

グループまたはカスタム境界から半開のシーケンス範囲を作成します。

###### 戻り値

list[tuple[int, int]]
    ソート済みの半開シーケンス範囲。

##### `DataPreparator._validate_seq_ranges`

```python
def DataPreparator._validate_seq_ranges(
    self,
    ranges,
)
```

データセット長に対して、カスタムの半開シーケンス範囲を検証します。

###### パラメータ

- **`ranges`** (`Any`): 半開の ``(start, end)`` インデックス範囲。

##### `DataPreparator._merge_ranges`

```python
def DataPreparator._merge_ranges(
    self,
    ranges,
)
```

重複または接している半開範囲をマージします。

###### パラメータ

- **`ranges`** (`Any`): 半開の ``(start, end)`` インデックス範囲。

###### 戻り値

list[tuple[int, int]]
    マージ後の半開範囲。

##### `DataPreparator._get_used_indexes_from_ranges`

```python
def DataPreparator._get_used_indexes_from_ranges(self)
```

シーケンス範囲に含まれる一意なサンプルインデックスを収集します。

###### 戻り値

list[int]
    範囲に含まれる一意な元サンプルインデックス。

##### `DataPreparator._get_pure_shape`

```python
def DataPreparator._get_pure_shape(
    self,
    data,
)
```

バッチ次元とシーケンス次元を除いた後の特徴量形状を推定します。

###### パラメータ

- **`data`** (`Any`): ``_get_pure_shape`` が使用する値。

###### 戻り値

tuple
    バッチ次元とシーケンス次元を除いた後の特徴量次元。

</details>

### 公開関数

#### `pick_data_by_indexes`

```python
def pick_data_by_indexes(
    data_dicts: List[Dict],
    used_indexes: List,
) -> List
```

元インデックスに基づいてサンプル辞書を選択し、各元インデックスを記録します。

##### パラメータ

- **`data_dicts`** (`List[Dict]`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`used_indexes`** (`List`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。

##### 戻り値

list[dict]
    選択されたサンプル辞書。

#### `gather_data_single_key`

```python
def gather_data_single_key(
    data_dicts: List[Dict],
    data_key: Any,
    seq_indexes: List,
    dtype=None,
    seq_padding_index: int=-1,
    seq_padding_mode: Literal['constant', 'edge']='constant',
    seq_padding_value: Any=0.0,
    squeeze_singleton_dims: bool=True,
    keep_batch_seq_dims: bool=True,
) -> np.array
```

1つのデータキーに対するすべてのシーケンスウィンドウを収集します。

##### パラメータ

- **`data_dicts`** (`List[Dict]`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`data_key`** (`Any`): 値を収集する辞書キー。
- **`seq_indexes`** (`List`): アンカーインデックスと、そのシーケンスインデックスリストの組。
- **`dtype`** (`Any`): 数値出力に使用するオプションの NumPy dtype。
- **`seq_padding_index`** (`int`): パディング位置を表すセンチネルインデックス。
- **`seq_padding_mode`** (`Literal['constant', 'edge']`): パディング方式：定数値、または最も近い端の値を反復します。
- **`seq_padding_value`** (`Any`): 定数パディングに使用する値。
- **`squeeze_singleton_dims`** (`bool`): サイズ1の次元を削除するかどうか。
- **`keep_batch_seq_dims`** (`bool`): 数値出力に明示的なバッチ次元とシーケンス次元を残すかどうか。

##### 戻り値

numpy.ndarray or list
    すべてのシーケンスアンカーについて収集した値。

#### `gather_data_by_indexes`

```python
def gather_data_by_indexes(
    data_dicts,
    data_key,
    used_indexes=None,
    sample_data=None,
    data_operation='array',
    dtype=None,
    seq_padding_index=-1,
    seq_padding_mode: Literal['constant', 'edge']='constant',
    seq_padding_value=0,
    squeeze_singleton_dims=False,
)
```

定数パディングまたはエッジパディングを適用して、1つの値シーケンスを収集します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`data_key`** (`Any`): 値を収集する辞書キー。
- **`used_indexes`** (`Any`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。
- **`sample_data`** (`Any`): 型と形状の推定に使用するオプションの代表値。
- **`data_operation`** (`Any`): 配列、テンソル、ネストしたリスト、またはスカラーを収集する内部モード。
- **`dtype`** (`Any`): 数値出力に使用するオプションの NumPy dtype。
- **`seq_padding_index`** (`Any`): パディング位置を表すセンチネルインデックス。
- **`seq_padding_mode`** (`Literal['constant', 'edge']`): パディング方式：定数値、または最も近い端の値を反復します。
- **`seq_padding_value`** (`Any`): 定数パディングに使用する値。
- **`squeeze_singleton_dims`** (`Any`): サイズ1の次元を削除するかどうか。

##### 戻り値

numpy.ndarray or list
    1つのシーケンスについて収集した値。

#### `shift_get_seq_indexes`

```python
def shift_get_seq_indexes(
    index_list: List,
    seq_len_before: int,
    seq_len_after: int,
    step_offset: int=0,
    stride: int=1,
    seq_pos_from_start: int=0,
    seq_pos_from_end: int=0,
    padding: bool=True,
    padding_index: int=-1,
) -> List[Tuple[int, List]]
```

フィルタ後のアンカーインデックスを中心に、ストライド付きコンテキストウィンドウを構築します。

##### パラメータ

- **`index_list`** (`List`): 順序付きの元サンプルインデックス、または汎用的なインデックス相当値。
- **`seq_len_before`** (`int`): 各アンカーより前のコンテキストステップ数。
- **`seq_len_after`** (`int`): 各アンカーより後のコンテキストステップ数。
- **`step_offset`** (`int`): 設定済みデータキーに関連付けられた相対オフセット。
- **`stride`** (`int`): 隣接するコンテキスト位置の間隔。
- **`seq_pos_from_start`** (`int`): 範囲の先頭から除外する候補アンカー数。
- **`seq_pos_from_end`** (`int`): 範囲の末尾から除外する候補アンカー数。
- **`padding`** (`bool`): 不完全な境界ウィンドウをパディングするかどうか。
- **`padding_index`** (`int`): パディング位置に挿入するセンチネルインデックス。

##### 戻り値

tuple[list, list]
    シーケンスタプルと、そのアンカーインデックス。

#### `get_strided_seq`

```python
def get_strided_seq(
    index_list,
    i,
    stride,
    seq_len_before,
    seq_len_after,
)
```

インデックスリスト内の1つの位置を中心にストライド付きウィンドウを抽出します。

##### パラメータ

- **`index_list`** (`Any`): 順序付きの元サンプルインデックス、または汎用的なインデックス相当値。
- **`i`** (`Any`): ``index_list`` 内の中心位置。
- **`stride`** (`Any`): 隣接するコンテキスト位置の間隔。
- **`seq_len_before`** (`Any`): 各アンカーより前のコンテキストステップ数。
- **`seq_len_after`** (`Any`): 各アンカーより後のコンテキストステップ数。

##### 戻り値

list[int]
    指定されたストライドウィンドウ内の有効な位置。

#### `data_split_independent`

```python
def data_split_independent(
    data_dicts: List[Dict],
    split_ref_key: Any,
    split_mode: Literal['holdout', 'kfold', 'leave_one_out']='kfold',
    folds: int=5,
    train_valid_ratio: float=0.9,
    holdout_test_ratio: float=0.2,
    use_stratified_split: bool=False,
    split_valid_by: Literal['index', 'ref_key']='index',
    focused_target_key: Any='label',
    is_focused_key_multi_dim: bool=False,
    target2indexes: Optional[Dict]=None,
    focused_target_task_type: Literal['c', 'r']='c',
    stratified_bin_num: Optional[int]=None,
    stratified_bin_size: Optional[float]=None,
    used_indexes: Optional[List[int]]=None,
    train_ref_values_override: Optional[Union[Mapping, Sequence]]=None,
    valid_ref_values_override: Optional[Union[Mapping, Sequence]]=None,
    test_ref_values_override: Optional[Union[Mapping, Sequence]]=None,
) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[Any]]]]
```

参照グループを独立に分割し、テストグループが学習または検証に現れないようにします。

##### パラメータ

- **`data_dicts`** (`List[Dict]`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`split_ref_key`** (`Any`): 話者、参加者、セッションなどのグループを定義する辞書キー。
- **`split_mode`** (`Literal['holdout', 'kfold', 'leave_one_out']`): テスト分割方式：ホールドアウト、K-fold、または leave-one-out。
- **`folds`** (`int`): K-fold 分割で指定する fold 数。
- **`train_valid_ratio`** (`float`): テスト以外のデータのうち学習へ割り当てる割合。
- **`holdout_test_ratio`** (`float`): ホールドアウトモードでテストへ割り当てる割合。
- **`use_stratified_split`** (`bool`): ターゲット比率を保つためにスカラーターゲットを使用するかどうか。
- **`split_valid_by`** (`Literal['index', 'ref_key']`): 独立な検証データを、インデックスまたは参照グループのどちらで分離するか。
- **`focused_target_key`** (`Any`): 層化に使用するターゲットキー。
- **`is_focused_key_multi_dim`** (`bool`): 注目ターゲットが1サンプル当たり複数の値を含むかどうか。
- **`target2indexes`** (`Optional[Dict]`): ターゲットまたはビンから元サンプルインデックスへのオプションのマッピング。
- **`focused_target_task_type`** (`Literal['c', 'r']`): 注目ターゲットの種類。分類は ``"c"``、回帰は ``"r"``。
- **`stratified_bin_num`** (`Optional[int]`): 層化に使用する回帰ビン数。
- **`stratified_bin_size`** (`Optional[float]`): 層化に使用する回帰ビン幅。
- **`used_indexes`** (`Optional[List[int]]`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。
- **`train_ref_values_override`** (`Optional[Union[Mapping, Sequence]]`): オプションの学習用参照グループ。全 fold で共有するか、fold ごとに指定できます。
- **`valid_ref_values_override`** (`Optional[Union[Mapping, Sequence]]`): オプションの検証用参照グループ。全 fold で共有するか、fold ごとに指定できます。
- **`test_ref_values_override`** (`Optional[Union[Mapping, Sequence]]`): 1つ以上の fold に指定するオプションのテスト用参照グループ。

##### 戻り値

tuple[dict, dict]
    インデックス fold と対応する参照値 fold。

#### `data_split_dependent`

```python
def data_split_dependent(
    data_dicts: List[Dict],
    split_ref_key: Any,
    split_mode: Literal['holdout', 'kfold', 'leave_one_out']='kfold',
    folds: int=5,
    train_valid_ratio: float=0.9,
    holdout_test_ratio: float=0.2,
    use_stratified_split: bool=False,
    split_valid_by: Literal['index', 'ref_key']='index',
    focused_target_key: Any='label',
    is_focused_key_multi_dim: bool=False,
    target2indexes: Optional[Dict]=None,
    focused_target_task_type: Literal['c', 'r']='c',
    stratified_bin_num: Optional[int]=None,
    stratified_bin_size: Optional[float]=None,
    used_indexes: Optional[List[int]]=None,
) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[Any]]]]
```

各参照グループ内でサンプルを分割するため、同じグループがすべてのデータセットに現れる場合があります。

##### パラメータ

- **`data_dicts`** (`List[Dict]`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`split_ref_key`** (`Any`): 話者、参加者、セッションなどのグループを定義する辞書キー。
- **`split_mode`** (`Literal['holdout', 'kfold', 'leave_one_out']`): テスト分割方式：ホールドアウト、K-fold、または leave-one-out。
- **`folds`** (`int`): K-fold 分割で指定する fold 数。
- **`train_valid_ratio`** (`float`): テスト以外のデータのうち学習へ割り当てる割合。
- **`holdout_test_ratio`** (`float`): ホールドアウトモードでテストへ割り当てる割合。
- **`use_stratified_split`** (`bool`): ターゲット比率を保つためにスカラーターゲットを使用するかどうか。
- **`split_valid_by`** (`Literal['index', 'ref_key']`): 独立な検証データを、インデックスまたは参照グループのどちらで分離するか。
- **`focused_target_key`** (`Any`): 層化に使用するターゲットキー。
- **`is_focused_key_multi_dim`** (`bool`): 注目ターゲットが1サンプル当たり複数の値を含むかどうか。
- **`target2indexes`** (`Optional[Dict]`): ターゲットまたはビンから元サンプルインデックスへのオプションのマッピング。
- **`focused_target_task_type`** (`Literal['c', 'r']`): 注目ターゲットの種類。分類は ``"c"``、回帰は ``"r"``。
- **`stratified_bin_num`** (`Optional[int]`): 層化に使用する回帰ビン数。
- **`stratified_bin_size`** (`Optional[float]`): 層化に使用する回帰ビン幅。
- **`used_indexes`** (`Optional[List[int]]`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。

##### 戻り値

tuple[dict, dict]
    インデックス fold と対応する参照値 fold。

#### `data_split_unconstrained`

```python
def data_split_unconstrained(
    data_dicts: List[Dict],
    split_ref_key: Any=None,
    split_mode: Literal['holdout', 'kfold', 'leave_one_out']='kfold',
    folds: int=5,
    train_valid_ratio: float=0.9,
    holdout_test_ratio: float=0.2,
    use_stratified_split: bool=False,
    split_valid_by: Literal['index', 'ref_key']='index',
    target2indexes: Optional[Dict]=None,
    focused_target_key: Any='label',
    is_focused_key_multi_dim: bool=False,
    focused_target_task_type: Literal['c', 'r']='c',
    stratified_bin_num: Optional[int]=None,
    stratified_bin_size: Optional[float]=None,
    used_indexes: Optional[List[int]]=None,
) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[Any]]]]
```

参照グループの独立性を強制せずに、サンプルインデックスを分割します。

##### パラメータ

- **`data_dicts`** (`List[Dict]`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`split_ref_key`** (`Any`): 話者、参加者、セッションなどのグループを定義する辞書キー。
- **`split_mode`** (`Literal['holdout', 'kfold', 'leave_one_out']`): テスト分割方式：ホールドアウト、K-fold、または leave-one-out。
- **`folds`** (`int`): K-fold 分割で指定する fold 数。
- **`train_valid_ratio`** (`float`): テスト以外のデータのうち学習へ割り当てる割合。
- **`holdout_test_ratio`** (`float`): ホールドアウトモードでテストへ割り当てる割合。
- **`use_stratified_split`** (`bool`): ターゲット比率を保つためにスカラーターゲットを使用するかどうか。
- **`split_valid_by`** (`Literal['index', 'ref_key']`): 独立な検証データを、インデックスまたは参照グループのどちらで分離するか。
- **`target2indexes`** (`Optional[Dict]`): ターゲットまたはビンから元サンプルインデックスへのオプションのマッピング。
- **`focused_target_key`** (`Any`): 層化に使用するターゲットキー。
- **`is_focused_key_multi_dim`** (`bool`): 注目ターゲットが1サンプル当たり複数の値を含むかどうか。
- **`focused_target_task_type`** (`Literal['c', 'r']`): 注目ターゲットの種類。分類は ``"c"``、回帰は ``"r"``。
- **`stratified_bin_num`** (`Optional[int]`): 層化に使用する回帰ビン数。
- **`stratified_bin_size`** (`Optional[float]`): 層化に使用する回帰ビン幅。
- **`used_indexes`** (`Optional[List[int]]`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。

##### 戻り値

tuple[dict, dict]
    インデックス fold と対応する参照値 fold。

#### `remove_overlapped_seq_split`

```python
def remove_overlapped_seq_split(
    data_dicts,
    index_split_folds,
    anchore_index2seq_indexes,
    *,
    split_ref_key='group',
    is_test_train_no_seq_overlap: bool=True,
    is_train_valid_no_seq_overlap: bool=True,
    priority_order=('test', 'train', 'valid'),
    padding_index: int=-1,
)
```

シーケンス内容が保護対象の分割と重複する、優先度の低いアンカーを削除します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`index_split_folds`** (`Any`): 各 fold の学習・検証・テストに割り当てられた元インデックス。
- **`anchore_index2seq_indexes`** (`Any`): 各アンカーから、そのシーケンスで使用するすべての元インデックスへのマッピング。
- **`split_ref_key`** (`Any`): 話者、参加者、セッションなどのグループを定義する辞書キー。
- **`is_test_train_no_seq_overlap`** (`bool`): 学習・テスト間のシーケンス重複を禁止するかどうか。
- **`is_train_valid_no_seq_overlap`** (`bool`): 学習・検証間のシーケンス重複を禁止するかどうか。
- **`priority_order`** (`Any`): 重複除去で使用する、分割の優先順位（高い順）。
- **`padding_index`** (`int`): パディング位置に挿入するセンチネルインデックス。

##### 戻り値

tuple[dict, dict]
    フィルタ後のインデックス fold と再構築した参照値 fold。

#### `get_target_info_cls`

```python
def get_target_info_cls(
    data_dicts,
    target_ref_key,
    used_indexes=None,
)
```

クラス統計、クラス ID マッピング、元インデックスのグループを構築します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`target_ref_key`** (`Any`): ターゲット値を含む辞書キー。
- **`used_indexes`** (`Any`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。

##### 戻り値

tuple
    クラス統計、クラスマッピング、ターゲットからインデックスへのマッピング。

#### `get_target_info_regression`

```python
def get_target_info_regression(
    data_dicts,
    target_ref_key,
    used_indexes=None,
    stratified_bin_size=None,
    stratified_bin_num=None,
    bin_closed_side: Literal['upper', 'lower']='lower',
)
```

スカラー回帰ターゲットをビン分けし、ビン統計とインデックスを収集します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`target_ref_key`** (`Any`): ターゲット値を含む辞書キー。
- **`used_indexes`** (`Any`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。
- **`stratified_bin_size`** (`Any`): 層化に使用する回帰ビン幅。
- **`stratified_bin_num`** (`Any`): 層化に使用する回帰ビン数。
- **`bin_closed_side`** (`Literal['upper', 'lower']`): 回帰ビンの境界規則。

##### 戻り値

tuple
    ビン統計、ビン範囲、ビンからインデックスへのマッピング。

#### `get_target2indexes`

```python
def get_target2indexes(
    data_dicts,
    target_ref_key,
    used_indexes=None,
    task_type='c',
    stratified_bin_size: Optional[float]=None,
    stratified_bin_num: Optional[int]=10,
)
```

各クラスまたは回帰ビンを元のサンプルインデックスへ対応付けます。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`target_ref_key`** (`Any`): ターゲット値を含む辞書キー。
- **`used_indexes`** (`Any`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。
- **`task_type`** (`Any`): ターゲットの種類：分類または回帰。
- **`stratified_bin_size`** (`Optional[float]`): 層化に使用する回帰ビン幅。
- **`stratified_bin_num`** (`Optional[int]`): 層化に使用する回帰ビン数。

##### 戻り値

dict
    クラス値または回帰ビンから元インデックスへのマッピング。

#### `get_stratified_bin_info`

```python
def get_stratified_bin_info(
    data_dicts,
    target_ref_key,
    used_indexes=None,
    bin_size=None,
    bin_num=None,
    target_list=None,
)
```

実際の回帰ビン幅とビン数を計算します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`target_ref_key`** (`Any`): ターゲット値を含む辞書キー。
- **`used_indexes`** (`Any`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。
- **`bin_size`** (`Any`): 指定する回帰ビン幅。
- **`bin_num`** (`Any`): 指定する回帰ビン数。
- **`target_list`** (`Any`): 事前に収集したオプションのスカラーターゲット。

##### 戻り値

tuple[float, int]
    実際のビン幅とビン数。

#### `generate_config_template`

```python
def generate_config_template(
    file: str='./data_prep_config.json',
    cls_target_keys: Union[List, str, None]=None,
    regression_target_keys: Union[List, str, None]=None,
    input_keys: Union[List, str, None]=None,
)
```

指定されたデータキー用の JSON 設定テンプレートを書き出します。

##### パラメータ

- **`file`** (`str`): JSON 設定ファイルのパス。
- **`cls_target_keys`** (`Union[List, str, None]`): 1つ以上の分類ターゲットキー。
- **`regression_target_keys`** (`Union[List, str, None]`): 1つ以上の回帰ターゲットキー。
- **`input_keys`** (`Union[List, str, None]`): 1つ以上のモデル入力キー。

#### `load_config`

```python
def load_config(file: str='./data_prep_config.json')
```

JSON 設定ファイルを設定オブジェクトとして読み込みます。

##### パラメータ

- **`file`** (`str`): JSON 設定ファイルのパス。

##### 戻り値

DataPrepConfig
    再構築されたデータ準備設定。

#### `make_config_from_json`

```python
def make_config_from_json(config_json)
```

解析済み JSON データからネストした設定オブジェクトを再構築します。

##### パラメータ

- **`config_json`** (`Any`): ``to_json`` が生成した解析済み設定表現。

##### 戻り値

DataPrepConfig
    再構築されたデータ準備設定。

#### `save_data`

```python
def save_data(
    collected_data,
    info_dict,
    data_prep_config,
    save_dir='./DataExperiment',
    overwrite_data=False,
)
```

明示的な引数または設定値を使って、準備済みデータとメタデータを保存します。

##### パラメータ

- **`collected_data`** (`Any`): 準備済みの入力・ターゲット配列またはリスト。
- **`info_dict`** (`Any`): データ準備メタデータと分割情報。
- **`data_prep_config`** (`Any`): 準備済みデータとともに保存される設定オブジェクト。
- **`save_dir`** (`Any`): 保存先ディレクトリ。``None`` の場合は設定済み保存ディレクトリを使用します。
- **`overwrite_data`** (`Any`): 既存のシリアライズ済みデータを置き換えてよいかどうか。

#### `load_data`

```python
def load_data(data_dir='./DataExperiment')
```

ディレクトリから準備済みデータ、メタデータ、設定を読み込みます。

##### パラメータ

- **`data_dir`** (`Any`): シリアライズ済みの準備データファイルを含むディレクトリ。

##### 戻り値

tuple
    準備済みデータ、メタデータ、設定。

#### `get_target_list`

```python
def get_target_list(
    data_dicts,
    target_ref_key,
    used_indexes=None,
)
```

選択された元サンプルインデックスからターゲット値を収集します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`target_ref_key`** (`Any`): ターゲット値を含む辞書キー。
- **`used_indexes`** (`Any`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。

##### 戻り値

list
    選択インデックス順のターゲット値。

#### `get_scalar_target_list`

```python
def get_scalar_target_list(
    data_dicts,
    target_ref_key,
    used_indexes=None,
)
```

選択されたターゲット値を収集し、Python スカラーへ正規化します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`target_ref_key`** (`Any`): ターゲット値を含む辞書キー。
- **`used_indexes`** (`Any`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。

##### 戻り値

list
    選択インデックス順の Python スカラーターゲット。

#### `convert_to_python_scalar`

```python
def convert_to_python_scalar(target_value)
```

スカラー相当の Python、NumPy、または PyTorch 値を Python スカラーへ変換します。

##### パラメータ

- **`target_value`** (`Any`): 変換するスカラー、または要素が1つのスカラー相当値。

##### 戻り値

object
    等価な Python スカラー値。

#### `calculate_miu_sigma`

```python
def calculate_miu_sigma(
    data_dicts,
    ref_key,
    index_list,
)
```

選択されたサンプルについて、特徴量ごとの平均と標準偏差を計算します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`ref_key`** (`Any`): 計算に使用する値を含む辞書キー。
- **`index_list`** (`Any`): 順序付きの元サンプルインデックス、または汎用的なインデックス相当値。

##### 戻り値

tuple[numpy.ndarray, numpy.ndarray]
    特徴量ごとの平均と標準偏差。

<details>
<summary><strong>内部モジュールヘルパー（25）</strong></summary>

これらの関数は公開 API を支える内部関数です。ソースに docstring が含まれるため本書にも掲載していますが、ユーザーコードが直接依存する互換性インターフェースとしては推奨されません。

#### `_get_test_indexes`

```python
def _get_test_indexes(
    indexes_list,
    current_split_group,
    num_split_group,
    one_fold_test_ratio,
    split_mode='kfold',
)
```

1つの fold と分割モードに対応するテストインデックスを選択します。

##### パラメータ

- **`indexes_list`** (`Any`): 現在のテスト選択で使用できる順序付きインデックス。
- **`current_split_group`** (`Any`): 0始まりの fold 位置。
- **`num_split_group`** (`Any`): 分割グループの総数。
- **`one_fold_test_ratio`** (`Any`): ``split_mode="holdout"`` の場合に使用するホールドアウト比率。
- **`split_mode`** (`Any`): テスト分割方式：ホールドアウト、K-fold、または leave-one-out。

##### 戻り値

list[int]
    指定されたテスト fold に割り当てられたインデックス。

#### `_distribute_indexes_to_folds`

```python
def _distribute_indexes_to_folds(
    index_list,
    current_group_index,
)
```

指定された fold 位置へ最大1つのインデックスを割り当てます。

##### パラメータ

- **`index_list`** (`Any`): 順序付きの元サンプルインデックス、または汎用的なインデックス相当値。
- **`current_group_index`** (`Any`): 0始まりのグループ位置。

##### 戻り値

list[int]
    要素が1つのリスト、または空リスト。

#### `_summary_seq_index_list`

```python
def _summary_seq_index_list(
    anchore_index_list,
    anchore_index2seq_index,
)
```

一連のアンカーが参照する一意なシーケンスインデックスを収集します。

##### パラメータ

- **`anchore_index_list`** (`Any`): シーケンス内容を要約するアンカーインデックス。
- **`anchore_index2seq_index`** (`Any`): アンカーインデックスからシーケンスインデックスリストへのマッピング。

##### 戻り値

list[int]
    ソート済みの一意なシーケンスインデックス。

#### `_validate_split_parameters`

```python
def _validate_split_parameters(
    split_mode,
    folds,
    train_valid_ratio,
    holdout_test_ratio,
)
```

一般的な分割モード、fold 数、比率を検証します。

##### パラメータ

- **`split_mode`** (`Any`): テスト分割方式：ホールドアウト、K-fold、または leave-one-out。
- **`folds`** (`Any`): K-fold 分割で指定する fold 数。
- **`train_valid_ratio`** (`Any`): テスト以外のデータのうち学習へ割り当てる割合。
- **`holdout_test_ratio`** (`Any`): ホールドアウトモードでテストへ割り当てる割合。

#### `_normalize_used_indexes`

```python
def _normalize_used_indexes(
    data_dicts,
    used_indexes,
)
```

データ準備で使用する元サンプルインデックスを解決・検証します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`used_indexes`** (`Any`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。

##### 戻り値

list[int]
    検証済みの元サンプルインデックス。

#### `_balanced_chunks`

```python
def _balanced_chunks(
    values,
    num_chunks,
    reverse=False,
)
```

順序付きの値を、ほぼ同じ大きさのチャンクへ分割します。

##### パラメータ

- **`values`** (`Any`): 処理対象の値。
- **`num_chunks`** (`Any`): 作成するチャンク数。
- **`reverse`** (`Any`): 分割後にチャンク順を反転するかどうか。

##### 戻り値

list[list]
    ほぼ同じ大きさの順序付きチャンク。

#### `_get_holdout_size`

```python
def _get_holdout_size(
    num_items,
    ratio,
)
```

可能な場合は学習データを残しながら、空でないホールドアウトサイズを計算します。

##### パラメータ

- **`num_items`** (`Any`): 利用可能な項目数。
- **`ratio`** (`Any`): 指定された比率。

##### 戻り値

int
    ホールドアウトテストデータへ割り当てる項目数。

#### `_split_train_valid_indexes`

```python
def _split_train_valid_indexes(
    indexes,
    train_ratio,
)
```

可能な場合、順序付きインデックスを空でない学習・検証サブセットへ分割します。

##### パラメータ

- **`indexes`** (`Any`): 順序付きの候補インデックス。
- **`train_ratio`** (`Any`): 学習へ割り当てる割合。

##### 戻り値

tuple[list, list]
    学習・検証インデックスのリスト。

#### `_stratified_train_valid_split`

```python
def _stratified_train_valid_split(
    indexes,
    target2indexes,
    train_ratio,
)
```

各ターゲット層の中で候補インデックスを分割します。

##### パラメータ

- **`indexes`** (`Any`): 順序付きの候補インデックス。
- **`target2indexes`** (`Any`): ターゲットまたはビンから元サンプルインデックスへのオプションのマッピング。
- **`train_ratio`** (`Any`): 学習へ割り当てる割合。

##### 戻り値

tuple[list, list]
    層化された学習・検証インデックスのリスト。

#### `_prepare_target2indexes`

```python
def _prepare_target2indexes(
    data_dicts,
    target_key,
    used_indexes,
    task_type,
    target2indexes,
    stratified_bin_size,
    stratified_bin_num,
)
```

ターゲットから元インデックスへのマッピングを作成またはフィルタします。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`target_key`** (`Any`): ターゲット値を含む辞書キー。
- **`used_indexes`** (`Any`): 処理対象の元サンプルインデックス。``None`` の場合はすべてのサンプルを選択します。
- **`task_type`** (`Any`): ターゲットの種類：分類または回帰。
- **`target2indexes`** (`Any`): ターゲットまたはビンから元サンプルインデックスへのオプションのマッピング。
- **`stratified_bin_size`** (`Any`): 層化に使用する回帰ビン幅。
- **`stratified_bin_num`** (`Any`): 層化に使用する回帰ビン数。

##### 戻り値

dict
    選択インデックスに限定したターゲットまたはビンのマッピング。

#### `_normalize_fold_override`

```python
def _normalize_fold_override(override)
```

分割オーバーライドを、fold ごとの値リストへ正規化します。

##### パラメータ

- **`override`** (`Any`): 共有形式または fold ごとの形式でユーザーが指定した分割値。

##### 戻り値

list[list]
    fold ごとのオーバーライド値。

#### `_get_fold_override`

```python
def _get_fold_override(
    override,
    fold,
)
```

1つの共有オーバーライドを許可しつつ、1つの fold のオーバーライドを読み取ります。

##### パラメータ

- **`override`** (`Any`): 共有形式または fold ごとの形式でユーザーが指定した分割値。
- **`fold`** (`Any`): 0始まりの fold 識別子。

##### 戻り値

list or None
    指定された fold に割り当てられた値。

#### `_resolve_train_valid_ref_values`

```python
def _resolve_train_valid_ref_values(
    candidate_values,
    train_ratio,
    train_override,
    valid_override,
)
```

自動生成またはユーザー定義の学習・検証用参照グループを解決します。

##### パラメータ

- **`candidate_values`** (`Any`): テストへ割り当てられていない参照値。
- **`train_ratio`** (`Any`): 学習へ割り当てる割合。
- **`train_override`** (`Any`): 明示的に割り当てるオプションの学習用参照値。
- **`valid_override`** (`Any`): 明示的に割り当てるオプションの検証用参照値。

##### 戻り値

tuple[list, list]
    学習・検証用の参照値。

#### `_validate_ref_values`

```python
def _validate_ref_values(
    values,
    available_values,
    split_name,
)
```

オーバーライドの参照値が利用可能なグループ内に存在することを確認します。

##### パラメータ

- **`values`** (`Any`): 処理対象の値。
- **`available_values`** (`Any`): 割り当てに使用できる参照値。
- **`split_name`** (`Any`): エラーメッセージで使用する、人が読みやすい分割名。

#### `_build_ref_value_split_dict`

```python
def _build_ref_value_split_dict(
    data_dicts,
    split_dict,
    split_ref_key,
)
```

元インデックスから各分割の一意な参照値を導出します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`split_dict`** (`Any`): 1つの学習・検証・テスト用インデックス辞書。
- **`split_ref_key`** (`Any`): 話者、参加者、セッションなどのグループを定義する辞書キー。

##### 戻り値

dict
    学習・検証・テストの一意な参照値。

#### `_build_ref_value_split_folds`

```python
def _build_ref_value_split_folds(
    data_dicts,
    index_split_folds,
    split_ref_key,
)
```

各インデックス fold の参照値要約を導出します。

##### パラメータ

- **`data_dicts`** (`Any`): 元サンプル位置でインデックス付けされた、順序付きのサンプル単位辞書。
- **`index_split_folds`** (`Any`): 各 fold の学習・検証・テストに割り当てられた元インデックス。
- **`split_ref_key`** (`Any`): 話者、参加者、セッションなどのグループを定義する辞書キー。

##### 戻り値

dict
    fold をキーとする参照値の要約。

#### `_normalize_index_split_folds`

```python
def _normalize_index_split_folds(index_split_folds)
```

リスト形式またはマッピング形式の fold を、一貫した整数キーのマッピングへ正規化します。

##### パラメータ

- **`index_split_folds`** (`Any`): 各 fold の学習・検証・テストに割り当てられた元インデックス。

##### 戻り値

dict
    整数キーと標準分割キーを持つ fold マッピング。

#### `_validate_split_dict`

```python
def _validate_split_dict(
    split_dict,
    allowed_indexes,
)
```

1つの fold の各分割が互いに重複せず、完全で、許可されたインデックスだけを含むことを確認します。

##### パラメータ

- **`split_dict`** (`Any`): 1つの学習・検証・テスト用インデックス辞書。
- **`allowed_indexes`** (`Any`): 必ず1回だけ割り当てる必要があるインデックスの完全な集合。

#### `_flatten`

```python
def _flatten(iterables)
```

ネストしたイテラブルを1階層平坦化します。

##### パラメータ

- **`iterables`** (`Any`): 1階層平坦化するネストしたイテラブル。

##### 戻り値

list
    1階層平坦化した値。

#### `_deduplicate_preserve_order`

```python
def _deduplicate_preserve_order(values)
```

初出順を保ったまま重複値を削除します。

##### パラメータ

- **`values`** (`Any`): 処理対象の値。

##### 戻り値

list
    初出順の一意な値。

#### `_make_regression_bin_ranges`

```python
def _make_regression_bin_ranges(
    values,
    bin_size=None,
    bin_num=None,
)
```

観測されたターゲット値の範囲を覆う連続したビン境界を作成します。

##### パラメータ

- **`values`** (`Any`): 処理対象の値。
- **`bin_size`** (`Any`): 指定する回帰ビン幅。
- **`bin_num`** (`Any`): 指定する回帰ビン数。

##### 戻り値

list[tuple[float, float]]
    連続するビンの下限・上限境界。

#### `_find_bin_id`

```python
def _find_bin_id(
    value,
    bin_ranges,
    bin_closed_side,
)
```

1つのスカラー回帰ターゲットを含むビンを検索します。

##### パラメータ

- **`value`** (`Any`): スカラー回帰ターゲット。
- **`bin_ranges`** (`Any`): ビン ID から下限・上限境界へのマッピング。
- **`bin_closed_side`** (`Any`): 回帰ビンの境界規則。

##### 戻り値

int
    値を含む回帰ビンの識別子。

#### `_convert_value_to_bin`

```python
def _convert_value_to_bin(
    value,
    bin_ranges,
    bin_closed_side,
)
```

スカラーターゲットを、そのビンに設定された境界の代表値へ変換します。

##### パラメータ

- **`value`** (`Any`): スカラー回帰ターゲット。
- **`bin_ranges`** (`Any`): ビン ID から下限・上限境界へのマッピング。
- **`bin_closed_side`** (`Any`): 回帰ビンの境界規則。

##### 戻り値

float
    ターゲットビンを表す、選択された下限または上限境界。

#### `_init_split_set_dict`

```python
def _init_split_set_dict()
```

空の学習・検証・テスト辞書を作成します。

##### 戻り値

dict[str, list]
    ``train``、``valid``、``test`` 用の空リスト。

#### `_unify_to_list`

```python
def _unify_to_list(content)
```

スカラー、タプル、または NumPy 配列を Python リストへ正規化します。

##### パラメータ

- **`content`** (`Any`): リストへ正規化するスカラーまたはコレクション。

##### 戻り値

list
    正規化された Python リスト。

</details>

---

## `flexmm.experiment`

[GitHub でソースを表示](https://github.com/Xia-code/flexmm/blob/main/flexmm/experiment.py)

実験設定、反復処理、データ変換、評価のためのツールです。

本モジュールでは、実験フローの管理を次の3つの概念に分けています。

``ExperimentManager``
    共有データを読み込むか準備し、反復のたびに新しい実験イテレータを作成します。

``ExperimentUnit``
    1つの入力組み合わせおよび1つの fold 条件に対応する学習・検証・テストデータを保持します。

``RunContext``
    個別の条件を実行・保存するために必要なメタデータを保持します。

準備済みデータの分割インデックスは、元の ``data_dicts``
のインデックスを参照することを想定しています。``collected_data`` を参照する前に、管理クラスは利用可能な場合、
``info_dict['ori_index2id']`` を使って準備済みデータ上の位置へ変換します。

### API 概要

#### クラス

| クラス | 概要 |
| --- | --- |
| [`ExperimentConfig`](#experimentconfig) | 実験組み合わせ、データ出力、再現性を設定します。 |
| [`TorchExperimentConfig`](#torchexperimentconfig) | PyTorch DataLoader の設定を追加して :class:`ExperimentConfig` を拡張します。 |
| [`ExperimentContext`](#experimentcontext) | 1つの入力組み合わせと fold 条件に対応する実行時メタデータを保持します。 |
| [`ExperimentUnit`](#experimentunit) | 1つの実行可能な実験条件のデータとコンテキストを表します。 |
| [`ExperimentManager`](#experimentmanager) | 共有実験状態を準備し、実験ユニットを反復処理します。 |
| [`ExperimentResultLoader`](#experimentresultloader) | 保存済みの実験設定、データ準備設定、結果を読み込みます。 |

#### 公開関数

| 関数 | 概要 |
| --- | --- |
| `generate_key_combs()` | 指定されたキーの空でないすべての組み合わせを生成します。 |
| `load_prepared_data()` | データ、準備情報、準備設定を読み込みます。 |
| `iter_experiment_units()` | 入力組み合わせと fold ごとに1つの :class:`ExperimentUnit` を生成します。 |
| `make_data_generator()` | 実験ユニットのジェネレータを作成します。 |
| `make_data_geneartor()` | :func:`make_data_generator` の非推奨の誤記エイリアス。 |
| `collect_combination_data()` | キーのグループについて、指定された準備済みデータ位置を選択します。 |
| `perform_zscore()` | 分散ゼロへの保護を適用して数値データを標準化します。 |
| `get_input_target_shapes()` | データセットで利用できる、設定済み入力形状とターゲット形状を収集します。 |
| `make_dataset()` | 辞書ベースの PyTorch データセットを作成します。 |
| `convert_single_data_to_tensor()` | 対応する分割データを PyTorch テンソルへ変換します。 |
| `make_dataloader()` | 空でないデータセットから PyTorch DataLoader を作成します。 |
| `torch_postprocess()` | テンソルまたは配列状の値を NumPy 評価配列へ変換します。 |
| `load_exp_config()` | 保存済みの実験設定 JSON ファイルを読み込みます。 |
| `make_config_from_json()` | 保存済み JSON 内容から実験設定を構築します。 |
| `compute_cls_metrics()` | 一般的な単一ラベル分類指標を計算します。 |
| `compute_regression_metrics()` | 回帰用の MAE、MSE、RMSE、Pearson 相関係数を計算します。 |
| `compute_target_key_average_metric()` | 1つのターゲットキーの下にネストされた指標を平均します。 |
| `compute_average_metric()` | 複数の結果辞書に含まれる最上位の指標を平均します。 |
| `change_average_summary_form()` | 平均値の要約を、指定された分析しやすい形式へ変換します。 |
| `change_average_summary_lists()` | 組み合わせの要約を、対応する組み合わせリストと指標リストへ変換します。 |

### クラス

### `ExperimentConfig`

```python
class ExperimentConfig
```

実験組み合わせ、データ出力、再現性を設定します。

#### パラメータ

- **`experiment_input_keys`** (`Any or list[Any]`): 実験で使用する候補入力キー。
- **`generate_input_comb`** (`bool, default=True`): ``True`` の場合は候補入力キーの空でないすべての組み合わせを生成します。``False`` の場合は、すべての候補キーを1つの組み合わせとして使用します。
- **`experiment_target_keys`** (`Any or list[Any]`): 各実験ユニットに含めるターゲットキー。
- **`input_comb_custom`** (`list[Any or list[Any]], optional`): 明示的な入力組み合わせ。指定した場合、この値は ``experiment_input_keys`` と ``generate_input_comb`` を上書きします。
- **`input_key_abbr`** (`dict or list, optional`): 組み合わせのディレクトリ名を作成するための短いラベル。リストは確定した入力キーの順序に合わせる必要があり、辞書には確定した入力キーを過不足なく含める必要があります。
- **`random_seed`** (`int, optional`): 基準となる乱数シード。省略時はシードを自動生成します。
- **`random_seed_scope`** (`list[str]`): シードを設定する乱数系。対応する値は ``"random"``、``"numpy"``、``"torch"``。
- **`data_level`** (`{"raw", "dataset", "dataloader"}`): 各分割から生成されるデータの表現レベル。
- **`data_representation`** (`{"original", "pt"}`): データセット内部のデータ表現。``"pt"`` は対応する配列を PyTorch テンソルへ変換します。
- **`load_prepared_data`** (`bool, default=True`): ``data_dicts`` からデータ準備を実行せず、``store_dir`` から準備済みデータを読み込みます。
- **`store_dir`** (`str`): 準備済みデータ、実験設定、結果ファイルを保存するルートディレクトリ。
- **`debug_flag`** (`int`): 設定に保持されるユーザー制御のデバッグフラグ。

#### 属性

input_keys : list[Any]
    確定した一意の入力キー。
input_combs : list[list[Any]]
    確定した入力組み合わせ。
target_keys : list[Any]
    確定したターゲットキー。

#### 宣言フィールド

| フィールド | 型 | デフォルト |
| --- | --- | --- |
| `experiment_input_keys` | `Union[Any, List[Any]]` | `field(default_factory=list)` |
| `generate_input_comb` | `bool` | `True` |
| `experiment_target_keys` | `Union[Any, List[Any]]` | `field(default_factory=list)` |
| `input_comb_custom` | `Optional[List[Union[Any, List[Any]]]]` | `None` |
| `input_key_abbr` | `Optional[Union[Dict[Any, Any], List[Any]]]` | `None` |
| `random_seed` | `Optional[int]` | `None` |
| `random_seed_scope` | `List[str]` | `field(default_factory=lambda: ['random', 'numpy', 'torch'])` |
| `data_level` | `DataLevel` | `'raw'` |
| `data_representation` | `DataRepresentation` | `'original'` |
| `load_prepared_data` | `bool` | `True` |
| `store_dir` | `str` | `'./ExperimentStore'` |
| `debug_flag` | `int` | `0` |

#### 公開メソッド

##### `ExperimentConfig.assert_config`

```python
def ExperimentConfig.assert_config(self) -> None
```

一般的な実験設定を検証します。

###### 例外

ValueError
    カテゴリ設定または乱数シードの適用範囲が無効な場合。

##### `ExperimentConfig.to_json`

```python
def ExperimentConfig.to_json(self) -> tuple[str, Dict[str, Any]]
```

dataclass フィールドをシリアライズ可能な設定形式へ変換します。

###### 戻り値

tuple[str, dict]
    クラス名と dataclass フィールドの辞書。

<details>
<summary><strong>内部メソッド（2）</strong></summary>

##### `ExperimentConfig.__post_init__`

```python
def ExperimentConfig.__post_init__(self) -> None
```

設定フィールドを正規化し、確定した設定を検証します。

##### `ExperimentConfig._normalize_input_key_abbr`

```python
def ExperimentConfig._normalize_input_key_abbr(
    self,
    abbreviations: Optional[Union[Dict[Any, Any], List[Any]]],
) -> Dict[Any, Any]
```

入力キーの略称を完全な辞書へ正規化します。

###### パラメータ

- **`abbreviations`** (`dict, list, or None`): ユーザーが指定した略称。

###### 戻り値

dict
    確定した各入力キーから略称へのマッピング。

</details>

### `TorchExperimentConfig`

```python
class TorchExperimentConfig(ExperimentConfig)
```

PyTorch DataLoader の設定を追加して :class:`ExperimentConfig` を拡張します。

#### パラメータ

- **`train_batch_size, valid_batch_size, test_batch_size`** (`int`): 対応する各分割で使用するバッチサイズ。
- **`shuffle_train_data, shuffle_valid_data, shuffle_test_data`** (`bool`): 対応する DataLoader がデータセットをシャッフルするかどうか。
- **`use_gpu`** (`bool`): テンソル変換ヘルパーがテンソルを CUDA に配置してよいかどうか。通常は学習ループ内でテンソルをモデルのデバイスへ移動する方が柔軟なため、デフォルトのデータセット処理ではテンソルを CPU に保持します。

#### 宣言フィールド

| フィールド | 型 | デフォルト |
| --- | --- | --- |
| `train_batch_size` | `int` | `4` |
| `valid_batch_size` | `int` | `1` |
| `test_batch_size` | `int` | `1` |
| `shuffle_train_data` | `bool` | `True` |
| `shuffle_valid_data` | `bool` | `False` |
| `shuffle_test_data` | `bool` | `False` |
| `use_gpu` | `bool` | `USE_GPU_DEFAULT` |

<details>
<summary><strong>内部メソッド（1）</strong></summary>

##### `TorchExperimentConfig.__post_init__`

```python
def TorchExperimentConfig.__post_init__(self) -> None
```

継承したフィールドを初期化し、PyTorch 固有の値を検証します。

</details>

### `ExperimentContext`

```python
class ExperimentContext
```

1つの入力組み合わせと fold 条件に対応する実行時メタデータを保持します。

#### パラメータ

- **`fold`** (`Any`): 準備済み分割情報に保存された fold 識別子。
- **`comb_index`** (`int`): 実験計画における入力組み合わせの0始まりの位置。
- **`comb_name`** (`str`): ファイルシステムで安全に使用できる入力組み合わせ名。
- **`input_comb`** (`list[Any]`): この条件で使用する入力キー。
- **`target_keys`** (`list[Any]`): この条件に含めるターゲットキー。
- **`split_indexes`** (`dict[str, list[int]]`): 各分割に割り当てられた元の ``data_dicts`` インデックス。
- **`prepared_split_indexes`** (`dict[str, list[int]]`): ``collected_data`` の参照に使用する対応位置。
- **`ref_value_splits`** (`dict[str, list[Any]]`): 話者やグループ識別子などの分割参照値。
- **`standardization_info`** (`dict`): 標準化する各入力に使用する平均、標準偏差、適用範囲。
- **`info_dict`** (`dict`): データ準備中に生成された共有情報。
- **`exp_config`** (`ExperimentConfig`): 実験設定。
- **`data_prep_config`** (`Any`): データ準備設定。
- **`output_dir`** (`str`): この条件に推奨される出力ディレクトリ。
- **`seed`** (`int`): 基準となる実験シード。
- **`user_extras`** (`dict`): 追加のユーザー定義実行時情報。

#### 宣言フィールド

| フィールド | 型 | デフォルト |
| --- | --- | --- |
| `fold` | `Any` | `_required_` |
| `comb_index` | `int` | `_required_` |
| `comb_name` | `str` | `_required_` |
| `input_comb` | `List[Any]` | `_required_` |
| `target_keys` | `List[Any]` | `_required_` |
| `split_indexes` | `Dict[str, List[int]]` | `_required_` |
| `prepared_split_indexes` | `Dict[str, List[int]]` | `_required_` |
| `ref_value_splits` | `Dict[str, List[Any]]` | `_required_` |
| `standardization_info` | `Dict[Any, Dict[str, Any]]` | `_required_` |
| `info_dict` | `Dict[str, Any]` | `_required_` |
| `exp_config` | `ExperimentConfig` | `_required_` |
| `data_prep_config` | `Any` | `_required_` |
| `output_dir` | `str` | `_required_` |
| `seed` | `int` | `_required_` |
| `user_extras` | `Dict[str, Any]` | `field(default_factory=dict)` |

#### 公開メソッド

##### `ExperimentContext.as_dict`

```python
def ExperimentContext.as_dict(self) -> Dict[str, Any]
```

コンテキストの浅い辞書表現を返します。

###### 戻り値

dict
    フィールド名をキーとするコンテキストフィールド。

### `ExperimentUnit`

```python
class ExperimentUnit
```

1つの実行可能な実験条件のデータとコンテキストを表します。

#### パラメータ

- **`data`** (`dict[str, Any]`): 設定されたデータレベルの学習・検証・テストデータ。
- **`context`** (`RunContext`): この条件に関連付けられた実行時情報。

#### 宣言フィールド

| フィールド | 型 | デフォルト |
| --- | --- | --- |
| `data` | `Dict[str, Any]` | `_required_` |
| `context` | `ExperimentContext` | `_required_` |

#### 公開メソッド

##### `ExperimentUnit.input_comb`

```python
def ExperimentUnit.input_comb(self) -> List[Any]
```

ユニットの入力キー組み合わせを返します。

##### `ExperimentUnit.fold`

```python
def ExperimentUnit.fold(self) -> Any
```

ユニットの fold 識別子を返します。

##### `ExperimentUnit.info_dict`

```python
def ExperimentUnit.info_dict(self) -> Dict[str, Any]
```

互換性維持用の共有データ準備情報を返します。

##### `ExperimentUnit.as_dict`

```python
def ExperimentUnit.as_dict(self) -> Dict[str, Any]
```

実験ユニットを辞書として返します。

###### 戻り値

dict
    ``data`` と ``context`` を含む辞書。

### `ExperimentManager`

```python
class ExperimentManager
```

共有実験状態を準備し、実験ユニットを反復処理します。

管理クラスは再反復可能です。``iter(manager)`` を呼び出すたびに、
すべての入力組み合わせと fold 条件を走査する新しいジェネレータが作成されます。

#### パラメータ

- **`exp_config`** (`ExperimentConfig`): 実験レベルの設定。
- **`data_dicts`** (`list[dict], optional`): 元サンプルの情報とデータ。``exp_config.load_prepared_data`` が ``False`` の場合は必須です。
- **`data_prep_config`** (`Any, optional`): ``data_prep.DataPreparator`` に渡す設定。準備済みデータを読み込まない場合は必須です。
- **`user_extras`** (`dict, optional`): 各 :class:`RunContext` へコピーする追加情報。

#### コンストラクタとプロトコルメソッド

##### `ExperimentManager.__init__`

```python
def ExperimentManager.__init__(
    self,
    exp_config: ExperimentConfig,
    data_dicts: Optional[List[Dict[str, Any]]]=None,
    data_prep_config: Any=None,
    user_extras: Optional[Dict[str, Any]]=None,
) -> None
```

データの読み込みや準備をまだ行わずに管理クラスを初期化します。

##### `ExperimentManager.__iter__`

```python
def ExperimentManager.__iter__(self) -> Iterator[ExperimentUnit]
```

設定済みのすべての実験ユニットを走査する新しいイテレータを返します。

#### 公開メソッド

##### `ExperimentManager.setup`

```python
def ExperimentManager.setup(self) -> 'ExperimentManager'
```

共有データを読み込むか準備し、実験設定を保存します。

###### 戻り値

ExperimentManager
    初期化済みで反復可能な管理クラス自身。

##### `ExperimentManager.init_random_seed`

```python
def ExperimentManager.init_random_seed(
    self,
    random_seed: int,
) -> None
```

設定された乱数系にシードを設定します。

###### パラメータ

- **`random_seed`** (`int`): 設定されたシステムへ適用するシード。

##### `ExperimentManager.get_prepared_data`

```python
def ExperimentManager.get_prepared_data(self) -> tuple[Dict[str, Any], Dict[str, Any], Any]
```

準備済みデータを読み込むか、データ準備パイプラインを実行します。

###### 戻り値

collected_data : dict
    モダリティとターゲットをキーとする準備済み配列またはリスト。
info_dict : dict
    分割、ターゲット、形状、インデックスマッピングの情報。
data_prep_config : Any
    データ準備に使用した設定。

##### `ExperimentManager.get_result`

```python
def ExperimentManager.get_result(
    self,
    pred: Any,
    true: Any,
    task_type: TaskType='c',
) -> Dict[str, Any]
```

予測結果を後処理し、タスク指標を計算します。

###### パラメータ

- **`pred, true`** (`Any`): 予測配列とターゲット配列、または対応するテンソル。
- **`task_type`** (`{"c", "r"}`): 分類または回帰のタスク種別。

###### 戻り値

dict
    指標辞書。

##### `ExperimentManager.torch_postprocess`

```python
def ExperimentManager.torch_postprocess(
    self,
    tensor: Any,
    *,
    task_type: TaskType='c',
)
```

テンソル後処理用のラッパー。テンソルまたは配列状の値を NumPy 評価配列へ変換します。

###### パラメータ

- **`tensor`** (`Any`): PyTorch テンソル、NumPy 配列、または配列状の値。
- **`task_type`** (`{"c", "r"}`): 分類または回帰のタスク種別。

###### 戻り値

dict
    指標辞書。

##### `ExperimentManager.compute_result`

```python
def ExperimentManager.compute_result(
    pred: Any,
    true: Any,
    task_type: TaskType='c',
) -> Dict[str, Any]
```

分類または回帰の指標を計算します。

##### `ExperimentManager.compute_average_result`

```python
def ExperimentManager.compute_average_result(
    result_dicts: Sequence[Mapping[str, Any]],
    target_key: Any,
    metric_key: str,
) -> Any
```

複数の結果辞書に含まれるネストしたターゲット指標を平均します。

##### `ExperimentManager.save_exp_config`

```python
def ExperimentManager.save_exp_config(self) -> None
```

実験設定を ``store_dir`` 配下へ保存します。

##### `ExperimentManager.save_result`

```python
def ExperimentManager.save_result(
    self,
    result: Any,
    context: Optional[ExperimentContext]=None,
    file_name: str='ExpResult.pkl',
) -> str
```

結果を全体の保存先、または1回の実行の出力ディレクトリへ保存します。

###### パラメータ

- **`result`** (`Any`): pickle でシリアライズ可能な結果オブジェクト。
- **`context`** (`RunContext, optional`): 指定した場合は ``context.output_dir`` 配下へ保存し、それ以外の場合は管理クラスの ``store_dir`` 配下へ保存します。
- **`file_name`** (`str`): 出力ファイル名。

###### 戻り値

str
    保存先ファイルパス。

<details>
<summary><strong>内部メソッド（1）</strong></summary>

##### `ExperimentManager._validate_prepared_data`

```python
def ExperimentManager._validate_prepared_data(self) -> None
```

実験計画に必要なキーと分割メタデータを検証します。

</details>

### `ExperimentResultLoader`

```python
class ExperimentResultLoader
```

保存済みの実験設定、データ準備設定、結果を読み込みます。

#### パラメータ

- **`result_dir`** (`str`): 保存済みファイルを含むディレクトリ。
- **`exp_config_file, data_prep_config_file, result_file`** (`str`): ``result_dir`` からの相対ファイル名。

#### コンストラクタとプロトコルメソッド

##### `ExperimentResultLoader.__init__`

```python
def ExperimentResultLoader.__init__(
    self,
    result_dir: str,
    exp_config_file: str='ExpConfig.json',
    data_prep_config_file: str='DataPrepConfig.json',
    result_file: str='ExpResult.pkl',
) -> None
```

要求されたすべての実験成果物を読み込みます。

### 公開関数

#### `generate_key_combs`

```python
def generate_key_combs(keys: Sequence[Any]) -> List[List[Any]]
```

指定されたキーの空でないすべての組み合わせを生成します。

##### パラメータ

- **`keys`** (`sequence`): 希望する組み合わせ順に並べた入力キー。

##### 戻り値

list[list]
    最初に組み合わせサイズ、次に入力順で並べた組み合わせ。

#### `load_prepared_data`

```python
def load_prepared_data(data_dir: str='./DataExperiment') -> tuple[Dict[str, Any], Dict[str, Any], Any]
```

データ、準備情報、準備設定を読み込みます。

##### パラメータ

- **`data_dir`** (`str`): ``data_prep.save_data`` により作成されたディレクトリ。

##### 戻り値

collected_data, info_dict, data_prep_config : tuple
    ``data_prep.load_data`` で読み込んだオブジェクト。

##### 例外

FileNotFoundError
    ディレクトリまたは必要なファイルが存在しない場合。

#### `iter_experiment_units`

```python
def iter_experiment_units(
    collected_data: Dict[str, Any],
    info_dict: Dict[str, Any],
    exp_config: ExperimentConfig,
    data_prep_config: Any,
    user_extras: Optional[Dict[str, Any]]=None,
) -> Iterator[ExperimentUnit]
```

入力組み合わせと fold ごとに1つの :class:`ExperimentUnit` を生成します。

``standardize_scope='split'`` の標準化では、現在の fold の学習データから
統計量を計算し、同じ統計量を学習・検証・
テストデータへ適用します。これにより検証・テストデータのリークを防ぎます。``'all'`` はすべての準備済み
サンプルを使用するため、意図を理解したうえで選択してください。

##### パラメータ

- **`collected_data`** (`dict`): 準備済みのモダリティ、ターゲット、インデックスデータ。
- **`info_dict`** (`dict`): 分割 fold とインデックスマッピングを含むデータ準備情報。
- **`exp_config`** (`ExperimentConfig`): 実験設定。
- **`data_prep_config`** (`Any`): ``key2config`` を含むデータ準備設定。
- **`user_extras`** (`dict, optional`): 各実行コンテキストへコピーする追加情報。

##### yield 値

ExperimentUnit
    1つの実験条件に対応するデータと実行時情報。

#### `make_data_generator`

```python
def make_data_generator(
    collected_data: Dict[str, Any],
    info_dict: Dict[str, Any],
    exp_config: ExperimentConfig,
    data_prep_config: Any,
    data_level: Optional[DataLevel]=None,
    data_representation: Optional[DataRepresentation]=None,
) -> Iterator[ExperimentUnit]
```

実験ユニットのジェネレータを作成します。

この互換関数は処理を :func:`iter_experiment_units` に委譲します。
オプションのデータレベル引数は設定の浅いコピーに適用されるため、
呼び出し側の設定は変更されません。

#### `make_data_geneartor`

```python
def make_data_geneartor(
    *args: Any,
    **kwargs: Any,
) -> Iterator[ExperimentUnit]
```

:func:`make_data_generator` の非推奨の誤記エイリアス。

#### `collect_combination_data`

```python
def collect_combination_data(
    collected_data: Mapping[Any, Any],
    keys: Sequence[Any],
    index_list: Sequence[int],
) -> Dict[Any, Any]
```

キーのグループについて、指定された準備済みデータ位置を選択します。

##### パラメータ

- **`collected_data`** (`mapping`): 入力キー、ターゲットキー、またはインデックスキーを持つ準備済みデータ。
- **`keys`** (`sequence`): 結果に含めるキー。
- **`index_list`** (`sequence[int]`): 準備済みデータ配列・リスト内の位置。

##### 戻り値

dict
    ``keys`` の各キーを持つ選択済みデータ。

#### `perform_zscore`

```python
def perform_zscore(
    data_array: Any,
    axis: Union[int, tuple[int, ...]]=0,
) -> np.ndarray
```

分散ゼロへの保護を適用して数値データを標準化します。

##### パラメータ

- **`data_array`** (`array-like`): 数値データ。
- **`axis`** (`int or tuple[int, ...]`): 平均と標準偏差を計算する軸。

##### 戻り値

numpy.ndarray
    標準化後のデータ。

#### `get_input_target_shapes`

```python
def get_input_target_shapes(
    datasets: Mapping[str, Any],
    info_dict: Mapping[str, Any],
) -> Dict[Any, tuple[int, ...]]
```

データセットで利用できる、設定済み入力形状とターゲット形状を収集します。

##### パラメータ

- **`datasets`** (`mapping`): ``"train"`` を含む分割データセット、または生の分割辞書。
- **`info_dict`** (`mapping`): ``input_shapes`` と ``target_info`` を含むデータ準備情報。

##### 戻り値

dict
    入力名またはターゲット名をキーとする形状情報。

#### `make_dataset`

```python
def make_dataset(
    single_condition_data: Dict[Any, Any],
    data_representation: DataRepresentation='original',
    use_gpu: bool=False,
) -> TorchDataset
```

辞書ベースの PyTorch データセットを作成します。

##### パラメータ

- **`single_condition_data`** (`dict`): 1つの学習・検証・テスト分割に対応するデータ。
- **`data_representation`** (`{"original", "pt"}`): 値を変更せず保持するか、対応する値をテンソルへ変換します。
- **`use_gpu`** (`bool`): 変換後のテンソルを CUDA へ配置します。データセットを DataLoader でラップする場合は CPU テンソルを推奨します。

##### 戻り値

TorchDataset
    作成されたデータセット。

#### `convert_single_data_to_tensor`

```python
def convert_single_data_to_tensor(
    single_condition_data: Mapping[Any, Any],
    *,
    exclude_keys: Optional[Iterable[Any]]=None,
    use_gpu: bool=False,
) -> Dict[Any, Any]
```

対応する分割データを PyTorch テンソルへ変換します。

可能な場合、インデックスメタデータは ``torch.long`` に変換されます。未対応の
Python オブジェクトはそのまま保持されます。

##### パラメータ

- **`single_condition_data`** (`mapping`): モダリティキー、ターゲットキー、またはメタデータキーを持つ分割データ。
- **`exclude_keys`** (`iterable, optional`): 変換せずにコピーするキー。
- **`use_gpu`** (`bool`): CUDA が利用可能な場合、変換後のテンソルを CUDA へ移動します。

##### 戻り値

dict
    テンソル変換後のデータ。

#### `make_dataloader`

```python
def make_dataloader(
    dataset: TorchDataset,
    batch_size: int,
    shuffle: bool,
) -> Any
```

空でないデータセットから PyTorch DataLoader を作成します。

##### パラメータ

- **`dataset`** (`TorchDataset`): バッチ化するデータセット。
- **`batch_size`** (`int`): 1バッチ当たりの正のサンプル数。
- **`shuffle`** (`bool`): サンプル順をシャッフルするかどうか。

##### 戻り値

torch.utils.data.DataLoader
    作成された DataLoader。

#### `torch_postprocess`

```python
def torch_postprocess(
    tensor: Any,
    *,
    mode: Literal['raw', 'argmax']='raw',
    use_gpu: Optional[bool]=None,
) -> np.ndarray
```

テンソルまたは配列状の値を NumPy 評価配列へ変換します。

##### パラメータ

- **`tensor`** (`Any`): PyTorch テンソル、NumPy 配列、または配列状の値。
- **`mode`** (`{"raw", "argmax"}`): 生の値を返すか、最終軸の argmax を先に計算します。
- **`use_gpu`** (`bool, optional`): 非推奨の互換性維持用引数。デバイス処理はテンソル自体から推定されます。

##### 戻り値

numpy.ndarray
    計算グラフから切り離された CPU 配列。

#### `load_exp_config`

```python
def load_exp_config(file: str='./ExpConfig.json') -> ExperimentConfig
```

保存済みの実験設定 JSON ファイルを読み込みます。

#### `make_config_from_json`

```python
def make_config_from_json(config_json: Sequence[Any]) -> ExperimentConfig
```

保存済み JSON 内容から実験設定を構築します。

##### パラメータ

- **`config_json`** (`sequence`): クラス名とコンストラクタフィールドを含む2要素のシーケンス。

##### 戻り値

ExperimentConfig
    再構築された設定オブジェクト。

#### `compute_cls_metrics`

```python
def compute_cls_metrics(
    pred_list: Any,
    true_list: Any,
) -> Dict[str, Any]
```

一般的な単一ラベル分類指標を計算します。

Pearson 相関係数を定義できない場合、たとえば
定数配列や要素数が2未満の配列では ``nan`` を返します。

#### `compute_regression_metrics`

```python
def compute_regression_metrics(
    pred_list: Any,
    true_list: Any,
) -> Dict[str, Any]
```

回帰用の MAE、MSE、RMSE、Pearson 相関係数を計算します。

#### `compute_target_key_average_metric`

```python
def compute_target_key_average_metric(
    result_dicts: Sequence[Mapping[Any, Any]],
    target_key: Any,
    metric_key: str,
) -> Any
```

1つのターゲットキーの下にネストされた指標を平均します。

#### `compute_average_metric`

```python
def compute_average_metric(
    result_dicts: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> Any
```

複数の結果辞書に含まれる最上位の指標を平均します。

#### `change_average_summary_form`

```python
def change_average_summary_form(
    average_summary: Mapping[str, Any],
    form: Literal['lists']='lists',
) -> Dict[str, Any]
```

平均値の要約を、指定された分析しやすい形式へ変換します。

#### `change_average_summary_lists`

```python
def change_average_summary_lists(average_summary: Mapping[str, Any]) -> Dict[str, Any]
```

組み合わせの要約を、対応する組み合わせリストと指標リストへ変換します。

<details>
<summary><strong>内部モジュールヘルパー（17）</strong></summary>

これらの関数は公開 API を支える内部関数です。ソースに docstring が含まれるため本書にも掲載していますが、ユーザーコードが直接依存する互換性インターフェースとしては推奨されません。

#### `_make_dataset_tensor`

```python
def _make_dataset_tensor(
    single_condition_data: Dict[Any, Any],
    use_gpu: bool=False,
) -> TorchDataset
```

分割データをテンソルへ変換し、:class:`TorchDataset` でラップします。

#### `_get_fold_average_summary`

```python
def _get_fold_average_summary(
    result_dir: Optional[str]=None,
    result_root: Optional[str]=None,
    experiment_name: Optional[str]=None,
    combs: Optional[Union[str, List[str]]]=None,
    folds: Optional[int]=None,
) -> Dict[str, Any]
```

fold の要約を読み込み、組み合わせごとの平均を計算します。

``result_dir``、または ``result_root`` と ``experiment_name`` の両方を
指定する必要があります。

#### `_unify_to_list`

```python
def _unify_to_list(value: Any) -> List[Any]
```

スカラーまたはイテラブルな設定入力を通常のリストへ変換します。

#### `_unique_in_order`

```python
def _unique_in_order(values: Iterable[Any]) -> List[Any]
```

初出順を保ちながら一意な値を返します。

#### `_normalize_fold_mapping`

```python
def _normalize_fold_mapping(folds: Any) -> Dict[Any, Dict[str, List[int]]]
```

リスト形式またはマッピング形式の分割 fold を順序付き辞書へ正規化します。

#### `_normalize_optional_fold_mapping`

```python
def _normalize_optional_fold_mapping(
    folds: Any,
    fold_keys: Iterable[Any],
) -> Dict[Any, Dict[str, List[Any]]]
```

オプションの参照値分割情報を正規化します。

#### `_translate_split_indexes`

```python
def _translate_split_indexes(
    split_indexes: Mapping[str, Sequence[int]],
    info_dict: Mapping[str, Any],
    collected_data: Mapping[Any, Any],
) -> Dict[str, List[int]]
```

元サンプルインデックスを準備済みデータ上の位置へ変換します。

#### `_infer_sample_count`

```python
def _infer_sample_count(collected_data: Mapping[Any, Any]) -> int
```

共有される第1次元のサンプル数を推定・検証します。

#### `_standardize_split_data`

```python
def _standardize_split_data(
    split_data: Dict[str, Dict[Any, Any]],
    collected_data: Mapping[Any, Any],
    input_comb: Sequence[Any],
    data_prep_config: Any,
) -> tuple[Dict[str, Dict[Any, Any]], Dict[Any, Dict[str, Any]]]
```

共有データを変更せずに、設定済みの数値入力を標準化します。

#### `_apply_standardization`

```python
def _apply_standardization(
    array: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray
```

分散がゼロの軸をゼロへ写像しながら、z-score 正規化を適用します。

#### `_as_numeric_array`

```python
def _as_numeric_array(
    value: Any,
    key: Any,
) -> np.ndarray
```

値を object 型ではない数値 NumPy 配列へ変換します。

#### `_convert_data_level`

```python
def _convert_data_level(
    split_data: Dict[str, Dict[Any, Any]],
    exp_config: ExperimentConfig,
) -> Dict[str, Any]
```

生の分割辞書を指定された実験データレベルへ変換します。

#### `_make_comb_name`

```python
def _make_comb_name(
    input_comb: Sequence[Any],
    abbreviations: Mapping[Any, Any],
) -> str
```

入力組み合わせ用に、ファイルシステムで安全なディレクトリ名を作成します。

#### `_sanitize_path_component`

```python
def _sanitize_path_component(value: str) -> str
```

パスに使用できない文字をアンダースコアへ置換します。

#### `_require_torch`

```python
def _require_torch(caller: str) -> None
```

PyTorch ヘルパーが利用できない場合、内容の分かるエラーを送出します。

#### `_needs_argmax`

```python
def _needs_argmax(prediction: Any) -> bool
```

分類予測にクラススコア軸があるかどうかを返します。

#### `_safe_pearson`

```python
def _safe_pearson(
    first: np.ndarray,
    second: np.ndarray,
) -> float
```

Pearson 相関係数を計算し、定義できない場合は ``nan`` を返します。

</details>

---

## `flexmm.info_utils`

[GitHub でソースを表示](https://github.com/Xia-code/flexmm/blob/main/flexmm/info_utils.py)

サンプル単位の情報レコードを検索・整理するためのユーティリティです。

本モジュールの関数は、``data_dicts`` が辞書のリストであることを想定しています。
各辞書は、1つのサンプル、発話、フレーム、またはその他の時系列項目を表します。
辞書フィールドには、識別子、グループ変数、ターゲット値、または
その他の軽量なサンプル情報を格納できます。

### API 概要

#### 公開関数

| 関数 | 概要 |
| --- | --- |
| `get_ref_value2indexes()` | 各参照値を対応するサンプルインデックスへ対応付けます。 |
| `get_ref_value_list()` | 処理した各サンプルから参照値を1つ収集します。 |
| `get_ref_value2another()` | 各参照値を別のサンプルフィールドの値へ対応付けます。 |
| `get_turn2ref_value_and_indexes()` | 同じ参照値を持つ連続サンプルをターンとしてグループ化します。 |
| `get_ref_value2turn_indexes()` | 各参照値を、その各ターンのサンプルインデックスリストへ対応付けます。 |
| `get_ref_value2turns()` | 各参照値を、0始まりのターン ID へ対応付けます。 |
| `get_ref_value2indexes_in_turns()` | 各参照値を、そのターンを通じて収集したインデックスへ対応付けます。 |
| `get_ref_value2adjacent_ref_value()` | サンプルインデックスまたはターン単位で、異なる参照値間の隣接関係を要約します。 |
| `get_interval_split_indexes()` | シーケンスを、ほぼ同じ大きさの時系列区間へ分割します。 |

### 公開関数

#### `get_ref_value2indexes`

```python
def get_ref_value2indexes(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> Dict[Any, List[int]]
```

各参照値を対応するサンプルインデックスへ対応付けます。

##### パラメータ

- **`data_dicts`** (`list of dict`): サンプル単位の情報レコード。処理する各辞書には ``ref_key`` が必要です。
- **`ref_key`** (`Any, optional`): 話者、参加者、セッション、グループ識別子など、参照グループを定義する値を持つ辞書キー。
- **`used_indexes`** (`list of int or None, optional`): 処理するサンプルインデックスのサブセット。``None`` の場合はすべてのサンプルを使用します。

##### 戻り値

dict
    ``ref_key`` の各値から、その値を含むサンプルインデックスへのマッピング。
    インデックス順は ``used_indexes`` に従います。

##### 例外

KeyError
    処理する辞書に ``ref_key`` が含まれない場合。
TypeError
    参照値がハッシュ可能ではなく、そのため
    辞書キー。

#### `get_ref_value_list`

```python
def get_ref_value_list(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> List[Any]
```

処理した各サンプルから参照値を1つ収集します。

##### パラメータ

- **`data_dicts`** (`list of dict`): サンプル単位の情報レコード。処理する各辞書には ``ref_key`` が必要です。
- **`ref_key`** (`Any, optional`): 各サンプルから収集する値を表す辞書キー。
- **`used_indexes`** (`list of int or None, optional`): 処理するサンプルインデックスのサブセット。``None`` の場合はすべてのサンプルを使用します。

##### 戻り値

list
    サンプル順に並んだ ``ref_key`` の値。重複値は保持されます。

##### 例外

KeyError
    処理する辞書に ``ref_key`` が含まれない場合。

#### `get_ref_value2another`

```python
def get_ref_value2another(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    another_key: Any='target',
    used_indexes: Union[List[int], None]=None,
    *,
    unique_values: bool=True,
) -> Dict[Any, List[Any]]
```

各参照値を別のサンプルフィールドの値へ対応付けます。

この関数は、2つの情報フィールド間の関係を確認する場合に便利です。
たとえば、各話者をそのターゲット値へ、各グループを
参加者識別子へ対応付けられます。

##### パラメータ

- **`data_dicts`** (`list of dict`): サンプル単位の情報レコード。処理する各辞書には ``ref_key`` と ``another_key`` の両方が必要です。
- **`ref_key`** (`Any, optional`): 返されるマッピングのキーとなる値を持つ辞書キー。
- **`another_key`** (`Any, optional`): 各参照値について収集する値を持つ辞書キー。
- **`used_indexes`** (`list of int or None, optional`): 処理するサンプルインデックスのサブセット。``None`` の場合はすべてのサンプルを使用します。
- **`unique_values`** (`bool, optional`): ``True`` の場合、各参照値について同等の重複値は1回だけ保存します。``False`` の場合、重複を含むすべての値を保持します。どちらのモードも配列状の値に対応します。

##### 戻り値

dict
    ``ref_key`` の各値から ``another_key`` の値へのマッピング。

##### 例外

KeyError
    処理する辞書に ``ref_key`` または ``another_key`` がない場合。

#### `get_turn2ref_value_and_indexes`

```python
def get_turn2ref_value_and_indexes(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> Dict[int, Dict[str, Any]]
```

同じ参照値を持つ連続サンプルをターンとしてグループ化します。

``ref_key`` の値が変化した場合、または次に処理する
サンプルインデックスが直前のインデックスと数値的に連続していない場合に、新しいターンを開始します。
``used_indexes`` で指定された順序は維持されます。

##### パラメータ

- **`data_dicts`** (`list of dict`): 1つの対話、会話、イベント、またはその他のシーケンスを表す順序付きサンプル単位レコード。
- **`ref_key`** (`Any, optional`): 連続サンプルが同じターンに属するかを判定する辞書キー。通常は話者または参加者の識別子です。
- **`used_indexes`** (`list of int or None, optional`): 処理するサンプルインデックスの順序付きサブセット。``None`` の場合はすべてのサンプルを使用します。

##### 戻り値

dict
    0始まりのターン ID から、2つのフィールドを持つ辞書へのマッピング：
    ``"ref_value"`` にはターンの参照値を、``"indexes"`` には
    そのターン内のサンプルインデックスを格納します。

##### 例外

KeyError
    処理する辞書に ``ref_key`` が含まれない場合。

#### `get_ref_value2turn_indexes`

```python
def get_ref_value2turn_indexes(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> Dict[Any, List[List[int]]]
```

各参照値を、その各ターンのサンプルインデックスリストへ対応付けます。

##### パラメータ

- **`data_dicts`** (`list of dict`): シーケンスを表す順序付きサンプル単位レコード。
- **`ref_key`** (`Any, optional`): 各ターンの所有者を識別する辞書キー。
- **`used_indexes`** (`list of int or None, optional`): 処理するサンプルインデックスの順序付きサブセット。``None`` の場合はすべてのサンプルを使用します。

##### 戻り値

dict
    各参照値からターンのリストへのマッピング。各ターンは
    そのサンプルインデックスリストで表されます。

#### `get_ref_value2turns`

```python
def get_ref_value2turns(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> Dict[Any, List[int]]
```

各参照値を、0始まりのターン ID へ対応付けます。

##### パラメータ

- **`data_dicts`** (`list of dict`): シーケンスを表す順序付きサンプル単位レコード。
- **`ref_key`** (`Any, optional`): 各ターンの所有者を識別する辞書キー。
- **`used_indexes`** (`list of int or None, optional`): 処理するサンプルインデックスの順序付きサブセット。``None`` の場合はすべてのサンプルを使用します。

##### 戻り値

dict
    各参照値から、そのターン ID へのマッピング。

#### `get_ref_value2indexes_in_turns`

```python
def get_ref_value2indexes_in_turns(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> Dict[Any, List[int]]
```

各参照値を、そのターンを通じて収集したインデックスへ対応付けます。

これは :func:`get_ref_value2turn_indexes` を平坦化した対応関数です。
戻り値にはターン境界を保持しません。

##### パラメータ

- **`data_dicts`** (`list of dict`): シーケンスを表す順序付きサンプル単位レコード。
- **`ref_key`** (`Any, optional`): 各ターンの所有者を識別する辞書キー。
- **`used_indexes`** (`list of int or None, optional`): 処理するサンプルインデックスの順序付きサブセット。``None`` の場合はすべてのサンプルを使用します。

##### 戻り値

dict
    各参照値から、そのすべてのサンプルインデックスへのマッピング。
    ターンの出現順に並びます。

#### `get_ref_value2adjacent_ref_value`

```python
def get_ref_value2adjacent_ref_value(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    prev_or_following: Literal['prev', 'following']='prev',
    adjacent_by: Literal['index', 'turn']='index',
    used_indexes: Union[List[int], None]=None,
) -> Dict[Any, Dict[Any, List[int]]]
```

サンプルインデックスまたはターン単位で、異なる参照値間の隣接関係を要約します。

最上位キーは対象項目の参照値です。各
ネストされたキーは隣接項目の参照値です。返される整数は、
``adjacent_by="index"`` の場合は現在のサンプルを、
``adjacent_by="turn"`` の場合は現在のターンを表します。

参照値が異なる隣接項目だけを含めます。指定された
サブセットに欠落区間がある場合、その
非連続区間をまたいだ隣接関係は作成しません。

##### パラメータ

- **`data_dicts`** (`list of dict`): シーケンスを表す順序付きサンプル単位レコード。
- **`ref_key`** (`Any, optional`): 隣接項目間で比較する値を持つ辞書キー。
- **`prev_or_following`** (`{"prev", "following"}, optional`): 現在のサンプルまたはターンを基準とした隣接方向。
- **`adjacent_by`** (`{"index", "turn"}, optional`): 隣接関係を定義する単位。``"index"`` は隣接サンプルインデックスを比較し、``"turn"`` は隣接ターンを比較します。
- **`used_indexes`** (`list of int or None, optional`): 処理するサンプルインデックスの順序付きサブセット。``None`` の場合はすべてのサンプルを使用します。

##### 戻り値

dict
    ネストしたマッピング：
    ``current_ref_value -> adjacent_ref_value -> current_indexes_or_turns``。

##### 例外

ValueError
    ``prev_or_following`` または ``adjacent_by`` が無効な場合。

#### `get_interval_split_indexes`

```python
def get_interval_split_indexes(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    interval_num: int=3,
    interval_split_by: Literal['index', 'turn']='index',
    used_indexes: Union[List[int], None]=None,
) -> Dict[int, List[int]]
```

シーケンスを、ほぼ同じ大きさの時系列区間へ分割します。

インデックス単位で分割する場合、区間にはサンプルインデックスを直接格納します。
ターン単位で分割する場合は、完全なターンを区間へ割り当てた後、
サンプルインデックスへ展開するため、1つのターンが
2つの区間に分割されることはありません。

##### パラメータ

- **`data_dicts`** (`list of dict`): シーケンスを表す順序付きサンプル単位レコード。
- **`ref_key`** (`Any, optional`): ``interval_split_by="turn"`` の場合にターンを識別する辞書キー。
- **`interval_num`** (`int, optional`): 作成する区間数。返される辞書には常にこの数の区間キーが含まれますが、サンプル数またはターン数が区間数より少ない場合、一部の区間は空になることがあります。
- **`interval_split_by`** (`{"index", "turn"}, optional`): 区間境界をサンプルインデックスまたは完全なターンのどちらに基づいて決めるか。
- **`used_indexes`** (`list of int or None, optional`): 処理するサンプルインデックスの順序付きサブセット。``None`` の場合はすべてのサンプルを使用します。

##### 戻り値

dict
    0始まりの区間 ID からサンプルインデックスリストへのマッピング。ターン展開前の
    各区間サイズの差は最大1単位です。

##### 例外

ValueError
    ``interval_num`` が正でない、または ``interval_split_by`` が無効な場合。

<details>
<summary><strong>内部モジュールヘルパー（3）</strong></summary>

これらの関数は公開 API を支える内部関数です。ソースに docstring が含まれるため本書にも掲載していますが、ユーザーコードが直接依存する互換性インターフェースとしては推奨されません。

#### `_resolve_used_indexes`

```python
def _resolve_used_indexes(
    data_dicts: List[Dict],
    used_indexes: Union[List[int], None],
) -> List[int]
```

処理するサンプルインデックスを解決・検証します。

##### パラメータ

- **`data_dicts`** (`list of dict`): サンプル単位の情報レコード。
- **`used_indexes`** (`list of int or None`): 処理するインデックス。``None`` の場合は、すべてのインデックスを元の順序で返します。

##### 戻り値

list of int
    検証済みの処理対象インデックス。

##### 例外

TypeError
    インデックスが整数ではない場合。
IndexError
    インデックスが ``data_dicts`` の有効範囲外にある場合。

#### `_values_equal`

```python
def _values_equal(
    left: Any,
    right: Any,
) -> bool
```

スカラーと配列状オブジェクトの両方に対応して、2つの値を比較します。

##### パラメータ

- **`left`** (`Any`): 比較する1つ目の値。
- **`right`** (`Any`): 比較する2つ目の値。

##### 戻り値

bool
    値が同等の場合は ``True``。配列状の比較結果は
    すべての要素について集約されます。

#### `_contains_equivalent`

```python
def _contains_equivalent(
    values: List[Any],
    candidate: Any,
) -> bool
```

リストに候補と同等の値が含まれるかを確認します。

##### パラメータ

- **`values`** (`list`): 検索対象の既存値。
- **`candidate`** (`Any`): 候補値。スカラーまたは配列状オブジェクトを指定できます。

##### 戻り値

bool
    同等の値がすでに存在する場合は ``True``。

</details>

---
