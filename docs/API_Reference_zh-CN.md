# FlexMM API 参考文档

[English](API_Reference.md) | **简体中文** | [日本語](API_Reference_ja.md)

本文档根据 FlexMM 源代码中的 docstring 生成，采用适合 GitHub 阅读的格式，供需要查阅配置类、实验对象、数据准备辅助函数、评估函数和信息工具的用户使用。

- **包名：** `flexmm`
- **代码仓库：** https://github.com/Xia-code/flexmm
- **生成日期：** 2026-06-26
- **Python：** 3.9 或更高版本

> 本文优先列出公开类、方法和函数。
> 以下划线开头的名称属于实现细节，并收录在
> 可折叠区域中；其行为在不同版本之间更可能发生变化。

## 目录

1. [包概览](#包概览)
2. [`flexmm.data_prep`](#flexmmdata_prep)
3. [`flexmm.experiment`](#flexmmexperiment)
4. [`flexmm.info_utils`](#flexmminfo_utils)

## 包概览

FlexMM 提供三个主要模块，用于多模态数据准备和实验流程编排。

| 模块 | 主要职责 |
| --- | --- |
| `flexmm.data_prep` | 支持序列的多模态数据收集、目标处理、数据划分、重叠预防和序列化。 |
| `flexmm.experiment` | 实验配置、模态组合迭代、PyTorch 转换、指标计算、上下文管理和结果加载。 |
| `flexmm.info_utils` | 针对样本级元数据、轮次、参考值、相邻关系和区间的查询与分组操作。 |

### 常用导入方式

```python
from flexmm import data_prep, experiment, info_utils
from flexmm.data_prep import DataPrepConfig, DataPreparator, InputConfig
from flexmm.experiment import ExperimentConfig, ExperimentManager
```

### 包初始化文件的 docstring

```text
20250328
__init__ file of flexmm
```

---

## `flexmm.data_prep`

[在 GitHub 上查看源代码](https://github.com/Xia-code/flexmm/blob/main/flexmm/data_prep.py)

准备支持序列的多模态数据、目标值以及训练/验证/测试划分。

本模块定义配置对象、数据收集工具、目标值
统计与转换辅助函数、确定性数据划分策略、序列
重叠移除及序列化辅助函数。样本级描述数据应
以字典列表的形式提供，列表索引用于标识原始样本。

### API 概览

#### 类

| 类 | 说明 |
| --- | --- |
| [`BaseConfig`](#baseconfig) | 一个输入或目标数据键组的通用配置字段。 |
| [`ClassificationTargetConfig`](#classificationtargetconfig) | 配置一个或多个标量分类目标。 |
| [`RegressionTargetConfig`](#regressiontargetconfig) | 配置一个或多个回归目标以及可选的分层分箱。 |
| [`InputConfig`](#inputconfig) | 配置一个或多个模型输入数据键。 |
| [`DataPrepConfig`](#dataprepconfig) | 配置序列构造、目标处理、数据划分和持久化。 |
| [`DataPreparator`](#datapreparator) | 对样本级字典运行完整的数据准备流程。 |

#### 公开函数

| 函数 | 说明 |
| --- | --- |
| `pick_data_by_indexes()` | 按原始索引选择样本字典，并记录每个源索引。 |
| `gather_data_single_key()` | 收集一个数据键对应的所有序列窗口。 |
| `gather_data_by_indexes()` | 收集一个值序列，并应用常量填充或边缘填充。 |
| `shift_get_seq_indexes()` | 围绕过滤后的锚点索引构造带步长的上下文窗口。 |
| `get_strided_seq()` | 提取索引列表中某一位置周围的带步长窗口。 |
| `data_split_independent()` | 独立划分参考组，确保测试组不会出现在训练集或验证集中。 |
| `data_split_dependent()` | 在每个参考组内部划分样本，因此同一组可以出现在所有数据集中。 |
| `data_split_unconstrained()` | 划分样本索引，但不强制参考组相互独立。 |
| `remove_overlapped_seq_split()` | 移除序列内容与受保护划分发生重叠的低优先级锚点。 |
| `get_target_info_cls()` | 构建类别统计、类别 ID 映射和原始索引分组。 |
| `get_target_info_regression()` | 对标量回归目标进行分箱，并收集分箱统计和索引。 |
| `get_target2indexes()` | 将每个类别或回归分箱映射到其原始样本索引。 |
| `get_stratified_bin_info()` | 计算实际的回归分箱宽度和分箱数量。 |
| `generate_config_template()` | 为指定的数据键写入 JSON 配置模板。 |
| `load_config()` | 将 JSON 配置文件加载为配置对象。 |
| `make_config_from_json()` | 根据解析后的 JSON 数据重建嵌套配置对象。 |
| `save_data()` | 使用显式参数或配置选项保存准备后的数据和元数据。 |
| `load_data()` | 从目录中加载准备后的数据、元数据和配置。 |
| `get_target_list()` | 从所选原始样本索引中收集目标值。 |
| `get_scalar_target_list()` | 收集所选目标值，并将其规范化为 Python 标量。 |
| `convert_to_python_scalar()` | 将类标量的 Python、NumPy 或 PyTorch 值转换为 Python 标量。 |
| `calculate_miu_sigma()` | 计算所选样本按特征划分的均值和标准差。 |

### 类

### `BaseConfig`

```python
class BaseConfig
```

一个输入或目标数据键组的通用配置字段。

#### 说明

实例用于保存经过验证的设置或本模块使用的流程状态。

#### 声明字段

| 字段 | 类型 | 默认值 |
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
<summary><strong>内部方法（1）</strong></summary>

##### `BaseConfig.__post_init__`

```python
def BaseConfig.__post_init__(self)
```

规范化派生字段，并在初始化后验证配置。

</details>

### `ClassificationTargetConfig`

```python
class ClassificationTargetConfig(BaseConfig)
```

配置一个或多个标量分类目标。

#### 说明

实例用于保存经过验证的设置或本模块使用的流程状态。

#### 声明字段

| 字段 | 类型 | 默认值 |
| --- | --- | --- |
| `convert_target_to_id` | `bool` | `False` |

<details>
<summary><strong>内部方法（1）</strong></summary>

##### `ClassificationTargetConfig.__post_init__`

```python
def ClassificationTargetConfig.__post_init__(self)
```

规范化派生字段，并在初始化后验证配置。

</details>

### `RegressionTargetConfig`

```python
class RegressionTargetConfig(BaseConfig)
```

配置一个或多个回归目标以及可选的分层分箱。

#### 说明

实例用于保存经过验证的设置或本模块使用的流程状态。

#### 声明字段

| 字段 | 类型 | 默认值 |
| --- | --- | --- |
| `is_multi_dim` | `bool` | `False` |
| `convert_target_to_bin` | `bool` | `False` |
| `stratified_bin_size` | `Union[float, int, None]` | `None` |
| `stratified_bin_num` | `Optional[int]` | `10` |
| `bin_closed_side` | `Literal['upper', 'lower']` | `'lower'` |

<details>
<summary><strong>内部方法（1）</strong></summary>

##### `RegressionTargetConfig.__post_init__`

```python
def RegressionTargetConfig.__post_init__(self)
```

规范化派生字段，并在初始化后验证配置。

</details>

### `InputConfig`

```python
class InputConfig(BaseConfig)
```

配置一个或多个模型输入数据键。

#### 说明

实例用于保存经过验证的设置或本模块使用的流程状态。

#### 声明字段

| 字段 | 类型 | 默认值 |
| --- | --- | --- |
| `is_non_numeric` | `bool` | `False` |

<details>
<summary><strong>内部方法（1）</strong></summary>

##### `InputConfig.__post_init__`

```python
def InputConfig.__post_init__(self)
```

规范化派生字段，并在初始化后验证配置。

</details>

### `DataPrepConfig`

```python
class DataPrepConfig
```

配置序列构造、目标处理、数据划分和持久化。

#### 说明

实例用于保存经过验证的设置或本模块使用的流程状态。

#### 声明字段

| 字段 | 类型 | 默认值 |
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

#### 公开方法

##### `DataPrepConfig.assert_config`

```python
def DataPrepConfig.assert_config(self)
```

验证与数据划分有关的配置值，并解析兼容的默认设置。

##### `DataPrepConfig.to_json`

```python
def DataPrepConfig.to_json(self)
```

将嵌套配置对象转换为可进行 JSON 序列化的结构。

###### 返回值

list
    包含类名和 dataclass 字段的序列化配置条目。

<details>
<summary><strong>内部方法（3）</strong></summary>

##### `DataPrepConfig.__post_init__`

```python
def DataPrepConfig.__post_init__(self)
```

规范化派生字段，并在初始化后验证配置。

##### `DataPrepConfig._init_keys_and_check`

```python
def DataPrepConfig._init_keys_and_check(
    self,
    configs,
    config_name='input',
)
```

收集某一配置类别的键，并拒绝重复键。

###### 参数

- **`configs`** (`Any`): 要检查的数据配置对象。
- **`config_name`** (`Any`): 要收集的配置类别，通常为 ``"input"`` 或 ``"target"``。

##### `DataPrepConfig._make_seq_info`

```python
def DataPrepConfig._make_seq_info(self) -> (Dict, int, int)
```

构建每个键对应的序列窗口设置。

</details>

### `DataPreparator`

```python
class DataPreparator
```

对样本级字典运行完整的数据准备流程。

#### 说明

实例用于保存经过验证的设置或本模块使用的流程状态。

#### 构造函数和协议方法

##### `DataPreparator.__init__`

```python
def DataPreparator.__init__(
    self,
    data_dicts,
    data_prep_config,
    split_postprocess_fn=None,
)
```

初始化数据准备器，并生成与序列有关的索引元数据。

###### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`data_prep_config`** (`Any`): 与准备后数据一起保存的配置对象。
- **`split_postprocess_fn`** (`Any`): 可选的可调用对象，在内置处理后接收索引折叠和参考值折叠。

#### 公开方法

##### `DataPreparator.run`

```python
def DataPreparator.run(self)
```

执行序列收集、目标处理、数据划分和可选保存。

###### 返回值

tuple
    完整流程生成的 ``(collected_data, info_dict)``。

##### `DataPreparator.get_seq_indexes`

```python
def DataPreparator.get_seq_indexes(self)
```

为每个已配置的输入键和目标键构建序列窗口。

##### `DataPreparator.gather_data`

```python
def DataPreparator.gather_data(self)
```

将已配置的数据字段收集为对齐的数组或 Python 列表。

##### `DataPreparator.process_target`

```python
def DataPreparator.process_target(self)
```

创建目标统计，并应用请求的目标转换。

##### `DataPreparator.convert_target_to_scalar`

```python
def DataPreparator.convert_target_to_scalar(self)
```

创建用于统计和数据划分的标量目标列表。

##### `DataPreparator.make_target_info_dict`

```python
def DataPreparator.make_target_info_dict(self)
```

构建标签映射、回归分箱、统计信息和索引分组。

###### 返回值

dict
    以目标名称为键的目标元数据。

##### `DataPreparator.convert_target_form`

```python
def DataPreparator.convert_target_form(self)
```

将收集的目标转换为类别 ID 或回归分箱代表值。

##### `DataPreparator.split_data`

```python
def DataPreparator.split_data(self)
```

根据已配置的策略创建训练、验证和测试折叠。

##### `DataPreparator.post_split_process`

```python
def DataPreparator.post_split_process(self)
```

移除会导致禁止的跨划分重叠的序列锚点。

##### `DataPreparator.make_info_dict`

```python
def DataPreparator.make_info_dict(self)
```

汇总用于保存和下游实验的数据准备元数据。

##### `DataPreparator.get_zscore_miu_sigma`

```python
def DataPreparator.get_zscore_miu_sigma(self)
```

为已配置的输入键计算每个折叠的归一化统计量。

###### 返回值

list[dict]
    每个折叠和划分的归一化统计量。

##### `DataPreparator.get_input_shapes`

```python
def DataPreparator.get_input_shapes(self)
```

根据收集的数据推断模型输入形状。

###### 返回值

dict
    输入键到推断特征形状的映射。

##### `DataPreparator.save_data`

```python
def DataPreparator.save_data(
    self,
    save_dir=None,
    overwrite_data=None,
)
```

使用显式参数或配置选项保存准备后的数据和元数据。

###### 参数

- **`save_dir`** (`Any`): 目标目录。``None`` 表示使用配置的存储目录。
- **`overwrite_data`** (`Any`): 是否允许替换已有的序列化数据。

<details>
<summary><strong>内部方法（6）</strong></summary>

##### `DataPreparator._init_seq_indexes_info`

```python
def DataPreparator._init_seq_indexes_info(self)
```

初始化序列范围、使用的索引以及原始索引映射。

##### `DataPreparator._init_seq_ranges`

```python
def DataPreparator._init_seq_ranges(self)
```

根据分组或自定义边界创建左闭右开的序列范围。

###### 返回值

list[tuple[int, int]]
    排序后的左闭右开序列范围。

##### `DataPreparator._validate_seq_ranges`

```python
def DataPreparator._validate_seq_ranges(
    self,
    ranges,
)
```

根据数据集长度验证自定义的左闭右开序列范围。

###### 参数

- **`ranges`** (`Any`): 左闭右开的 ``(start, end)`` 索引范围。

##### `DataPreparator._merge_ranges`

```python
def DataPreparator._merge_ranges(
    self,
    ranges,
)
```

合并重叠或相接的左闭右开范围。

###### 参数

- **`ranges`** (`Any`): 左闭右开的 ``(start, end)`` 索引范围。

###### 返回值

list[tuple[int, int]]
    合并后的左闭右开范围。

##### `DataPreparator._get_used_indexes_from_ranges`

```python
def DataPreparator._get_used_indexes_from_ranges(self)
```

收集序列范围覆盖的唯一样本索引。

###### 返回值

list[int]
    被覆盖的唯一原始样本索引。

##### `DataPreparator._get_pure_shape`

```python
def DataPreparator._get_pure_shape(
    self,
    data,
)
```

移除批次维度和序列维度后推断特征形状。

###### 参数

- **`data`** (`Any`): 供 ``_get_pure_shape`` 使用的值。

###### 返回值

tuple
    移除批次维度和序列维度后的特征维度。

</details>

### 公开函数

#### `pick_data_by_indexes`

```python
def pick_data_by_indexes(
    data_dicts: List[Dict],
    used_indexes: List,
) -> List
```

按原始索引选择样本字典，并记录每个源索引。

##### 参数

- **`data_dicts`** (`List[Dict]`): 按原始样本位置建立索引的有序样本级字典。
- **`used_indexes`** (`List`): 要处理的原始样本索引。``None`` 表示选择全部样本。

##### 返回值

list[dict]
    所选样本字典。

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

收集一个数据键对应的所有序列窗口。

##### 参数

- **`data_dicts`** (`List[Dict]`): 按原始样本位置建立索引的有序样本级字典。
- **`data_key`** (`Any`): 需要收集其值的字典键。
- **`seq_indexes`** (`List`): 锚点索引及其序列索引列表组成的配对。
- **`dtype`** (`Any`): 用于数值输出的可选 NumPy dtype。
- **`seq_padding_index`** (`int`): 表示填充位置的哨兵索引。
- **`seq_padding_mode`** (`Literal['constant', 'edge']`): 填充策略：使用常量值或重复最近的边缘值。
- **`seq_padding_value`** (`Any`): 常量填充所使用的值。
- **`squeeze_singleton_dims`** (`bool`): 是否移除单例维度。
- **`keep_batch_seq_dims`** (`bool`): 数值输出是否保留显式的批次维度和序列维度。

##### 返回值

numpy.ndarray or list
    为所有序列锚点收集的值。

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

收集一个值序列，并应用常量填充或边缘填充。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`data_key`** (`Any`): 需要收集其值的字典键。
- **`used_indexes`** (`Any`): 要处理的原始样本索引。``None`` 表示选择全部样本。
- **`sample_data`** (`Any`): 用于推断类型和形状的可选代表值。
- **`data_operation`** (`Any`): 针对数组、张量、嵌套列表或标量的内部收集模式。
- **`dtype`** (`Any`): 用于数值输出的可选 NumPy dtype。
- **`seq_padding_index`** (`Any`): 表示填充位置的哨兵索引。
- **`seq_padding_mode`** (`Literal['constant', 'edge']`): 填充策略：使用常量值或重复最近的边缘值。
- **`seq_padding_value`** (`Any`): 常量填充所使用的值。
- **`squeeze_singleton_dims`** (`Any`): 是否移除单例维度。

##### 返回值

numpy.ndarray or list
    为一个序列收集的值。

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

围绕过滤后的锚点索引构造带步长的上下文窗口。

##### 参数

- **`index_list`** (`List`): 有序的原始样本索引或通用的类索引值。
- **`seq_len_before`** (`int`): 每个锚点之前的上下文步数。
- **`seq_len_after`** (`int`): 每个锚点之后的上下文步数。
- **`step_offset`** (`int`): 与已配置数据键关联的相对偏移量。
- **`stride`** (`int`): 相邻上下文位置之间的距离。
- **`seq_pos_from_start`** (`int`): 从范围起点排除的候选锚点数量。
- **`seq_pos_from_end`** (`int`): 从范围终点排除的候选锚点数量。
- **`padding`** (`bool`): 是否对不完整的边界窗口进行填充。
- **`padding_index`** (`int`): 插入填充位置的哨兵索引。

##### 返回值

tuple[list, list]
    序列元组及其锚点索引。

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

提取索引列表中某一位置周围的带步长窗口。

##### 参数

- **`index_list`** (`Any`): 有序的原始样本索引或通用的类索引值。
- **`i`** (`Any`): ``index_list`` 中的中心位置。
- **`stride`** (`Any`): 相邻上下文位置之间的距离。
- **`seq_len_before`** (`Any`): 每个锚点之前的上下文步数。
- **`seq_len_after`** (`Any`): 每个锚点之后的上下文步数。

##### 返回值

list[int]
    请求的带步长窗口中的有效位置。

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

独立划分参考组，确保测试组不会出现在训练集或验证集中。

##### 参数

- **`data_dicts`** (`List[Dict]`): 按原始样本位置建立索引的有序样本级字典。
- **`split_ref_key`** (`Any`): 用于定义分组的字典键，例如说话者、参与者或会话。
- **`split_mode`** (`Literal['holdout', 'kfold', 'leave_one_out']`): 测试集划分策略：留出法、K 折或留一法。
- **`folds`** (`int`): K 折划分所请求的折数。
- **`train_valid_ratio`** (`float`): 非测试数据中分配给训练集的比例。
- **`holdout_test_ratio`** (`float`): 在留出模式下分配给测试集的比例。
- **`use_stratified_split`** (`bool`): 是否使用标量目标以保持目标比例。
- **`split_valid_by`** (`Literal['index', 'ref_key']`): 独立验证数据是按索引还是按参考组进行划分。
- **`focused_target_key`** (`Any`): 用于分层的目标键。
- **`is_focused_key_multi_dim`** (`bool`): 焦点目标的每个样本是否包含多个值。
- **`target2indexes`** (`Optional[Dict]`): 目标值或分箱到原始样本索引的可选映射。
- **`focused_target_task_type`** (`Literal['c', 'r']`): 焦点目标类型：``"c"`` 表示分类，``"r"`` 表示回归。
- **`stratified_bin_num`** (`Optional[int]`): 用于分层的回归分箱数量。
- **`stratified_bin_size`** (`Optional[float]`): 用于分层的回归分箱宽度。
- **`used_indexes`** (`Optional[List[int]]`): 要处理的原始样本索引。``None`` 表示选择全部样本。
- **`train_ref_values_override`** (`Optional[Union[Mapping, Sequence]]`): 可选的训练参考组，可由所有折叠共享，也可按折叠指定。
- **`valid_ref_values_override`** (`Optional[Union[Mapping, Sequence]]`): 可选的验证参考组，可由所有折叠共享，也可按折叠指定。
- **`test_ref_values_override`** (`Optional[Union[Mapping, Sequence]]`): 为一个或多个折叠指定的可选测试参考组。

##### 返回值

tuple[dict, dict]
    索引折叠及对应的参考值折叠。

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

在每个参考组内部划分样本，因此同一组可以出现在所有数据集中。

##### 参数

- **`data_dicts`** (`List[Dict]`): 按原始样本位置建立索引的有序样本级字典。
- **`split_ref_key`** (`Any`): 用于定义分组的字典键，例如说话者、参与者或会话。
- **`split_mode`** (`Literal['holdout', 'kfold', 'leave_one_out']`): 测试集划分策略：留出法、K 折或留一法。
- **`folds`** (`int`): K 折划分所请求的折数。
- **`train_valid_ratio`** (`float`): 非测试数据中分配给训练集的比例。
- **`holdout_test_ratio`** (`float`): 在留出模式下分配给测试集的比例。
- **`use_stratified_split`** (`bool`): 是否使用标量目标以保持目标比例。
- **`split_valid_by`** (`Literal['index', 'ref_key']`): 独立验证数据是按索引还是按参考组进行划分。
- **`focused_target_key`** (`Any`): 用于分层的目标键。
- **`is_focused_key_multi_dim`** (`bool`): 焦点目标的每个样本是否包含多个值。
- **`target2indexes`** (`Optional[Dict]`): 目标值或分箱到原始样本索引的可选映射。
- **`focused_target_task_type`** (`Literal['c', 'r']`): 焦点目标类型：``"c"`` 表示分类，``"r"`` 表示回归。
- **`stratified_bin_num`** (`Optional[int]`): 用于分层的回归分箱数量。
- **`stratified_bin_size`** (`Optional[float]`): 用于分层的回归分箱宽度。
- **`used_indexes`** (`Optional[List[int]]`): 要处理的原始样本索引。``None`` 表示选择全部样本。

##### 返回值

tuple[dict, dict]
    索引折叠及对应的参考值折叠。

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

划分样本索引，但不强制参考组相互独立。

##### 参数

- **`data_dicts`** (`List[Dict]`): 按原始样本位置建立索引的有序样本级字典。
- **`split_ref_key`** (`Any`): 用于定义分组的字典键，例如说话者、参与者或会话。
- **`split_mode`** (`Literal['holdout', 'kfold', 'leave_one_out']`): 测试集划分策略：留出法、K 折或留一法。
- **`folds`** (`int`): K 折划分所请求的折数。
- **`train_valid_ratio`** (`float`): 非测试数据中分配给训练集的比例。
- **`holdout_test_ratio`** (`float`): 在留出模式下分配给测试集的比例。
- **`use_stratified_split`** (`bool`): 是否使用标量目标以保持目标比例。
- **`split_valid_by`** (`Literal['index', 'ref_key']`): 独立验证数据是按索引还是按参考组进行划分。
- **`target2indexes`** (`Optional[Dict]`): 目标值或分箱到原始样本索引的可选映射。
- **`focused_target_key`** (`Any`): 用于分层的目标键。
- **`is_focused_key_multi_dim`** (`bool`): 焦点目标的每个样本是否包含多个值。
- **`focused_target_task_type`** (`Literal['c', 'r']`): 焦点目标类型：``"c"`` 表示分类，``"r"`` 表示回归。
- **`stratified_bin_num`** (`Optional[int]`): 用于分层的回归分箱数量。
- **`stratified_bin_size`** (`Optional[float]`): 用于分层的回归分箱宽度。
- **`used_indexes`** (`Optional[List[int]]`): 要处理的原始样本索引。``None`` 表示选择全部样本。

##### 返回值

tuple[dict, dict]
    索引折叠及对应的参考值折叠。

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

移除序列内容与受保护划分发生重叠的低优先级锚点。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`index_split_folds`** (`Any`): 每个折叠中训练、验证和测试数据的原始索引。
- **`anchore_index2seq_indexes`** (`Any`): 每个锚点到其序列使用的全部原始索引的映射。
- **`split_ref_key`** (`Any`): 用于定义分组的字典键，例如说话者、参与者或会话。
- **`is_test_train_no_seq_overlap`** (`bool`): 是否禁止训练集与测试集之间的序列重叠。
- **`is_train_valid_no_seq_overlap`** (`bool`): 是否禁止训练集与验证集之间的序列重叠。
- **`priority_order`** (`Any`): 移除重叠时使用的由高到低的划分优先级。
- **`padding_index`** (`int`): 插入填充位置的哨兵索引。

##### 返回值

tuple[dict, dict]
    过滤后的索引折叠和重建后的参考值折叠。

#### `get_target_info_cls`

```python
def get_target_info_cls(
    data_dicts,
    target_ref_key,
    used_indexes=None,
)
```

构建类别统计、类别 ID 映射和原始索引分组。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`target_ref_key`** (`Any`): 包含目标值的字典键。
- **`used_indexes`** (`Any`): 要处理的原始样本索引。``None`` 表示选择全部样本。

##### 返回值

tuple
    类别统计、类别映射以及目标值到索引的映射。

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

对标量回归目标进行分箱，并收集分箱统计和索引。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`target_ref_key`** (`Any`): 包含目标值的字典键。
- **`used_indexes`** (`Any`): 要处理的原始样本索引。``None`` 表示选择全部样本。
- **`stratified_bin_size`** (`Any`): 用于分层的回归分箱宽度。
- **`stratified_bin_num`** (`Any`): 用于分层的回归分箱数量。
- **`bin_closed_side`** (`Literal['upper', 'lower']`): 回归分箱的边界约定。

##### 返回值

tuple
    分箱统计、分箱范围以及分箱到索引的映射。

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

将每个类别或回归分箱映射到其原始样本索引。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`target_ref_key`** (`Any`): 包含目标值的字典键。
- **`used_indexes`** (`Any`): 要处理的原始样本索引。``None`` 表示选择全部样本。
- **`task_type`** (`Any`): 目标类型：分类或回归。
- **`stratified_bin_size`** (`Optional[float]`): 用于分层的回归分箱宽度。
- **`stratified_bin_num`** (`Optional[int]`): 用于分层的回归分箱数量。

##### 返回值

dict
    类别值或回归分箱到原始索引的映射。

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

计算实际的回归分箱宽度和分箱数量。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`target_ref_key`** (`Any`): 包含目标值的字典键。
- **`used_indexes`** (`Any`): 要处理的原始样本索引。``None`` 表示选择全部样本。
- **`bin_size`** (`Any`): 请求的回归分箱宽度。
- **`bin_num`** (`Any`): 请求的回归分箱数量。
- **`target_list`** (`Any`): 可选的预先收集的标量目标。

##### 返回值

tuple[float, int]
    实际分箱宽度和分箱数量。

#### `generate_config_template`

```python
def generate_config_template(
    file: str='./data_prep_config.json',
    cls_target_keys: Union[List, str, None]=None,
    regression_target_keys: Union[List, str, None]=None,
    input_keys: Union[List, str, None]=None,
)
```

为指定的数据键写入 JSON 配置模板。

##### 参数

- **`file`** (`str`): JSON 配置文件的路径。
- **`cls_target_keys`** (`Union[List, str, None]`): 一个或多个分类目标键。
- **`regression_target_keys`** (`Union[List, str, None]`): 一个或多个回归目标键。
- **`input_keys`** (`Union[List, str, None]`): 一个或多个模型输入键。

#### `load_config`

```python
def load_config(file: str='./data_prep_config.json')
```

将 JSON 配置文件加载为配置对象。

##### 参数

- **`file`** (`str`): JSON 配置文件的路径。

##### 返回值

DataPrepConfig
    重建后的数据准备配置。

#### `make_config_from_json`

```python
def make_config_from_json(config_json)
```

根据解析后的 JSON 数据重建嵌套配置对象。

##### 参数

- **`config_json`** (`Any`): 由 ``to_json`` 生成并解析后的配置表示形式。

##### 返回值

DataPrepConfig
    重建后的数据准备配置。

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

使用显式参数或配置选项保存准备后的数据和元数据。

##### 参数

- **`collected_data`** (`Any`): 准备后的输入数组、目标数组或列表。
- **`info_dict`** (`Any`): 数据准备元数据和划分信息。
- **`data_prep_config`** (`Any`): 与准备后数据一起保存的配置对象。
- **`save_dir`** (`Any`): 目标目录。``None`` 表示使用配置的存储目录。
- **`overwrite_data`** (`Any`): 是否允许替换已有的序列化数据。

#### `load_data`

```python
def load_data(data_dir='./DataExperiment')
```

从目录中加载准备后的数据、元数据和配置。

##### 参数

- **`data_dir`** (`Any`): 包含序列化准备后数据文件的目录。

##### 返回值

tuple
    准备后的数据、元数据和配置。

#### `get_target_list`

```python
def get_target_list(
    data_dicts,
    target_ref_key,
    used_indexes=None,
)
```

从所选原始样本索引中收集目标值。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`target_ref_key`** (`Any`): 包含目标值的字典键。
- **`used_indexes`** (`Any`): 要处理的原始样本索引。``None`` 表示选择全部样本。

##### 返回值

list
    按所选索引顺序排列的目标值。

#### `get_scalar_target_list`

```python
def get_scalar_target_list(
    data_dicts,
    target_ref_key,
    used_indexes=None,
)
```

收集所选目标值，并将其规范化为 Python 标量。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`target_ref_key`** (`Any`): 包含目标值的字典键。
- **`used_indexes`** (`Any`): 要处理的原始样本索引。``None`` 表示选择全部样本。

##### 返回值

list
    按所选索引顺序排列的 Python 标量目标。

#### `convert_to_python_scalar`

```python
def convert_to_python_scalar(target_value)
```

将类标量的 Python、NumPy 或 PyTorch 值转换为 Python 标量。

##### 参数

- **`target_value`** (`Any`): 要转换的标量或仅含一个元素的类标量值。

##### 返回值

object
    等价的 Python 标量值。

#### `calculate_miu_sigma`

```python
def calculate_miu_sigma(
    data_dicts,
    ref_key,
    index_list,
)
```

计算所选样本按特征划分的均值和标准差。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`ref_key`** (`Any`): 包含计算所用值的字典键。
- **`index_list`** (`Any`): 有序的原始样本索引或通用的类索引值。

##### 返回值

tuple[numpy.ndarray, numpy.ndarray]
    按特征计算的均值和标准差。

<details>
<summary><strong>内部模块辅助函数（25）</strong></summary>

这些函数用于支持公开 API。由于源代码中包含 docstring，因此也收录于本文档，但不建议用户代码直接依赖这些内部接口。

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

为一个折叠和一种划分模式选择测试索引。

##### 参数

- **`indexes_list`** (`Any`): 可用于当前测试集选择的有序索引。
- **`current_split_group`** (`Any`): 从 0 开始的折叠位置。
- **`num_split_group`** (`Any`): 划分组的总数。
- **`one_fold_test_ratio`** (`Any`): 当 ``split_mode="holdout"`` 时使用的留出比例。
- **`split_mode`** (`Any`): 测试集划分策略：留出法、K 折或留一法。

##### 返回值

list[int]
    分配给请求测试折叠的索引。

#### `_distribute_indexes_to_folds`

```python
def _distribute_indexes_to_folds(
    index_list,
    current_group_index,
)
```

最多为请求的折叠位置分配一个索引。

##### 参数

- **`index_list`** (`Any`): 有序的原始样本索引或通用的类索引值。
- **`current_group_index`** (`Any`): 从 0 开始的组位置。

##### 返回值

list[int]
    包含一个元素的列表或空列表。

#### `_summary_seq_index_list`

```python
def _summary_seq_index_list(
    anchore_index_list,
    anchore_index2seq_index,
)
```

收集一组锚点引用的唯一序列索引。

##### 参数

- **`anchore_index_list`** (`Any`): 需要汇总其序列内容的锚点索引。
- **`anchore_index2seq_index`** (`Any`): 锚点索引到序列索引列表的映射。

##### 返回值

list[int]
    排序后的唯一序列索引。

#### `_validate_split_parameters`

```python
def _validate_split_parameters(
    split_mode,
    folds,
    train_valid_ratio,
    holdout_test_ratio,
)
```

验证常见的划分模式、折数和比例。

##### 参数

- **`split_mode`** (`Any`): 测试集划分策略：留出法、K 折或留一法。
- **`folds`** (`Any`): K 折划分所请求的折数。
- **`train_valid_ratio`** (`Any`): 非测试数据中分配给训练集的比例。
- **`holdout_test_ratio`** (`Any`): 在留出模式下分配给测试集的比例。

#### `_normalize_used_indexes`

```python
def _normalize_used_indexes(
    data_dicts,
    used_indexes,
)
```

解析并验证用于数据准备的原始样本索引。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`used_indexes`** (`Any`): 要处理的原始样本索引。``None`` 表示选择全部样本。

##### 返回值

list[int]
    验证后的原始样本索引。

#### `_balanced_chunks`

```python
def _balanced_chunks(
    values,
    num_chunks,
    reverse=False,
)
```

将有序值划分为大小近似相等的分块。

##### 参数

- **`values`** (`Any`): 待处理的值。
- **`num_chunks`** (`Any`): 要创建的分块数量。
- **`reverse`** (`Any`): 划分后是否反转分块顺序。

##### 返回值

list[list]
    大小近似相等的有序分块。

#### `_get_holdout_size`

```python
def _get_holdout_size(
    num_items,
    ratio,
)
```

在可能的情况下保留训练数据，并计算非空留出集大小。

##### 参数

- **`num_items`** (`Any`): 可用项的数量。
- **`ratio`** (`Any`): 请求的比例。

##### 返回值

int
    分配给留出测试数据的项数。

#### `_split_train_valid_indexes`

```python
def _split_train_valid_indexes(
    indexes,
    train_ratio,
)
```

在可能的情况下，将有序索引划分为非空的训练子集和验证子集。

##### 参数

- **`indexes`** (`Any`): 有序的候选索引。
- **`train_ratio`** (`Any`): 分配给训练集的比例。

##### 返回值

tuple[list, list]
    训练索引列表和验证索引列表。

#### `_stratified_train_valid_split`

```python
def _stratified_train_valid_split(
    indexes,
    target2indexes,
    train_ratio,
)
```

在每个目标分层内部划分候选索引。

##### 参数

- **`indexes`** (`Any`): 有序的候选索引。
- **`target2indexes`** (`Any`): 目标值或分箱到原始样本索引的可选映射。
- **`train_ratio`** (`Any`): 分配给训练集的比例。

##### 返回值

tuple[list, list]
    分层后的训练索引列表和验证索引列表。

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

创建或过滤目标值到原始索引的映射。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`target_key`** (`Any`): 包含目标值的字典键。
- **`used_indexes`** (`Any`): 要处理的原始样本索引。``None`` 表示选择全部样本。
- **`task_type`** (`Any`): 目标类型：分类或回归。
- **`target2indexes`** (`Any`): 目标值或分箱到原始样本索引的可选映射。
- **`stratified_bin_size`** (`Any`): 用于分层的回归分箱宽度。
- **`stratified_bin_num`** (`Any`): 用于分层的回归分箱数量。

##### 返回值

dict
    限定于所选索引的目标值或分箱映射。

#### `_normalize_fold_override`

```python
def _normalize_fold_override(override)
```

将划分覆盖设置规范化为每个折叠对应的值列表。

##### 参数

- **`override`** (`Any`): 用户提供的共享形式或逐折叠形式的划分值。

##### 返回值

list[list]
    每个折叠的覆盖值。

#### `_get_fold_override`

```python
def _get_fold_override(
    override,
    fold,
)
```

读取某一折叠的覆盖设置，并允许使用一个共享覆盖设置。

##### 参数

- **`override`** (`Any`): 用户提供的共享形式或逐折叠形式的划分值。
- **`fold`** (`Any`): 从 0 开始的折叠标识符。

##### 返回值

list or None
    分配给请求折叠的值。

#### `_resolve_train_valid_ref_values`

```python
def _resolve_train_valid_ref_values(
    candidate_values,
    train_ratio,
    train_override,
    valid_override,
)
```

解析自动生成或用户定义的训练参考组和验证参考组。

##### 参数

- **`candidate_values`** (`Any`): 尚未分配给测试集的参考值。
- **`train_ratio`** (`Any`): 分配给训练集的比例。
- **`train_override`** (`Any`): 可选的显式指定训练参考值。
- **`valid_override`** (`Any`): 可选的显式指定验证参考值。

##### 返回值

tuple[list, list]
    训练参考值和验证参考值。

#### `_validate_ref_values`

```python
def _validate_ref_values(
    values,
    available_values,
    split_name,
)
```

确保覆盖设置中的参考值存在于可用分组中。

##### 参数

- **`values`** (`Any`): 待处理的值。
- **`available_values`** (`Any`): 可供分配的参考值。
- **`split_name`** (`Any`): 错误消息中使用的易读划分名称。

#### `_build_ref_value_split_dict`

```python
def _build_ref_value_split_dict(
    data_dicts,
    split_dict,
    split_ref_key,
)
```

根据原始索引生成每个划分中的唯一参考值。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`split_dict`** (`Any`): 一个训练/验证/测试索引字典。
- **`split_ref_key`** (`Any`): 用于定义分组的字典键，例如说话者、参与者或会话。

##### 返回值

dict
    训练、验证和测试中的唯一参考值。

#### `_build_ref_value_split_folds`

```python
def _build_ref_value_split_folds(
    data_dicts,
    index_split_folds,
    split_ref_key,
)
```

为每个索引折叠生成参考值摘要。

##### 参数

- **`data_dicts`** (`Any`): 按原始样本位置建立索引的有序样本级字典。
- **`index_split_folds`** (`Any`): 每个折叠中训练、验证和测试数据的原始索引。
- **`split_ref_key`** (`Any`): 用于定义分组的字典键，例如说话者、参与者或会话。

##### 返回值

dict
    以折叠为键的参考值摘要。

#### `_normalize_index_split_folds`

```python
def _normalize_index_split_folds(index_split_folds)
```

将列表形式或映射形式的折叠规范化为一致的整数键映射。

##### 参数

- **`index_split_folds`** (`Any`): 每个折叠中训练、验证和测试数据的原始索引。

##### 返回值

dict
    使用标准划分键并以整数为键的折叠映射。

#### `_validate_split_dict`

```python
def _validate_split_dict(
    split_dict,
    allowed_indexes,
)
```

确保一个折叠中的各划分互不相交、内容完整且仅包含允许的索引。

##### 参数

- **`split_dict`** (`Any`): 一个训练/验证/测试索引字典。
- **`allowed_indexes`** (`Any`): 必须恰好分配一次的完整索引集合。

#### `_flatten`

```python
def _flatten(iterables)
```

将嵌套可迭代对象展平一层。

##### 参数

- **`iterables`** (`Any`): 需要展平一层的嵌套可迭代对象。

##### 返回值

list
    展平一层后的值。

#### `_deduplicate_preserve_order`

```python
def _deduplicate_preserve_order(values)
```

移除重复值，同时保留首次出现顺序。

##### 参数

- **`values`** (`Any`): 待处理的值。

##### 返回值

list
    按首次出现顺序排列的唯一值。

#### `_make_regression_bin_ranges`

```python
def _make_regression_bin_ranges(
    values,
    bin_size=None,
    bin_num=None,
)
```

创建覆盖已观测目标值的连续分箱边界。

##### 参数

- **`values`** (`Any`): 待处理的值。
- **`bin_size`** (`Any`): 请求的回归分箱宽度。
- **`bin_num`** (`Any`): 请求的回归分箱数量。

##### 返回值

list[tuple[float, float]]
    连续的分箱下界/上界。

#### `_find_bin_id`

```python
def _find_bin_id(
    value,
    bin_ranges,
    bin_closed_side,
)
```

查找包含某个标量回归目标的分箱。

##### 参数

- **`value`** (`Any`): 标量回归目标。
- **`bin_ranges`** (`Any`): 分箱 ID 到下界和上界的映射。
- **`bin_closed_side`** (`Any`): 回归分箱的边界约定。

##### 返回值

int
    包含该值的回归分箱标识符。

#### `_convert_value_to_bin`

```python
def _convert_value_to_bin(
    value,
    bin_ranges,
    bin_closed_side,
)
```

将标量目标转换为其所在分箱配置的边界代表值。

##### 参数

- **`value`** (`Any`): 标量回归目标。
- **`bin_ranges`** (`Any`): 分箱 ID 到下界和上界的映射。
- **`bin_closed_side`** (`Any`): 回归分箱的边界约定。

##### 返回值

float
    表示目标分箱的所选下界或上界。

#### `_init_split_set_dict`

```python
def _init_split_set_dict()
```

创建空的训练/验证/测试字典。

##### 返回值

dict[str, list]
    ``train``、``valid`` 和 ``test`` 对应的空列表。

#### `_unify_to_list`

```python
def _unify_to_list(content)
```

将标量、元组或 NumPy 数组规范化为 Python 列表。

##### 参数

- **`content`** (`Any`): 要规范化为列表的标量或集合。

##### 返回值

list
    规范化后的 Python 列表。

</details>

---

## `flexmm.experiment`

[在 GitHub 上查看源代码](https://github.com/Xia-code/flexmm/blob/main/flexmm/experiment.py)

实验配置、迭代、数据转换和评估工具。

本模块将实验流程编排划分为三个概念：

``ExperimentManager``
    加载或准备共享数据，并在每次迭代时创建一个新的实验迭代器。

``ExperimentUnit``
    保存一个输入组合和一个折叠条件下的训练、验证及测试数据。

``RunContext``
    携带执行和保存该单个实验条件所需的元数据。

准备后数据的划分索引应指向原始 ``data_dicts``
中的索引。在访问 ``collected_data`` 前，管理器会在可用时通过
``info_dict['ori_index2id']`` 将其转换为准备后数据中的位置。

### API 概览

#### 类

| 类 | 说明 |
| --- | --- |
| [`ExperimentConfig`](#experimentconfig) | 配置实验组合、数据输出和可复现性。 |
| [`TorchExperimentConfig`](#torchexperimentconfig) | 使用 PyTorch DataLoader 设置扩展 :class:`ExperimentConfig`。 |
| [`ExperimentContext`](#experimentcontext) | 携带一个输入组合和折叠条件的运行时元数据。 |
| [`ExperimentUnit`](#experimentunit) | 表示一个可执行实验条件的数据和上下文。 |
| [`ExperimentManager`](#experimentmanager) | 准备共享实验状态，并迭代实验单元。 |
| [`ExperimentResultLoader`](#experimentresultloader) | 加载已保存的实验配置、数据准备配置和结果。 |

#### 公开函数

| 函数 | 说明 |
| --- | --- |
| `generate_key_combs()` | 生成给定键的所有非空组合。 |
| `load_prepared_data()` | 加载数据、准备信息和准备配置。 |
| `iter_experiment_units()` | 为每个输入组合和折叠生成一个 :class:`ExperimentUnit`。 |
| `make_data_generator()` | 创建实验单元生成器。 |
| `make_data_geneartor()` | :func:`make_data_generator` 的已弃用拼写错误别名。 |
| `collect_combination_data()` | 为一组键选择指定的准备后数据位置。 |
| `perform_zscore()` | 使用零方差保护对数值数据进行标准化。 |
| `get_input_target_shapes()` | 收集数据集中可用的已配置输入形状和目标形状。 |
| `make_dataset()` | 创建基于字典的 PyTorch 数据集。 |
| `convert_single_data_to_tensor()` | 将支持的划分数据转换为 PyTorch 张量。 |
| `make_dataloader()` | 根据非空数据集创建 PyTorch DataLoader。 |
| `torch_postprocess()` | 将张量或类数组值转换为 NumPy 评估数组。 |
| `load_exp_config()` | 加载已保存的实验配置 JSON 文件。 |
| `make_config_from_json()` | 根据已保存的 JSON 内容构造实验配置。 |
| `compute_cls_metrics()` | 计算常用的单标签分类指标。 |
| `compute_regression_metrics()` | 计算回归任务的 MAE、MSE、RMSE 和 Pearson 相关系数。 |
| `compute_target_key_average_metric()` | 对嵌套在某个目标键下的指标求平均值。 |
| `compute_average_metric()` | 对多个结果字典中的顶层指标求平均值。 |
| `change_average_summary_form()` | 将平均值摘要转换为所选的便于分析的格式。 |
| `change_average_summary_lists()` | 将组合摘要转换为并行的组合列表和指标列表。 |

### 类

### `ExperimentConfig`

```python
class ExperimentConfig
```

配置实验组合、数据输出和可复现性。

#### 参数

- **`experiment_input_keys`** (`Any or list[Any]`): 实验使用的候选输入键。
- **`generate_input_comb`** (`bool, default=True`): 若为 ``True``，生成候选输入键的所有非空组合；若为 ``False``，将全部候选键作为一个组合。
- **`experiment_target_keys`** (`Any or list[Any]`): 每个实验单元中包含的目标键。
- **`input_comb_custom`** (`list[Any or list[Any]], optional`): 显式指定的输入组合。提供后，该值将覆盖 ``experiment_input_keys`` 和 ``generate_input_comb``。
- **`input_key_abbr`** (`dict or list, optional`): 用于构造组合目录名的短标签。列表需要与解析后的输入键顺序一致；字典必须恰好包含所有解析后的输入键。
- **`random_seed`** (`int, optional`): 基础随机种子。省略时自动生成一个种子。
- **`random_seed_scope`** (`list[str]`): 需要设置种子的随机数系统。支持的值为 ``"random"``、``"numpy"`` 和 ``"torch"``。
- **`data_level`** (`{"raw", "dataset", "dataloader"}`): 每个划分生成的数据表示层级。
- **`data_representation`** (`{"original", "pt"}`): 数据集内部的数据表示形式。``"pt"`` 会将支持的数组转换为 PyTorch 张量。
- **`load_prepared_data`** (`bool, default=True`): 从 ``store_dir`` 加载准备后数据，而不是根据 ``data_dicts`` 运行数据准备。
- **`store_dir`** (`str`): 准备后数据、实验配置和结果文件的根目录。
- **`debug_flag`** (`int`): 配置中携带的用户可控调试标志。

#### 属性

input_keys : list[Any]
    解析后的唯一输入键。
input_combs : list[list[Any]]
    解析后的输入组合。
target_keys : list[Any]
    解析后的目标键。

#### 声明字段

| 字段 | 类型 | 默认值 |
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

#### 公开方法

##### `ExperimentConfig.assert_config`

```python
def ExperimentConfig.assert_config(self) -> None
```

验证通用实验设置。

###### 异常

ValueError
    当类别设置或随机种子作用域无效时。

##### `ExperimentConfig.to_json`

```python
def ExperimentConfig.to_json(self) -> tuple[str, Dict[str, Any]]
```

将 dataclass 字段转换为可序列化的配置格式。

###### 返回值

tuple[str, dict]
    类名和 dataclass 字段字典。

<details>
<summary><strong>内部方法（2）</strong></summary>

##### `ExperimentConfig.__post_init__`

```python
def ExperimentConfig.__post_init__(self) -> None
```

规范化配置字段，并验证解析后的设置。

##### `ExperimentConfig._normalize_input_key_abbr`

```python
def ExperimentConfig._normalize_input_key_abbr(
    self,
    abbreviations: Optional[Union[Dict[Any, Any], List[Any]]],
) -> Dict[Any, Any]
```

将输入键缩写规范化为完整字典。

###### 参数

- **`abbreviations`** (`dict, list, or None`): 用户提供的缩写。

###### 返回值

dict
    每个解析后的输入键到其缩写的映射。

</details>

### `TorchExperimentConfig`

```python
class TorchExperimentConfig(ExperimentConfig)
```

使用 PyTorch DataLoader 设置扩展 :class:`ExperimentConfig`。

#### 参数

- **`train_batch_size, valid_batch_size, test_batch_size`** (`int`): 相应划分所使用的批次大小。
- **`shuffle_train_data, shuffle_valid_data, shuffle_test_data`** (`bool`): 相应 DataLoader 是否打乱其数据集。
- **`use_gpu`** (`bool`): 张量转换辅助函数是否可以将张量放置到 CUDA。通常在训练循环中将张量移至模型设备更灵活，因此默认数据集流程将张量保留在 CPU。

#### 声明字段

| 字段 | 类型 | 默认值 |
| --- | --- | --- |
| `train_batch_size` | `int` | `4` |
| `valid_batch_size` | `int` | `1` |
| `test_batch_size` | `int` | `1` |
| `shuffle_train_data` | `bool` | `True` |
| `shuffle_valid_data` | `bool` | `False` |
| `shuffle_test_data` | `bool` | `False` |
| `use_gpu` | `bool` | `USE_GPU_DEFAULT` |

<details>
<summary><strong>内部方法（1）</strong></summary>

##### `TorchExperimentConfig.__post_init__`

```python
def TorchExperimentConfig.__post_init__(self) -> None
```

初始化继承字段，并验证 PyTorch 专用设置。

</details>

### `ExperimentContext`

```python
class ExperimentContext
```

携带一个输入组合和折叠条件的运行时元数据。

#### 参数

- **`fold`** (`Any`): 保存在准备后划分信息中的折叠标识符。
- **`comb_index`** (`int`): 输入组合在实验计划中的从 0 开始的位置。
- **`comb_name`** (`str`): 适合文件系统使用的输入组合名称。
- **`input_comb`** (`list[Any]`): 该条件使用的输入键。
- **`target_keys`** (`list[Any]`): 该条件中包含的目标键。
- **`split_indexes`** (`dict[str, list[int]]`): 分配给各划分的原始 ``data_dicts`` 索引。
- **`prepared_split_indexes`** (`dict[str, list[int]]`): 用于访问 ``collected_data`` 的对应位置。
- **`ref_value_splits`** (`dict[str, list[Any]]`): 划分参考值，例如说话者标识符或组标识符。
- **`standardization_info`** (`dict`): 每个标准化输入使用的均值、标准差和作用域。
- **`info_dict`** (`dict`): 数据准备期间生成的共享信息。
- **`exp_config`** (`ExperimentConfig`): 实验配置。
- **`data_prep_config`** (`Any`): 数据准备配置。
- **`output_dir`** (`str`): 该条件推荐使用的输出目录。
- **`seed`** (`int`): 基础实验种子。
- **`user_extras`** (`dict`): 附加的用户自定义运行时信息。

#### 声明字段

| 字段 | 类型 | 默认值 |
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

#### 公开方法

##### `ExperimentContext.as_dict`

```python
def ExperimentContext.as_dict(self) -> Dict[str, Any]
```

返回上下文的浅层字典表示。

###### 返回值

dict
    以字段名为键的上下文字段。

### `ExperimentUnit`

```python
class ExperimentUnit
```

表示一个可执行实验条件的数据和上下文。

#### 参数

- **`data`** (`dict[str, Any]`): 采用已配置数据层级的训练、验证和测试数据。
- **`context`** (`RunContext`): 与该条件关联的运行时信息。

#### 声明字段

| 字段 | 类型 | 默认值 |
| --- | --- | --- |
| `data` | `Dict[str, Any]` | `_required_` |
| `context` | `ExperimentContext` | `_required_` |

#### 公开方法

##### `ExperimentUnit.input_comb`

```python
def ExperimentUnit.input_comb(self) -> List[Any]
```

返回该单元的输入键组合。

##### `ExperimentUnit.fold`

```python
def ExperimentUnit.fold(self) -> Any
```

返回该单元的折叠标识符。

##### `ExperimentUnit.info_dict`

```python
def ExperimentUnit.info_dict(self) -> Dict[str, Any]
```

返回用于兼容性的共享数据准备信息。

##### `ExperimentUnit.as_dict`

```python
def ExperimentUnit.as_dict(self) -> Dict[str, Any]
```

以字典形式返回实验单元。

###### 返回值

dict
    包含 ``data`` 和 ``context`` 的字典。

### `ExperimentManager`

```python
class ExperimentManager
```

准备共享实验状态，并迭代实验单元。

管理器可以重复迭代：每次调用 ``iter(manager)`` 都会创建一个新的
生成器，用于遍历所有输入组合和折叠条件。

#### 参数

- **`exp_config`** (`ExperimentConfig`): 实验级配置。
- **`data_dicts`** (`list[dict], optional`): 源样本信息和数据。当 ``exp_config.load_prepared_data`` 为 ``False`` 时必需。
- **`data_prep_config`** (`Any, optional`): 传递给 ``data_prep.DataPreparator`` 的配置。未加载准备后数据时必需。
- **`user_extras`** (`dict, optional`): 复制到每个 :class:`RunContext` 的附加信息。

#### 构造函数和协议方法

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

初始化管理器，但暂不加载或准备数据。

##### `ExperimentManager.__iter__`

```python
def ExperimentManager.__iter__(self) -> Iterator[ExperimentUnit]
```

返回一个新的迭代器，用于遍历所有已配置的实验单元。

#### 公开方法

##### `ExperimentManager.setup`

```python
def ExperimentManager.setup(self) -> 'ExperimentManager'
```

加载或准备共享数据，并保存实验配置。

###### 返回值

ExperimentManager
    初始化完成且可迭代的管理器自身。

##### `ExperimentManager.init_random_seed`

```python
def ExperimentManager.init_random_seed(
    self,
    random_seed: int,
) -> None
```

为已配置的随机数系统设置种子。

###### 参数

- **`random_seed`** (`int`): 应用于已配置系统的种子。

##### `ExperimentManager.get_prepared_data`

```python
def ExperimentManager.get_prepared_data(self) -> tuple[Dict[str, Any], Dict[str, Any], Any]
```

加载准备后数据，或执行数据准备流程。

###### 返回值

collected_data : dict
    以模态和目标为键的准备后数组或列表。
info_dict : dict
    划分、目标、形状和索引映射信息。
data_prep_config : Any
    用于准备数据的配置。

##### `ExperimentManager.get_result`

```python
def ExperimentManager.get_result(
    self,
    pred: Any,
    true: Any,
    task_type: TaskType='c',
) -> Dict[str, Any]
```

对预测结果进行后处理，并计算任务指标。

###### 参数

- **`pred, true`** (`Any`): 预测数组和目标数组，或对应张量。
- **`task_type`** (`{"c", "r"}`): 分类或回归任务类型。

###### 返回值

dict
    指标字典。

##### `ExperimentManager.torch_postprocess`

```python
def ExperimentManager.torch_postprocess(
    self,
    tensor: Any,
    *,
    task_type: TaskType='c',
)
```

张量后处理包装器。将张量或类数组值转换为 NumPy 评估数组。

###### 参数

- **`tensor`** (`Any`): PyTorch 张量、NumPy 数组或类数组值。
- **`task_type`** (`{"c", "r"}`): 分类或回归任务类型。

###### 返回值

dict
    指标字典。

##### `ExperimentManager.compute_result`

```python
def ExperimentManager.compute_result(
    pred: Any,
    true: Any,
    task_type: TaskType='c',
) -> Dict[str, Any]
```

计算分类或回归指标。

##### `ExperimentManager.compute_average_result`

```python
def ExperimentManager.compute_average_result(
    result_dicts: Sequence[Mapping[str, Any]],
    target_key: Any,
    metric_key: str,
) -> Any
```

在多个结果字典间对嵌套目标指标求平均值。

##### `ExperimentManager.save_exp_config`

```python
def ExperimentManager.save_exp_config(self) -> None
```

将实验配置保存到 ``store_dir`` 下。

##### `ExperimentManager.save_result`

```python
def ExperimentManager.save_result(
    self,
    result: Any,
    context: Optional[ExperimentContext]=None,
    file_name: str='ExpResult.pkl',
) -> str
```

全局保存结果，或将结果保存到某次运行的输出目录中。

###### 参数

- **`result`** (`Any`): 可使用 pickle 序列化的结果对象。
- **`context`** (`RunContext, optional`): 提供时保存到 ``context.output_dir`` 下；否则保存到管理器的 ``store_dir`` 下。
- **`file_name`** (`str`): 输出文件名。

###### 返回值

str
    已保存文件的路径。

<details>
<summary><strong>内部方法（1）</strong></summary>

##### `ExperimentManager._validate_prepared_data`

```python
def ExperimentManager._validate_prepared_data(self) -> None
```

验证实验计划所需的键和划分元数据。

</details>

### `ExperimentResultLoader`

```python
class ExperimentResultLoader
```

加载已保存的实验配置、数据准备配置和结果。

#### 参数

- **`result_dir`** (`str`): 包含已保存文件的目录。
- **`exp_config_file, data_prep_config_file, result_file`** (`str`): 相对于 ``result_dir`` 的文件名。

#### 构造函数和协议方法

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

加载所有请求的实验产物。

### 公开函数

#### `generate_key_combs`

```python
def generate_key_combs(keys: Sequence[Any]) -> List[List[Any]]
```

生成给定键的所有非空组合。

##### 参数

- **`keys`** (`sequence`): 按期望组合顺序排列的输入键。

##### 返回值

list[list]
    先按组合大小、再按输入顺序排列的组合。

#### `load_prepared_data`

```python
def load_prepared_data(data_dir: str='./DataExperiment') -> tuple[Dict[str, Any], Dict[str, Any], Any]
```

加载数据、准备信息和准备配置。

##### 参数

- **`data_dir`** (`str`): 由 ``data_prep.save_data`` 创建的目录。

##### 返回值

collected_data, info_dict, data_prep_config : tuple
    通过 ``data_prep.load_data`` 加载的对象。

##### 异常

FileNotFoundError
    当目录或所需文件不存在时。

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

为每个输入组合和折叠生成一个 :class:`ExperimentUnit`。

当 ``standardize_scope='split'`` 时，标准化使用当前折叠的训练数据
计算统计量，并将相同统计量应用于训练、验证和
测试数据，从而避免验证/测试数据泄漏。``'all'`` 会使用全部准备后
样本，因此应仅在明确了解其含义时选择。

##### 参数

- **`collected_data`** (`dict`): 准备后的模态、目标和索引数据。
- **`info_dict`** (`dict`): 包含划分折叠和索引映射的数据准备信息。
- **`exp_config`** (`ExperimentConfig`): 实验配置。
- **`data_prep_config`** (`Any`): 包含 ``key2config`` 的数据准备配置。
- **`user_extras`** (`dict, optional`): 复制到每个运行上下文中的附加信息。

##### 生成值

ExperimentUnit
    一个实验条件对应的数据和运行时信息。

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

创建实验单元生成器。

此兼容函数将处理委托给 :func:`iter_experiment_units`。
可选的数据层级参数会应用于配置对象的浅拷贝，
因此不会修改调用方的原始配置。

#### `make_data_geneartor`

```python
def make_data_geneartor(
    *args: Any,
    **kwargs: Any,
) -> Iterator[ExperimentUnit]
```

:func:`make_data_generator` 的已弃用拼写错误别名。

#### `collect_combination_data`

```python
def collect_combination_data(
    collected_data: Mapping[Any, Any],
    keys: Sequence[Any],
    index_list: Sequence[int],
) -> Dict[Any, Any]
```

为一组键选择指定的准备后数据位置。

##### 参数

- **`collected_data`** (`mapping`): 以输入键、目标键或索引键组织的准备后数据。
- **`keys`** (`sequence`): 结果中需要包含的键。
- **`index_list`** (`sequence[int]`): 准备后数据数组/列表中的位置。

##### 返回值

dict
    以 ``keys`` 中的键组织的所选数据。

#### `perform_zscore`

```python
def perform_zscore(
    data_array: Any,
    axis: Union[int, tuple[int, ...]]=0,
) -> np.ndarray
```

使用零方差保护对数值数据进行标准化。

##### 参数

- **`data_array`** (`array-like`): 数值数据。
- **`axis`** (`int or tuple[int, ...]`): 用于计算均值和标准差的轴。

##### 返回值

numpy.ndarray
    标准化后的数据。

#### `get_input_target_shapes`

```python
def get_input_target_shapes(
    datasets: Mapping[str, Any],
    info_dict: Mapping[str, Any],
) -> Dict[Any, tuple[int, ...]]
```

收集数据集中可用的已配置输入形状和目标形状。

##### 参数

- **`datasets`** (`mapping`): 包含 ``"train"`` 的划分数据集或原始划分字典。
- **`info_dict`** (`mapping`): 包含 ``input_shapes`` 和 ``target_info`` 的数据准备信息。

##### 返回值

dict
    以输入名称或目标名称为键的形状信息。

#### `make_dataset`

```python
def make_dataset(
    single_condition_data: Dict[Any, Any],
    data_representation: DataRepresentation='original',
    use_gpu: bool=False,
) -> TorchDataset
```

创建基于字典的 PyTorch 数据集。

##### 参数

- **`single_condition_data`** (`dict`): 一个训练、验证或测试划分的数据。
- **`data_representation`** (`{"original", "pt"}`): 保持值不变，或将支持的值转换为张量。
- **`use_gpu`** (`bool`): 将转换后的张量放置到 CUDA。当数据集将由 DataLoader 包装时，建议使用 CPU 张量。

##### 返回值

TorchDataset
    创建的数据集。

#### `convert_single_data_to_tensor`

```python
def convert_single_data_to_tensor(
    single_condition_data: Mapping[Any, Any],
    *,
    exclude_keys: Optional[Iterable[Any]]=None,
    use_gpu: bool=False,
) -> Dict[Any, Any]
```

将支持的划分数据转换为 PyTorch 张量。

在可能的情况下，索引元数据会转换为 ``torch.long``。不支持的
Python 对象将保持不变。

##### 参数

- **`single_condition_data`** (`mapping`): 以模态键、目标键或元数据键组织的划分数据。
- **`exclude_keys`** (`iterable, optional`): 不进行转换而直接复制的键。
- **`use_gpu`** (`bool`): 在 CUDA 可用时，将转换后的张量移动到 CUDA。

##### 返回值

dict
    转换为张量后的数据。

#### `make_dataloader`

```python
def make_dataloader(
    dataset: TorchDataset,
    batch_size: int,
    shuffle: bool,
) -> Any
```

根据非空数据集创建 PyTorch DataLoader。

##### 参数

- **`dataset`** (`TorchDataset`): 需要进行批处理的数据集。
- **`batch_size`** (`int`): 每个批次的正整数样本数量。
- **`shuffle`** (`bool`): 是否打乱样本顺序。

##### 返回值

torch.utils.data.DataLoader
    创建的 DataLoader。

#### `torch_postprocess`

```python
def torch_postprocess(
    tensor: Any,
    *,
    mode: Literal['raw', 'argmax']='raw',
    use_gpu: Optional[bool]=None,
) -> np.ndarray
```

将张量或类数组值转换为 NumPy 评估数组。

##### 参数

- **`tensor`** (`Any`): PyTorch 张量、NumPy 数组或类数组值。
- **`mode`** (`{"raw", "argmax"}`): 返回原始值，或先对最后一个轴执行 argmax。
- **`use_gpu`** (`bool, optional`): 已弃用的兼容参数。设备处理将根据张量本身自动推断。

##### 返回值

numpy.ndarray
    已从计算图分离的 CPU 数组。

#### `load_exp_config`

```python
def load_exp_config(file: str='./ExpConfig.json') -> ExperimentConfig
```

加载已保存的实验配置 JSON 文件。

#### `make_config_from_json`

```python
def make_config_from_json(config_json: Sequence[Any]) -> ExperimentConfig
```

根据已保存的 JSON 内容构造实验配置。

##### 参数

- **`config_json`** (`sequence`): 包含类名和构造字段的双元素序列。

##### 返回值

ExperimentConfig
    重建后的配置对象。

#### `compute_cls_metrics`

```python
def compute_cls_metrics(
    pred_list: Any,
    true_list: Any,
) -> Dict[str, Any]
```

计算常用的单标签分类指标。

当 Pearson 相关系数无法定义时（例如
数组为常量或元素少于两个），返回 ``nan``。

#### `compute_regression_metrics`

```python
def compute_regression_metrics(
    pred_list: Any,
    true_list: Any,
) -> Dict[str, Any]
```

计算回归任务的 MAE、MSE、RMSE 和 Pearson 相关系数。

#### `compute_target_key_average_metric`

```python
def compute_target_key_average_metric(
    result_dicts: Sequence[Mapping[Any, Any]],
    target_key: Any,
    metric_key: str,
) -> Any
```

对嵌套在某个目标键下的指标求平均值。

#### `compute_average_metric`

```python
def compute_average_metric(
    result_dicts: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> Any
```

对多个结果字典中的顶层指标求平均值。

#### `change_average_summary_form`

```python
def change_average_summary_form(
    average_summary: Mapping[str, Any],
    form: Literal['lists']='lists',
) -> Dict[str, Any]
```

将平均值摘要转换为所选的便于分析的格式。

#### `change_average_summary_lists`

```python
def change_average_summary_lists(average_summary: Mapping[str, Any]) -> Dict[str, Any]
```

将组合摘要转换为并行的组合列表和指标列表。

<details>
<summary><strong>内部模块辅助函数（17）</strong></summary>

这些函数用于支持公开 API。由于源代码中包含 docstring，因此也收录于本文档，但不建议用户代码直接依赖这些内部接口。

#### `_make_dataset_tensor`

```python
def _make_dataset_tensor(
    single_condition_data: Dict[Any, Any],
    use_gpu: bool=False,
) -> TorchDataset
```

将划分数据转换为张量，并包装为 :class:`TorchDataset`。

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

加载各折叠摘要，并计算每个组合的平均值。

必须提供 ``result_dir``，或者同时提供 ``result_root`` 和
``experiment_name``。

#### `_unify_to_list`

```python
def _unify_to_list(value: Any) -> List[Any]
```

将标量或可迭代配置输入转换为普通列表。

#### `_unique_in_order`

```python
def _unique_in_order(values: Iterable[Any]) -> List[Any]
```

返回唯一值，同时保留首次出现顺序。

#### `_normalize_fold_mapping`

```python
def _normalize_fold_mapping(folds: Any) -> Dict[Any, Dict[str, List[int]]]
```

将列表形式或映射形式的划分折叠规范化为有序字典。

#### `_normalize_optional_fold_mapping`

```python
def _normalize_optional_fold_mapping(
    folds: Any,
    fold_keys: Iterable[Any],
) -> Dict[Any, Dict[str, List[Any]]]
```

规范化可选的参考值划分信息。

#### `_translate_split_indexes`

```python
def _translate_split_indexes(
    split_indexes: Mapping[str, Sequence[int]],
    info_dict: Mapping[str, Any],
    collected_data: Mapping[Any, Any],
) -> Dict[str, List[int]]
```

将原始样本索引转换为准备后数据位置。

#### `_infer_sample_count`

```python
def _infer_sample_count(collected_data: Mapping[Any, Any]) -> int
```

推断并验证共享的第一维样本数量。

#### `_standardize_split_data`

```python
def _standardize_split_data(
    split_data: Dict[str, Dict[Any, Any]],
    collected_data: Mapping[Any, Any],
    input_comb: Sequence[Any],
    data_prep_config: Any,
) -> tuple[Dict[str, Dict[Any, Any]], Dict[Any, Dict[str, Any]]]
```

在不修改共享数据的情况下，对已配置的数值输入进行标准化。

#### `_apply_standardization`

```python
def _apply_standardization(
    array: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray
```

应用 z-score 归一化，并将零方差轴映射为零。

#### `_as_numeric_array`

```python
def _as_numeric_array(
    value: Any,
    key: Any,
) -> np.ndarray
```

将值转换为非 object 类型的数值 NumPy 数组。

#### `_convert_data_level`

```python
def _convert_data_level(
    split_data: Dict[str, Dict[Any, Any]],
    exp_config: ExperimentConfig,
) -> Dict[str, Any]
```

将原始划分字典转换为请求的实验数据层级。

#### `_make_comb_name`

```python
def _make_comb_name(
    input_comb: Sequence[Any],
    abbreviations: Mapping[Any, Any],
) -> str
```

为输入组合创建适合文件系统使用的目录名。

#### `_sanitize_path_component`

```python
def _sanitize_path_component(value: str) -> str
```

将路径中不安全的字符替换为下划线。

#### `_require_torch`

```python
def _require_torch(caller: str) -> None
```

当 PyTorch 辅助函数不可用时，抛出信息明确的错误。

#### `_needs_argmax`

```python
def _needs_argmax(prediction: Any) -> bool
```

返回分类预测是否包含类别得分轴。

#### `_safe_pearson`

```python
def _safe_pearson(
    first: np.ndarray,
    second: np.ndarray,
) -> float
```

计算 Pearson 相关系数；无法定义时返回 ``nan``。

</details>

---

## `flexmm.info_utils`

[在 GitHub 上查看源代码](https://github.com/Xia-code/flexmm/blob/main/flexmm/info_utils.py)

用于查询和组织样本级信息记录的工具。

本模块中的函数要求 ``data_dicts`` 为字典列表。
每个字典描述一个样本、话语、帧或其他序列项。
字典字段可以包含标识符、分组变量、目标值或
其他轻量级样本信息。

### API 概览

#### 公开函数

| 函数 | 说明 |
| --- | --- |
| `get_ref_value2indexes()` | 将每个参考值映射到其对应的样本索引。 |
| `get_ref_value_list()` | 为每个已处理样本收集一个参考值。 |
| `get_ref_value2another()` | 将每个参考值映射到另一个样本字段中的值。 |
| `get_turn2ref_value_and_indexes()` | 将参考值相同的连续样本分组为轮次。 |
| `get_ref_value2turn_indexes()` | 将每个参考值映射到其各轮次的样本索引列表。 |
| `get_ref_value2turns()` | 将每个参考值映射到其从 0 开始的轮次 ID。 |
| `get_ref_value2indexes_in_turns()` | 将每个参考值映射到通过其轮次收集的索引。 |
| `get_ref_value2adjacent_ref_value()` | 按样本索引或轮次汇总不同参考值之间的相邻关系。 |
| `get_interval_split_indexes()` | 将序列划分为大小近似相等的时间顺序区间。 |

### 公开函数

#### `get_ref_value2indexes`

```python
def get_ref_value2indexes(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> Dict[Any, List[int]]
```

将每个参考值映射到其对应的样本索引。

##### 参数

- **`data_dicts`** (`list of dict`): 样本级信息记录。每个已处理字典都必须包含 ``ref_key``。
- **`ref_key`** (`Any, optional`): 其值用于定义参考组的字典键，例如说话者、参与者、会话或组标识符。
- **`used_indexes`** (`list of int or None, optional`): 要处理的样本索引子集。``None`` 表示使用全部样本。

##### 返回值

dict
    ``ref_key`` 的每个值到包含该值的样本索引的映射；
    索引顺序遵循 ``used_indexes``。

##### 异常

KeyError
    当已处理字典不包含 ``ref_key`` 时。
TypeError
    当参考值不可哈希，因而不能用作
    字典键。

#### `get_ref_value_list`

```python
def get_ref_value_list(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> List[Any]
```

为每个已处理样本收集一个参考值。

##### 参数

- **`data_dicts`** (`list of dict`): 样本级信息记录。每个已处理字典都必须包含 ``ref_key``。
- **`ref_key`** (`Any, optional`): 从每个样本中收集其值的字典键。
- **`used_indexes`** (`list of int or None, optional`): 要处理的样本索引子集。``None`` 表示使用全部样本。

##### 返回值

list
    按样本顺序排列的 ``ref_key`` 值；重复值会被保留。

##### 异常

KeyError
    当已处理字典不包含 ``ref_key`` 时。

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

将每个参考值映射到另一个样本字段中的值。

此函数适合检查两个信息字段之间的关系，
例如将每位说话者映射到其目标值，或将每个组映射到
其参与者标识符。

##### 参数

- **`data_dicts`** (`list of dict`): 样本级信息记录。每个已处理字典都必须同时包含 ``ref_key`` 和 ``another_key``。
- **`ref_key`** (`Any, optional`): 其值将成为返回映射键的字典键。
- **`another_key`** (`Any, optional`): 针对每个参考值收集其值的字典键。
- **`used_indexes`** (`list of int or None, optional`): 要处理的样本索引子集。``None`` 表示使用全部样本。
- **`unique_values`** (`bool, optional`): 若为 ``True``，每个参考值的等价重复值只保存一次；若为 ``False``，保留所有值，包括重复值。两种模式均支持类数组值。

##### 返回值

dict
    ``ref_key`` 的每个值到 ``another_key`` 值的映射。

##### 异常

KeyError
    当已处理字典缺少 ``ref_key`` 或 ``another_key`` 时。

#### `get_turn2ref_value_and_indexes`

```python
def get_turn2ref_value_and_indexes(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> Dict[int, Dict[str, Any]]
```

将参考值相同的连续样本分组为轮次。

当 ``ref_key`` 的值发生变化，或下一个
待处理样本索引在数值上不与前一个索引连续时，将开始一个新的轮次。
保留 ``used_indexes`` 中给出的顺序。

##### 参数

- **`data_dicts`** (`list of dict`): 表示一段对话、会话、事件或其他序列的有序样本级记录。
- **`ref_key`** (`Any, optional`): 用于判断连续样本是否属于同一轮次的字典键，通常为说话者或参与者标识符。
- **`used_indexes`** (`list of int or None, optional`): 要处理的有序样本索引子集。``None`` 表示使用全部样本。

##### 返回值

dict
    从 0 开始的轮次 ID 到包含两个字段的字典的映射：
    ``"ref_value"`` 保存轮次的参考值，``"indexes"``
    保存该轮次中的样本索引。

##### 异常

KeyError
    当已处理字典不包含 ``ref_key`` 时。

#### `get_ref_value2turn_indexes`

```python
def get_ref_value2turn_indexes(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> Dict[Any, List[List[int]]]
```

将每个参考值映射到其各轮次的样本索引列表。

##### 参数

- **`data_dicts`** (`list of dict`): 表示一个序列的有序样本级记录。
- **`ref_key`** (`Any, optional`): 用于标识每个轮次所属者的字典键。
- **`used_indexes`** (`list of int or None, optional`): 要处理的有序样本索引子集。``None`` 表示使用全部样本。

##### 返回值

dict
    每个参考值到轮次列表的映射，其中每个轮次
    由其样本索引列表表示。

#### `get_ref_value2turns`

```python
def get_ref_value2turns(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> Dict[Any, List[int]]
```

将每个参考值映射到其从 0 开始的轮次 ID。

##### 参数

- **`data_dicts`** (`list of dict`): 表示一个序列的有序样本级记录。
- **`ref_key`** (`Any, optional`): 用于标识每个轮次所属者的字典键。
- **`used_indexes`** (`list of int or None, optional`): 要处理的有序样本索引子集。``None`` 表示使用全部样本。

##### 返回值

dict
    每个参考值到其轮次 ID 的映射。

#### `get_ref_value2indexes_in_turns`

```python
def get_ref_value2indexes_in_turns(
    data_dicts: List[Dict],
    ref_key: Any='speaker',
    used_indexes: Union[List[int], None]=None,
) -> Dict[Any, List[int]]
```

将每个参考值映射到通过其轮次收集的索引。

这是 :func:`get_ref_value2turn_indexes` 的扁平化对应函数。
返回值中不保留轮次边界。

##### 参数

- **`data_dicts`** (`list of dict`): 表示一个序列的有序样本级记录。
- **`ref_key`** (`Any, optional`): 用于标识每个轮次所属者的字典键。
- **`used_indexes`** (`list of int or None, optional`): 要处理的有序样本索引子集。``None`` 表示使用全部样本。

##### 返回值

dict
    每个参考值到其全部样本索引的映射，并按
    轮次出现顺序排列。

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

按样本索引或轮次汇总不同参考值之间的相邻关系。

顶层键是当前关注项的参考值。每个
嵌套键是相邻项的参考值。返回的整数
在 ``adjacent_by="index"`` 时表示当前样本，在
``adjacent_by="turn"`` 时表示当前轮次。

仅包含参考值不同的相邻项。当提供的
子集存在索引间隔时，不会跨越这些
不连续的间隔建立相邻关系。

##### 参数

- **`data_dicts`** (`list of dict`): 表示一个序列的有序样本级记录。
- **`ref_key`** (`Any, optional`): 其值用于比较相邻项的字典键。
- **`prev_or_following`** (`{"prev", "following"}, optional`): 相对于当前样本或轮次的相邻方向。
- **`adjacent_by`** (`{"index", "turn"}, optional`): 用于定义相邻关系的单位。``"index"`` 比较相邻样本索引；``"turn"`` 比较相邻轮次。
- **`used_indexes`** (`list of int or None, optional`): 要处理的有序样本索引子集。``None`` 表示使用全部样本。

##### 返回值

dict
    嵌套映射：
    ``current_ref_value -> adjacent_ref_value -> current_indexes_or_turns``。

##### 异常

ValueError
    当 ``prev_or_following`` 或 ``adjacent_by`` 无效时。

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

将序列划分为大小近似相等的时间顺序区间。

按索引划分时，区间直接包含样本索引。按
轮次划分时，完整轮次先分配到各区间，然后
再展开为其样本索引，因此一个轮次不会被拆分到
两个区间中。

##### 参数

- **`data_dicts`** (`list of dict`): 表示一个序列的有序样本级记录。
- **`ref_key`** (`Any, optional`): 当 ``interval_split_by="turn"`` 时用于标识轮次的字典键。
- **`interval_num`** (`int, optional`): 要创建的区间数量。返回的字典始终包含该数量的区间键；当样本数或轮次数少于区间数时，部分区间可能为空。
- **`interval_split_by`** (`{"index", "turn"}, optional`): 区间边界是基于样本索引还是完整轮次。
- **`used_indexes`** (`list of int or None, optional`): 要处理的有序样本索引子集。``None`` 表示使用全部样本。

##### 返回值

dict
    从 0 开始的区间 ID 到样本索引列表的映射。展开轮次前，
    各区间大小最多相差一个单位。

##### 异常

ValueError
    当 ``interval_num`` 不是正数或 ``interval_split_by`` 无效时。

<details>
<summary><strong>内部模块辅助函数（3）</strong></summary>

这些函数用于支持公开 API。由于源代码中包含 docstring，因此也收录于本文档，但不建议用户代码直接依赖这些内部接口。

#### `_resolve_used_indexes`

```python
def _resolve_used_indexes(
    data_dicts: List[Dict],
    used_indexes: Union[List[int], None],
) -> List[int]
```

解析并验证要处理的样本索引。

##### 参数

- **`data_dicts`** (`list of dict`): 样本级信息记录。
- **`used_indexes`** (`list of int or None`): 要处理的索引。``None`` 表示按原始顺序返回全部索引。

##### 返回值

list of int
    验证后的待处理索引。

##### 异常

TypeError
    当索引不是整数时。
IndexError
    当索引超出 ``data_dicts`` 的有效范围时。

#### `_values_equal`

```python
def _values_equal(
    left: Any,
    right: Any,
) -> bool
```

比较两个值，同时支持标量和类数组对象。

##### 参数

- **`left`** (`Any`): 第一个待比较的值。
- **`right`** (`Any`): 第二个待比较的值。

##### 返回值

bool
    当两个值等价时为 ``True``。类数组比较结果会
    在所有元素上进行归约。

#### `_contains_equivalent`

```python
def _contains_equivalent(
    values: List[Any],
    candidate: Any,
) -> bool
```

检查列表中是否包含与候选值等价的值。

##### 参数

- **`values`** (`list`): 待搜索的现有值。
- **`candidate`** (`Any`): 候选值，可以是标量或类数组对象。

##### 返回值

bool
    若已存在等价值，则为 ``True``。

</details>

---

## 文档维护

本文档目前记录了从源代码中提取的 **160** 个类、方法和模块级函数。

当函数签名或 docstring 发生变化时，请重新生成本文档，以保持文档与代码同步。有关概念性流程说明和端到端示例，请参阅仓库中的 [README](https://github.com/Xia-code/flexmm)。
