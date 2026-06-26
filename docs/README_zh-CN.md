# FlexMM

[English](../README.md) | **中文** | [日本語](README_ja.md)

**FlexMM** 是一个与模型架构无关的 Python 框架，用于支持具备序列结构的多模态数据准备与可复现实验编排。

它为从“按顺序排列的样本字典列表”到以下内容提供统一流程：

- 对齐后的多模态输入与目标数组；
- 可配置的时间上下文窗口；
- 分类与回归目标的统计和映射信息；
- 按组独立、按组相关或不受组约束的训练/验证/测试划分；
- 跨折、跨输入模态组合的实验；
- 原始字典、PyTorch Dataset 或 PyTorch DataLoader；
- 每次运行所需的上下文、标准化统计量、评价指标和结果保存。

FlexMM 有意**不规定**模型结构或训练循环。模型构建、优化器、早停和日志记录仍由用户控制，而 FlexMM 负责其中重复、繁琐且容易出错的实验准备工作。

---

## 为什么使用 FlexMM？

多模态实验通常会反复进行以下基础工作：

1. 对齐不同模态与目标值；
2. 构造时间序列窗口；
3. 按说话人、参与者、组别或会话进行数据划分；
4. 防止序列在不同数据集之间发生泄漏；
5. 对每一种模态组合和每一个 fold 重复实验；
6. 将 fold 相关信息传入训练与测试流程；
7. 保存足够的信息以便复现实验。

FlexMM 将这些步骤封装为三个职责明确的模块：

| 模块 | 职责 |
|---|---|
| `flexmm.info_utils` | 查询和组织轻量级样本信息、turn、相邻关系和区间。 |
| `flexmm.data_prep` | 构建对齐后的序列数据、处理目标值、生成数据划分并序列化准备结果。 |
| `flexmm.experiment` | 通过 `ExperimentManager`、`ExperimentUnit` 和 `RunContext` 遍历输入组合与 fold。 |

---

## 主要功能

- **统一的有序样本格式**：`data_dicts` 中的每个元素表示一个样本、帧、话语或事件。
- **多模态输入**：可独立配置任意数量的输入键。
- **序列感知的数据准备**：支持过去/未来上下文、stride、offset、边界过滤以及常数/边缘 padding。
- **分类与回归任务**：支持标签映射、目标统计、回归分箱和多维回归目标。
- **三种划分语义**：
  - 按说话人/组别/会话独立划分；
  - 在每个 reference group 内部相关划分；
  - 不受 reference group 约束、直接按样本划分。
- **Holdout、K-fold 与 Leave-one-out** 测试策略。
- **可选目标分层**：支持标量分类目标和标量回归目标。
- **序列重叠保护**：防止 test/train，以及可选的 train/validation 之间出现窗口重叠。
- **输入模态组合实验**：可自动生成全部组合，也可显式指定。
- **避免泄漏的标准化**：每个 fold 的标准化参数只由训练集拟合，并应用于验证集和测试集。
- **可选 PyTorch 集成**：数据准备阶段不强制依赖 PyTorch。
- **可重复迭代的实验管理器**：同一个 manager 可以多次遍历，而不是一次性 generator。
- **可复现实验产物**：保存数据准备配置、实验配置、划分信息、目标映射和索引映射。

---

## 安装

FlexMM 需要 Python 3.9 或更高版本。

在仓库根目录执行：

```bash
pip install -e .
```

核心运行依赖：

```bash
pip install numpy scipy scikit-learn
```

PyTorch 为可选依赖，仅在使用 `data_level="dataset"`、`data_level="dataloader"` 或显式张量转换时需要：

```bash
pip install torch
```

> 正式发布前，请在 `pyproject.toml` 或等效的打包文件中声明这些依赖，使用户通常只需执行 `pip install -e .` 或 `pip install flexmm`。

---

## 数据模型

FlexMM 接收一个**按顺序排列的字典列表**。列表位置就是用于数据对齐、序列构造和划分记录的原始样本索引。

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

虽然框架内部依赖原始列表索引完成对齐，但仍强烈建议提供稳定的 `sample_id`，便于审计、导出结果和与外部数据对应。

字典中常见的字段包括：

- 标识信息：`sample_id`、`speaker`、`participant`、`group`、`session`；
- 目标值：`label`、`score`、`valence`、`arousal`；
- 模型输入：`audio`、`video`、`text`、传感器向量、手工特征；
- 描述信息：时间戳、实验条件、源文件路径、trial ID。

同一个配置键在不同样本中的数值形状应当兼容。可以组成规则数组的值会被收集为 NumPy 数组；异构或非数值内容会保留为 Python 列表。

> 当前数据准备接口要求配置中使用的值在准备阶段可从 `data_dicts` 访问。对于无法全部载入内存的大型数据集，可以在 `data_dicts` 中保存紧凑引用，并增加项目自定义加载层，或进一步扩展为 feature-store 接口。

---

## 快速开始

### 1. 配置并准备数据

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

该步骤会生成对齐后的数据和元信息。启用保存后，输出目录如下：

```text
ExperimentStore/demo/
├── Data.pkl
├── Info.pkl
└── DataPrepConfig.json
```

### 2. 创建实验管理器

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

当有两个输入键并设置 `generate_input_comb=True` 时，FlexMM 会为每个 fold 生成：

```text
[audio]
[text]
[audio, text]
```

### 3. 运行自己的训练循环

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

请在训练循环内部将每个 batch 移动到模型所在设备：

```python
batch = {
    key: value.to(device) if isinstance(value, torch.Tensor) else value
    for key, value in batch.items()
}
```

创建 DataLoader 时，manager 会让 dataset tensor 保持在 CPU 上，这对显存占用和多进程加载更安全。

---

## 不单独调用数据准备的一体化模式

manager 也可以在内部执行数据准备：

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

如果同一份准备后数据会被多个模型或超参数实验重复使用，推荐采用“两阶段流程”；如果只是编写简洁脚本或进行一次性实验，可以使用一体化模式。

---

## 实验对象

### `ExperimentManager`

负责共享实验状态。它会加载或准备数据、初始化随机种子、检查所需输入和目标键、保存 `ExpConfig.json`，并为全部“输入组合 × fold”条件创建可重复迭代的流程。

### `ExperimentUnit`

表示一个可执行的实验条件：

```python
unit.data       # {"train": ..., "valid": ..., "test": ...}
unit.context    # RunContext
unit.input_comb # 兼容属性
unit.fold       # 兼容属性
```

### `RunContext`

将当前运行需要的信息传入模型构建、训练、评价和结果保存流程：

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

`split_indexes` 保存原始 `data_dicts` 索引；`prepared_split_indexes` 保存经过序列过滤和数据准备之后在 `collected_data` 中的位置。

---

## 数据划分策略

### Independent split

```python
split_dependency="independent"
```

由 `split_ref_key` 标识的 reference value 会在测试集和非测试集之间完全分离。适合说话人独立、参与者独立、组别独立或会话独立评价。

验证集可通过以下方式选取：

```python
independent_split_valid_by="ref_key"  # 验证组也与训练组独立
independent_split_valid_by="index"    # 验证样本可以与训练样本来自相同组
```

### Dependent split

```python
split_dependency="dependent"
independent_split_valid_by=None
```

每个 reference group 都会向训练集、验证集和测试集提供样本。仅当评价目标允许同一参与者或同一组同时出现在各数据集时使用。

### Unconstrained split

```python
split_dependency="none"
independent_split_valid_by=None
```

忽略 reference group，直接按符合条件的样本索引划分。

### 测试模式

所有划分策略都支持：

```python
split_mode="holdout"
split_mode="kfold"
split_mode="leave_one_out"
```

划分过程是确定性的，并保留输入/reference 的原始顺序。FlexMM 不会在划分前自动打乱数据。如果需要随机划分，请有意识地整理数据、使用显式划分覆盖，或在准备前进行可复现的排序/打乱。

---

## 序列构造

每个输入或目标配置都可以指定：

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

例如：

```python
InputConfig(
    keys="audio",
    seq_len_before=2,
    seq_len_after=1,
    stride=1,
    seq_padding=True,
)
```

概念上会构造如下窗口：

```text
[t-2, t-1, t, t+1]
```

序列边界由 `DataPrepConfig` 控制：

```python
seq_group_mode="ref_key"  # 根据 seq_group_key/split_ref_key 的连续区间分组
seq_group_key="speaker"
```

或者：

```python
seq_group_mode="index"    # 将完整有序数据集视为一个范围
```

也支持自定义半开区间：

```python
seq_ranges_custom=[(0, 100), (150, 220)]
```

当不同 split 中的序列窗口发生交叉时，可以按优先级移除低优先级 anchor：

```python
remove_test_train_overlap_range=True
remove_train_valid_overlap_range=False
remove_overlap_priority=["test", "train", "valid"]
```

默认优先保留测试集，其次是训练集，最后是验证集。

---

## 目标值

### 分类

```python
ClassificationTargetConfig(
    keys="label",
    convert_target_to_id=True,
)
```

数据准备阶段会记录：

```python
info_dict["target_info"]["label"]["target2id"]
info_dict["target_info"]["label"]["id2target"]
info_dict["target_info"]["label"]["target_stats"]
info_dict["target_info"]["label"]["target2indexes"]
```

分类目标应为标量或可转换为标量的值。

### 回归

```python
from flexmm.data_prep import RegressionTargetConfig

RegressionTargetConfig(
    keys="score",
    stratified_bin_num=10,
    convert_target_to_bin=False,
)
```

标量回归目标可以分箱后用于近似分层划分。多维回归目标可通过以下方式配置：

```python
RegressionTargetConfig(
    keys="trajectory",
    is_multi_dim=True,
)
```

多维目标不能用于分层划分。

---

## 标准化与数据泄漏

在输入配置中启用标准化：

```python
InputConfig(
    keys="audio",
    standardize_data=True,
    standardize_scope="split",
)
```

- `standardize_scope="split"`：使用当前 fold 的**训练集**拟合均值和标准差，再将同一组参数应用于 train、validation 和 test。推荐用于正式评价。
- `standardize_scope="all"`：使用全部准备后样本拟合统计量。这会有意使用验证集/测试集信息，在普通实验评价中可能造成数据泄漏。

每次运行对应的统计量保存在：

```python
unit.context.standardization_info
```

当前实验流程实现的是 z-score 标准化。`standardize_method` 字段为后续扩展保留；运行时尚未执行 `minmax`。

---

## 数据输出层级

`ExperimentConfig.data_level` 决定每个 split 的对象类型：

| 值 | Split 对象 |
|---|---|
| `"raw"` | 数组/列表组成的字典。 |
| `"dataset"` | `TorchDataset`。 |
| `"dataloader"` | PyTorch `DataLoader`。 |

`data_representation` 决定 dataset 内部的数据形式：

| 值 | 行为 |
|---|---|
| `"original"` | 保持 NumPy 数组、列表或 tensor 的原始形式。 |
| `"pt"` | 将支持的数值数据转换为 PyTorch tensor。 |

---

## 评价指标

内置辅助函数包括：

### 分类

```python
from flexmm.experiment import compute_cls_metrics

metrics = compute_cls_metrics(predictions, targets)
```

返回内容：

- accuracy；
- macro F1；
- weighted F1；
- macro precision；
- macro recall；
- 可定义时的 Pearson correlation；
- confusion matrix；
- prediction 和 target 数组。

### 回归

```python
from flexmm.experiment import compute_regression_metrics

metrics = compute_regression_metrics(predictions, targets)
```

返回内容：

- MAE；
- MSE；
- RMSE；
- 可定义时的 Pearson correlation；
- prediction 和 target 数组。

`ExperimentManager.get_result()` 同时支持 NumPy 数组和 PyTorch tensor；当分类预测为 score/logit 数组时，会自动执行 `argmax`。

---

## 信息处理工具

`flexmm.info_utils` 可在数据准备前对 `data_dicts` 进行轻量分析：

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

其他工具还支持 turn index、相邻 reference value 和基于区间的索引分组。

---

## 保存的产物

典型实验目录如下：

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

`Data.pkl` 和 `Info.pkl` 使用 Python pickle。请仅加载来自可信来源的 pickle 文件。

---

## 项目结构

最小化的包结构如下：

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

## 当前适用范围

FlexMM 当前专注于实验准备和实验编排。请注意以下边界：

- 模型定义和优化流程由用户提供；
- 配置中使用的特征值必须在数据准备阶段可访问；
- 分类指标面向单标签分类；
- 分层划分要求标量目标；
- 划分过程保留输入顺序，而不会自动随机打乱；
- 当前仅实现 z-score 标准化；
- PyTorch 集成为可选项，且不会自动将 DataLoader 的完整 dataset 移入 GPU。

这些限制是明确的扩展点，而不是隐式行为。

---

## 详细文档

完整流程请参阅 [`docs/WORKFLOW_zh-CN.md`](docs/WORKFLOW_zh-CN.md)，内容包括：

- 完整的数据准备和实验生命周期；
- 不同索引空间之间的转换；
- 详细的数据划分语义；
- 序列窗口行为；
- 目标值处理；
- 标准化规则；
- 扩展点与常见错误。

---

## 贡献

欢迎贡献代码。提交 pull request 前，请：

1. 为行为变更增加或更新测试；
2. 保持公开 docstring 与实现一致；
3. 避免引入 validation/test 数据泄漏；
4. 记录兼容性或序列化格式变化；
5. 除非显式引入并固定随机种子，否则请维持确定性行为。

---

## 引用
