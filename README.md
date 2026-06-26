# FlexMM

Language: **English** | [中文](docs/README_zh-CN.md) | [日本語](docs/README_ja.md)

**FlexMM** is a model-agnostic Python framework for sequence-aware multimodal data preparation and reproducible experiment orchestration.

It provides a consistent path from an ordered list of sample dictionaries to:

- aligned multimodal input and target arrays;
- configurable temporal context windows;
- classification and regression target metadata;
- group-independent, group-dependent, or unconstrained train/validation/test splits;
- input-modality combination experiments across folds;
- raw dictionaries, PyTorch datasets, or PyTorch data loaders;
- per-run context, normalization statistics, metrics, and result storage.

FlexMM deliberately does **not** prescribe a model architecture or training loop. You keep control of model construction, optimization, early stopping, and logging, while FlexMM handles the repetitive and error-prone experiment setup around them.

---

## Why FlexMM?

Multimodal experiments often repeat the same infrastructure work:

1. align modalities and targets;
2. construct temporal windows;
3. split by speaker, participant, group, or session;
4. prevent sequence leakage across splits;
5. repeat experiments for every modality combination and fold;
6. carry fold-specific metadata into training and evaluation;
7. save enough information to reproduce the run.

FlexMM packages these steps into three focused modules:

| Module | Responsibility |
|---|---|
| `flexmm.info_utils` | Query and organize lightweight sample information, turns, adjacency, and intervals. |
| `flexmm.data_prep` | Build aligned sequence data, process targets, create splits, and serialize prepared artifacts. |
| `flexmm.experiment` | Iterate over input combinations and folds through `ExperimentManager`, `ExperimentUnit`, and `RunContext`. |

---

## Key features

- **One ordered sample contract**: each element in `data_dicts` is one sample, frame, utterance, or event.
- **Multimodal inputs**: configure any number of input keys independently.
- **Sequence-aware preparation**: past/future context, stride, offsets, filtering, and constant/edge padding.
- **Classification and regression**: label mappings, target statistics, regression bins, and multidimensional regression targets.
- **Three split semantics**:
  - independent by speaker/group/session;
  - dependent within each reference group;
  - unconstrained by sample index.
- **Holdout, k-fold, and leave-one-out** test strategies.
- **Optional target stratification** for scalar classification and scalar regression targets.
- **Sequence-overlap protection** across test/train and optionally train/validation.
- **Input-combination experiments** generated automatically or specified explicitly.
- **Leakage-aware normalization**: fold-level normalization is fitted on training data and reused for validation/test.
- **Optional PyTorch integration** without making PyTorch a requirement for the preparation stage.
- **Re-iterable experiment management**: iterate over the same manager more than once without consuming a one-shot generator.
- **Reproducible artifacts**: saved preparation config, experiment config, split information, target mappings, and index mappings.

---

## Installation

FlexMM requires Python 3.9 or later.

From the repository root:

```bash
pip install -e .
```

Core runtime dependencies are:

```bash
pip install numpy scipy scikit-learn
```

PyTorch is optional and is only required for `data_level="dataset"`, `data_level="dataloader"`, or explicit tensor conversion:

```bash
pip install torch
```

> Before publishing, declare these dependencies in your `pyproject.toml` or equivalent packaging file so users normally need only `pip install -e .` or `pip install flexmm`.

---

## Data model

FlexMM expects an **ordered list of dictionaries**. List position is the original sample index used for alignment, sequence construction, and split bookkeeping.

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

A `sample_id` is strongly recommended for auditing and external references, although the framework's internal alignment is based on the original list index.

Typical dictionary fields include:

- identifiers: `sample_id`, `speaker`, `participant`, `group`, `session`;
- targets: `label`, `score`, `valence`, `arousal`;
- model inputs: `audio`, `video`, `text`, sensor vectors, handcrafted features;
- descriptive information: timestamps, conditions, source paths, trial IDs.

Configured numeric values should have compatible shapes across samples. Values are collected into NumPy arrays when possible; heterogeneous or nonnumeric values remain Python lists.

> The current preparation API expects configured values to be available from `data_dicts` at preparation time. For datasets too large to keep in memory, store compact references and add a project-specific loading layer, or extend the gathering interface with a feature-store abstraction.

---

## Quick start

### 1. Configure and prepare data

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

This creates aligned prepared data and metadata. With saving enabled, the output directory contains:

```text
ExperimentStore/demo/
├── Data.pkl
├── Info.pkl
└── DataPrepConfig.json
```

### 2. Create an experiment manager

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

With two input keys and `generate_input_comb=True`, FlexMM generates:

```text
[audio]
[text]
[audio, text]
```

for every prepared fold.

### 3. Run your own training loop

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

Move individual batches to the model device inside the training loop:

```python
batch = {
    key: value.to(device) if isinstance(value, torch.Tensor) else value
    for key, value in batch.items()
}
```

The manager keeps dataset tensors on CPU when creating data loaders, which is safer for memory use and multiprocessing.

---

## End-to-end mode without a separate preparation call

The manager can run data preparation internally:

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

Use the two-stage workflow when prepared data will be reused across many model runs. Use end-to-end mode for compact scripts or one-off experiments.

---

## Experiment objects

### `ExperimentManager`

Owns shared experiment state. It loads or prepares data, seeds configured random-number systems, validates required keys, saves `ExpConfig.json`, and creates a fresh iterator over all input-combination/fold conditions.

### `ExperimentUnit`

Represents one executable condition:

```python
unit.data       # {"train": ..., "valid": ..., "test": ...}
unit.context    # RunContext
unit.input_comb # compatibility property
unit.fold       # compatibility property
```

### `RunContext`

Carries run-specific information into model construction, training, evaluation, and result saving:

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

`split_indexes` contains original `data_dicts` indexes. `prepared_split_indexes` contains positions in `collected_data` after sequence filtering and preparation.

---

## Split strategies

### Independent split

```python
split_dependency="independent"
```

Reference values identified by `split_ref_key` are separated between test and non-test data. This is appropriate for speaker-independent, participant-independent, group-independent, or session-independent evaluation.

Validation can be selected by:

```python
independent_split_valid_by="ref_key"  # validation groups are also independent
independent_split_valid_by="index"    # validation samples may share train groups
```

### Dependent split

```python
split_dependency="dependent"
independent_split_valid_by=None
```

Each reference group contributes samples to train, validation, and test. This is useful when the intended evaluation permits the same participant or group in all sets.

### Unconstrained split

```python
split_dependency="none"
independent_split_valid_by=None
```

Samples are split without reference-group constraints.

### Test modes

All split strategies support:

```python
split_mode="holdout"
split_mode="kfold"
split_mode="leave_one_out"
```

Splits are deterministic and preserve input/reference order. FlexMM does not silently shuffle before splitting. Arrange data intentionally, use explicit split overrides, or perform reproducible ordering before preparation when random partitioning is required.

---

## Sequence construction

Each input or target config can specify:

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

For example:

```python
InputConfig(
    keys="audio",
    seq_len_before=2,
    seq_len_after=1,
    stride=1,
    seq_padding=True,
)
```

constructs windows conceptually equivalent to:

```text
[t-2, t-1, t, t+1]
```

Sequence boundaries are controlled by `DataPrepConfig`:

```python
seq_group_mode="ref_key"  # consecutive runs defined by seq_group_key/split_ref_key
seq_group_key="speaker"
```

or:

```python
seq_group_mode="index"    # treat the complete ordered dataset as one range
```

Custom half-open ranges are also supported:

```python
seq_ranges_custom=[(0, 100), (150, 220)]
```

When sequence windows cross split boundaries, overlap removal can discard lower-priority anchors:

```python
remove_test_train_overlap_range=True
remove_train_valid_overlap_range=False
remove_overlap_priority=["test", "train", "valid"]
```

The default priority preserves test samples first, then train, then validation.

---

## Targets

### Classification

```python
ClassificationTargetConfig(
    keys="label",
    convert_target_to_id=True,
)
```

Preparation records:

```python
info_dict["target_info"]["label"]["target2id"]
info_dict["target_info"]["label"]["id2target"]
info_dict["target_info"]["label"]["target_stats"]
info_dict["target_info"]["label"]["target2indexes"]
```

Classification targets are expected to be scalar or scalar-like values.

### Regression

```python
from flexmm.data_prep import RegressionTargetConfig

RegressionTargetConfig(
    keys="score",
    stratified_bin_num=10,
    convert_target_to_bin=False,
)
```

Scalar regression targets can be binned for stratified splitting. Multidimensional regression targets are supported with:

```python
RegressionTargetConfig(
    keys="trajectory",
    is_multi_dim=True,
)
```

Multidimensional targets cannot be used for stratified splitting.

---

## Standardization and leakage

Enable normalization per input config:

```python
InputConfig(
    keys="audio",
    standardize_data=True,
    standardize_scope="split",
)
```

- `standardize_scope="split"`: fit mean and standard deviation on the current fold's **training set**, then apply them to train, validation, and test. This is the recommended setting.
- `standardize_scope="all"`: fit statistics on all prepared samples. This intentionally uses validation/test information and may cause data leakage.

Run-specific statistics are available in:

```python
unit.context.standardization_info
```

The current experiment pipeline implements z-score normalization. The `standardize_method` configuration field is reserved for future extension; `minmax` is not yet applied by the runtime pipeline.

---

## Data levels

`ExperimentConfig.data_level` controls what each split contains:

| Value | Split object |
|---|---|
| `"raw"` | Dictionary of arrays/lists. |
| `"dataset"` | `TorchDataset`. |
| `"dataloader"` | PyTorch `DataLoader`. |

`data_representation` controls values inside a dataset:

| Value | Behavior |
|---|---|
| `"original"` | Keep NumPy arrays/lists/tensors unchanged. |
| `"pt"` | Convert supported numeric values to PyTorch tensors. |

---

## Metrics

Built-in helpers include:

### Classification

```python
from flexmm.experiment import compute_cls_metrics

metrics = compute_cls_metrics(predictions, targets)
```

Returned fields:

- accuracy;
- macro F1;
- weighted F1;
- macro precision;
- macro recall;
- Pearson correlation when defined;
- confusion matrix;
- prediction and target arrays.

### Regression

```python
from flexmm.experiment import compute_regression_metrics

metrics = compute_regression_metrics(predictions, targets)
```

Returned fields:

- MAE;
- MSE;
- RMSE;
- Pearson correlation when defined;
- prediction and target arrays.

`ExperimentManager.get_result()` also accepts NumPy arrays or PyTorch tensors and applies classification `argmax` when score/logit arrays are supplied.

---

## Information utilities

`flexmm.info_utils` provides lightweight analysis of `data_dicts` before preparation:

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

Additional utilities cover turn indexes, adjacent reference values, and interval-based index grouping.

---

## Saved artifacts

A typical experiment directory becomes:

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

`Data.pkl` and `Info.pkl` use Python pickle. Only load pickle files from trusted sources.

---

## Project structure

A minimal package layout is:

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

## Current scope

FlexMM currently focuses on experiment preparation and orchestration. Keep these boundaries in mind:

- model definition and optimization are user-provided;
- configured feature values must be accessible during preparation;
- classification metrics target single-label classification;
- stratification requires scalar targets;
- split construction preserves input order rather than randomly shuffling;
- z-score is the currently implemented normalization method;
- PyTorch integration is optional and does not move DataLoader datasets to GPU automatically.

These boundaries are intentional places for extension rather than hidden behavior.

---

## Detailed documentation

See [`docs/WORKFLOW.md`](docs/WORKFLOW.md) for:

- the complete preparation and experiment lifecycle;
- index-space translation;
- detailed split semantics;
- sequence-window behavior;
- target processing;
- standardization rules;
- extension points and common pitfalls.

---

## Contributing

Contributions are welcome. Before opening a pull request:

1. add or update tests for behavior changes;
2. keep public docstrings synchronized with the implementation;
3. avoid introducing validation/test leakage;
4. document compatibility or serialization changes;
5. preserve deterministic behavior unless randomness is explicit and seeded.

---

## Citation

Citation metadata has not been supplied yet. Before the first public release, add a `CITATION.cff` file and replace this section with the preferred software citation.

---

## License

Add the selected open-source license as `LICENSE` before public release, and update this section with its name.
