# FlexMM API Reference

This document is generated from the source-code docstrings of FlexMM. It is intended as a GitHub-friendly reference for users who need to inspect configuration classes, experiment objects, data-preparation helpers, evaluation functions, and information utilities.

- **Package:** `flexmm`
- **Repository:** https://github.com/Xia-code/flexmm
- **Generated:** 2026-06-26
- **Python:** 3.9 or later

> Public classes, methods, and functions are shown first.
> Names beginning with an underscore are implementation details and are placed
> in collapsible sections. Their behavior may change more easily between releases.

## Contents

1. [Package overview](#package-overview)
2. [`flexmm.data_prep`](#flexmmdata_prep)
3. [`flexmm.experiment`](#flexmmexperiment)
4. [`flexmm.info_utils`](#flexmminfo_utils)

## Package overview

FlexMM exposes three primary modules for multimodal data preparation and experiment orchestration.

| Module | Main responsibility |
|---|---|
| `flexmm.data_prep` | Sequence-aware multimodal data collection, target processing, splitting, overlap prevention, and serialization. |
| `flexmm.experiment` | Experiment configuration, modality-combination iteration, PyTorch conversion, metrics, context management, and result loading. |
| `flexmm.info_utils` | Queries and grouping operations for sample-level metadata, turns, reference values, adjacency, and intervals. |

### Typical imports

```python
from flexmm import data_prep, experiment, info_utils

from flexmm.data_prep import DataPrepConfig, DataPreparator, InputConfig
from flexmm.experiment import ExperimentConfig, ExperimentManager
```

### Package initializer docstring

```text
20250328
__init__ file of flexmm
```

---

## `flexmm.data_prep`

[View source on GitHub](https://github.com/Xia-code/flexmm/blob/main/flexmm/data_prep.py)

Prepare sequence-aware multimodal data, targets, and train/validation/test splits.

The module defines configuration objects, data-gathering utilities, target
statistics and conversion helpers, deterministic split strategies, sequence
overlap removal, and serialization helpers. Sample-level descriptive data are
expected as a list of dictionaries whose indexes identify the original samples.

### API summary

#### Classes

| Class | Summary |
|---|---|
| [`BaseConfig`](#baseconfig) | Common configuration fields for one input or target data key group.  |
| [`ClassificationTargetConfig`](#classificationtargetconfig) | Configure one or more scalar classification targets.  |
| [`RegressionTargetConfig`](#regressiontargetconfig) | Configure one or more regression targets and optional stratification bins.  |
| [`InputConfig`](#inputconfig) | Configure one or more model-input data keys.  |
| [`DataPrepConfig`](#dataprepconfig) | Configure sequence construction, target processing, splitting, and persistence.  |
| [`DataPreparator`](#datapreparator) | Run the complete data-preparation pipeline for sample-level dictionaries.  |

#### Public functions

| Function | Summary |
|---|---|
| `pick_data_by_indexes()` | Select sample dictionaries by original index and record each source index.  |
| `gather_data_single_key()` | Gather all sequence windows for one data key.  |
| `gather_data_by_indexes()` | Gather one sequence of values, applying constant or edge padding.  |
| `shift_get_seq_indexes()` | Construct strided context windows around filtered anchor indexes.  |
| `get_strided_seq()` | Extract a strided window around one position in an index list.  |
| `data_split_independent()` | Split reference groups independently so test groups never appear in train or validation.  |
| `data_split_dependent()` | Split samples within every reference group so groups may appear in all sets.  |
| `data_split_unconstrained()` | Split sample indexes without enforcing reference-group independence.  |
| `remove_overlapped_seq_split()` | Remove lower-priority anchors whose sequence contents overlap protected splits.  |
| `get_target_info_cls()` | Build class statistics, class-ID mappings, and original-index groups.  |
| `get_target_info_regression()` | Bin scalar regression targets and collect bin statistics and indexes.  |
| `get_target2indexes()` | Map each class or regression bin to its original sample indexes.  |
| `get_stratified_bin_info()` | Calculate effective regression-bin width and count.  |
| `generate_config_template()` | Write a JSON configuration template for specified data keys.  |
| `load_config()` | Load a JSON configuration file into configuration objects.  |
| `make_config_from_json()` | Reconstruct nested configuration objects from parsed JSON data.  |
| `save_data()` | Save prepared data and metadata using explicit or configured options.  |
| `load_data()` | Load prepared data, metadata, and configuration from a directory.  |
| `get_target_list()` | Collect target values from selected original sample indexes.  |
| `get_scalar_target_list()` | Collect and normalize selected target values to Python scalars.  |
| `convert_to_python_scalar()` | Convert a scalar-like Python, NumPy, or PyTorch value to a Python scalar.  |
| `calculate_miu_sigma()` | Calculate feature-wise mean and standard deviation over selected samples.  |

### Classes

### `BaseConfig`

```python
class BaseConfig
```

Common configuration fields for one input or target data key group.

#### Notes

Instances store validated settings or pipeline state used by this module.

#### Declared fields

| Field | Type | Default |
|---|---|---|
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
<summary><strong>Internal methods (1)</strong></summary>

##### `BaseConfig.__post_init__`

```python
def BaseConfig.__post_init__(self)
```

Normalize derived fields and validate the configuration after initialization.

</details>

### `ClassificationTargetConfig`

```python
class ClassificationTargetConfig(BaseConfig)
```

Configure one or more scalar classification targets.

#### Notes

Instances store validated settings or pipeline state used by this module.

#### Declared fields

| Field | Type | Default |
|---|---|---|
| `convert_target_to_id` | `bool` | `False` |

<details>
<summary><strong>Internal methods (1)</strong></summary>

##### `ClassificationTargetConfig.__post_init__`

```python
def ClassificationTargetConfig.__post_init__(self)
```

Normalize derived fields and validate the configuration after initialization.

</details>

### `RegressionTargetConfig`

```python
class RegressionTargetConfig(BaseConfig)
```

Configure one or more regression targets and optional stratification bins.

#### Notes

Instances store validated settings or pipeline state used by this module.

#### Declared fields

| Field | Type | Default |
|---|---|---|
| `is_multi_dim` | `bool` | `False` |
| `convert_target_to_bin` | `bool` | `False` |
| `stratified_bin_size` | `Union[float, int, None]` | `None` |
| `stratified_bin_num` | `Optional[int]` | `10` |
| `bin_closed_side` | `Literal['upper', 'lower']` | `'lower'` |

<details>
<summary><strong>Internal methods (1)</strong></summary>

##### `RegressionTargetConfig.__post_init__`

```python
def RegressionTargetConfig.__post_init__(self)
```

Normalize derived fields and validate the configuration after initialization.

</details>

### `InputConfig`

```python
class InputConfig(BaseConfig)
```

Configure one or more model-input data keys.

#### Notes

Instances store validated settings or pipeline state used by this module.

#### Declared fields

| Field | Type | Default |
|---|---|---|
| `is_non_numeric` | `bool` | `False` |

<details>
<summary><strong>Internal methods (1)</strong></summary>

##### `InputConfig.__post_init__`

```python
def InputConfig.__post_init__(self)
```

Normalize derived fields and validate the configuration after initialization.

</details>

### `DataPrepConfig`

```python
class DataPrepConfig
```

Configure sequence construction, target processing, splitting, and persistence.

#### Notes

Instances store validated settings or pipeline state used by this module.

#### Declared fields

| Field | Type | Default |
|---|---|---|
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

#### Public methods

##### `DataPrepConfig.assert_config`

```python
def DataPrepConfig.assert_config(self)
```

Validate split-related configuration values and resolve compatible defaults.

##### `DataPrepConfig.to_json`

```python
def DataPrepConfig.to_json(self)
```

Convert nested configuration objects into a JSON-serializable structure.

###### Returns

list
    Serialized configuration entries containing class names and dataclass fields.

<details>
<summary><strong>Internal methods (3)</strong></summary>

##### `DataPrepConfig.__post_init__`

```python
def DataPrepConfig.__post_init__(self)
```

Normalize derived fields and validate the configuration after initialization.

##### `DataPrepConfig._init_keys_and_check`

```python
def DataPrepConfig._init_keys_and_check(self, configs, config_name='input')
```

Collect keys for one config category and reject duplicated keys.

###### Parameters

configs : Any
    Data configuration objects to inspect.
config_name : Any
    Configuration category to collect, usually ``"input"`` or ``"target"``.

##### `DataPrepConfig._make_seq_info`

```python
def DataPrepConfig._make_seq_info(self) -> (Dict, int, int)
```

Build per-key sequence-window settings.

</details>

### `DataPreparator`

```python
class DataPreparator
```

Run the complete data-preparation pipeline for sample-level dictionaries.

#### Notes

Instances store validated settings or pipeline state used by this module.

#### Constructor and protocol methods

##### `DataPreparator.__init__`

```python
def DataPreparator.__init__(self, data_dicts, data_prep_config, split_postprocess_fn=None)
```

Initialize the data preparator and derive sequence-related index metadata.

###### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
data_prep_config : Any
    Configuration object saved with the prepared data.
split_postprocess_fn : Any
    Optional callable receiving index folds and reference-value folds after built-in processing.

#### Public methods

##### `DataPreparator.run`

```python
def DataPreparator.run(self)
```

Execute sequence gathering, target processing, splitting, and optional saving.

###### Returns

tuple
    ``(collected_data, info_dict)`` produced by the complete pipeline.

##### `DataPreparator.get_seq_indexes`

```python
def DataPreparator.get_seq_indexes(self)
```

Build sequence windows for every configured input and target key.

##### `DataPreparator.gather_data`

```python
def DataPreparator.gather_data(self)
```

Gather configured data fields into aligned arrays or Python lists.

##### `DataPreparator.process_target`

```python
def DataPreparator.process_target(self)
```

Create target statistics and apply requested target conversions.

##### `DataPreparator.convert_target_to_scalar`

```python
def DataPreparator.convert_target_to_scalar(self)
```

Create scalar target lists for statistics and splitting.

##### `DataPreparator.make_target_info_dict`

```python
def DataPreparator.make_target_info_dict(self)
```

Build label mappings, regression bins, statistics, and index groups.

###### Returns

dict
    Target metadata keyed by target name.

##### `DataPreparator.convert_target_form`

```python
def DataPreparator.convert_target_form(self)
```

Convert collected targets to class IDs or regression-bin representatives.

##### `DataPreparator.split_data`

```python
def DataPreparator.split_data(self)
```

Create train, validation, and test folds according to the configured strategy.

##### `DataPreparator.post_split_process`

```python
def DataPreparator.post_split_process(self)
```

Remove sequence anchors that cause prohibited cross-split overlap.

##### `DataPreparator.make_info_dict`

```python
def DataPreparator.make_info_dict(self)
```

Assemble preparation metadata for saving and downstream experiments.

##### `DataPreparator.get_zscore_miu_sigma`

```python
def DataPreparator.get_zscore_miu_sigma(self)
```

Calculate per-fold normalization statistics for configured input keys.

###### Returns

list[dict]
    Normalization statistics for every fold and split.

##### `DataPreparator.get_input_shapes`

```python
def DataPreparator.get_input_shapes(self)
```

Infer model-input shapes from the collected data.

###### Returns

dict
    Mapping from input keys to inferred feature shapes.

##### `DataPreparator.save_data`

```python
def DataPreparator.save_data(self, save_dir=None, overwrite_data=None)
```

Save prepared data and metadata using explicit or configured options.

###### Parameters

save_dir : Any
    Destination directory. ``None`` uses the configured store directory.
overwrite_data : Any
    Whether existing serialized data may be replaced.

<details>
<summary><strong>Internal methods (6)</strong></summary>

##### `DataPreparator._init_seq_indexes_info`

```python
def DataPreparator._init_seq_indexes_info(self)
```

Initialize sequence ranges, used indexes, and original-index mappings.

##### `DataPreparator._init_seq_ranges`

```python
def DataPreparator._init_seq_ranges(self)
```

Create half-open sequence ranges from groups or custom boundaries.

###### Returns

list[tuple[int, int]]
    Sorted half-open sequence ranges.

##### `DataPreparator._validate_seq_ranges`

```python
def DataPreparator._validate_seq_ranges(self, ranges)
```

Validate custom half-open sequence ranges against the dataset length.

###### Parameters

ranges : Any
    Half-open ``(start, end)`` index ranges.

##### `DataPreparator._merge_ranges`

```python
def DataPreparator._merge_ranges(self, ranges)
```

Merge overlapping or touching half-open ranges.

###### Parameters

ranges : Any
    Half-open ``(start, end)`` index ranges.

###### Returns

list[tuple[int, int]]
    Merged half-open ranges.

##### `DataPreparator._get_used_indexes_from_ranges`

```python
def DataPreparator._get_used_indexes_from_ranges(self)
```

Collect unique sample indexes covered by the sequence ranges.

###### Returns

list[int]
    Unique covered original sample indexes.

##### `DataPreparator._get_pure_shape`

```python
def DataPreparator._get_pure_shape(self, data)
```

Infer the feature shape after removing batch and sequence dimensions.

###### Parameters

data : Any
    Value used by ``_get_pure_shape``.

###### Returns

tuple
    Feature dimensions after batch and sequence dimensions.

</details>

### Public functions

#### `pick_data_by_indexes`

```python
def pick_data_by_indexes(data_dicts: List[Dict], used_indexes: List) -> List
```

Select sample dictionaries by original index and record each source index.

##### Parameters

data_dicts : List[Dict]
    Ordered sample-level dictionaries indexed by original sample position.
used_indexes : List
    Original sample indexes to process. ``None`` selects all samples.

##### Returns

list[dict]
    Selected sample dictionaries.

#### `gather_data_single_key`

```python
def gather_data_single_key(data_dicts: List[Dict], data_key: Any, seq_indexes: List, dtype=None, seq_padding_index: int=-1, seq_padding_mode: Literal['constant', 'edge']='constant', seq_padding_value: Any=0.0, squeeze_singleton_dims: bool=True, keep_batch_seq_dims: bool=True) -> np.array
```

Gather all sequence windows for one data key.

##### Parameters

data_dicts : List[Dict]
    Ordered sample-level dictionaries indexed by original sample position.
data_key : Any
    Dictionary key whose values are gathered.
seq_indexes : List
    Pairs of anchor indexes and their sequence index lists.
dtype : Any
    Optional NumPy dtype used for numeric output.
seq_padding_index : int
    Sentinel index representing a padded position.
seq_padding_mode : Literal['constant', 'edge']
    Padding strategy: constant values or nearest-edge repetition.
seq_padding_value : Any
    Value used by constant padding.
squeeze_singleton_dims : bool
    Whether singleton dimensions are removed.
keep_batch_seq_dims : bool
    Whether numeric outputs retain explicit batch and sequence dimensions.

##### Returns

numpy.ndarray or list
    Gathered values for all sequence anchors.

#### `gather_data_by_indexes`

```python
def gather_data_by_indexes(data_dicts, data_key, used_indexes=None, sample_data=None, data_operation='array', dtype=None, seq_padding_index=-1, seq_padding_mode: Literal['constant', 'edge']='constant', seq_padding_value=0, squeeze_singleton_dims=False)
```

Gather one sequence of values, applying constant or edge padding.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
data_key : Any
    Dictionary key whose values are gathered.
used_indexes : Any
    Original sample indexes to process. ``None`` selects all samples.
sample_data : Any
    Optional representative value used to infer type and shape.
data_operation : Any
    Internal gathering mode for arrays, tensors, nested lists, or scalars.
dtype : Any
    Optional NumPy dtype used for numeric output.
seq_padding_index : Any
    Sentinel index representing a padded position.
seq_padding_mode : Literal['constant', 'edge']
    Padding strategy: constant values or nearest-edge repetition.
seq_padding_value : Any
    Value used by constant padding.
squeeze_singleton_dims : Any
    Whether singleton dimensions are removed.

##### Returns

numpy.ndarray or list
    Gathered values for one sequence.

#### `shift_get_seq_indexes`

```python
def shift_get_seq_indexes(index_list: List, seq_len_before: int, seq_len_after: int, step_offset: int=0, stride: int=1, seq_pos_from_start: int=0, seq_pos_from_end: int=0, padding: bool=True, padding_index: int=-1) -> List[Tuple[int, List]]
```

Construct strided context windows around filtered anchor indexes.

##### Parameters

index_list : List
    Ordered original sample indexes or generic index-like values.
seq_len_before : int
    Number of context steps before each anchor.
seq_len_after : int
    Number of context steps after each anchor.
step_offset : int
    Relative offset associated with the configured data key.
stride : int
    Distance between neighboring context positions.
seq_pos_from_start : int
    Number of candidate anchors excluded from the range start.
seq_pos_from_end : int
    Number of candidate anchors excluded from the range end.
padding : bool
    Whether incomplete boundary windows are padded.
padding_index : int
    Sentinel index inserted for padded positions.

##### Returns

tuple[list, list]
    Sequence tuples and their anchor indexes.

#### `get_strided_seq`

```python
def get_strided_seq(index_list, i, stride, seq_len_before, seq_len_after)
```

Extract a strided window around one position in an index list.

##### Parameters

index_list : Any
    Ordered original sample indexes or generic index-like values.
i : Any
    Center position within ``index_list``.
stride : Any
    Distance between neighboring context positions.
seq_len_before : Any
    Number of context steps before each anchor.
seq_len_after : Any
    Number of context steps after each anchor.

##### Returns

list[int]
    Valid positions in the requested strided window.

#### `data_split_independent`

```python
def data_split_independent(data_dicts: List[Dict], split_ref_key: Any, split_mode: Literal['holdout', 'kfold', 'leave_one_out']='kfold', folds: int=5, train_valid_ratio: float=0.9, holdout_test_ratio: float=0.2, use_stratified_split: bool=False, split_valid_by: Literal['index', 'ref_key']='index', focused_target_key: Any='label', is_focused_key_multi_dim: bool=False, target2indexes: Optional[Dict]=None, focused_target_task_type: Literal['c', 'r']='c', stratified_bin_num: Optional[int]=None, stratified_bin_size: Optional[float]=None, used_indexes: Optional[List[int]]=None, train_ref_values_override: Optional[Union[Mapping, Sequence]]=None, valid_ref_values_override: Optional[Union[Mapping, Sequence]]=None, test_ref_values_override: Optional[Union[Mapping, Sequence]]=None) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[Any]]]]
```

Split reference groups independently so test groups never appear in train or validation.

##### Parameters

data_dicts : List[Dict]
    Ordered sample-level dictionaries indexed by original sample position.
split_ref_key : Any
    Dictionary key defining groups such as speakers, participants, or sessions.
split_mode : Literal['holdout', 'kfold', 'leave_one_out']
    Test-split strategy: holdout, k-fold, or leave-one-out.
folds : int
    Requested number of folds for k-fold splitting.
train_valid_ratio : float
    Fraction of non-test data assigned to training.
holdout_test_ratio : float
    Fraction assigned to test in holdout mode.
use_stratified_split : bool
    Whether scalar targets are used to preserve target proportions.
split_valid_by : Literal['index', 'ref_key']
    Whether independent validation data are separated by indexes or reference groups.
focused_target_key : Any
    Target key used for stratification.
is_focused_key_multi_dim : bool
    Whether the focused target contains more than one value per sample.
target2indexes : Optional[Dict]
    Optional mapping from targets or bins to original sample indexes.
focused_target_task_type : Literal['c', 'r']
    Focused target type: ``"c"`` for classification or ``"r"`` for regression.
stratified_bin_num : Optional[int]
    Number of regression bins used for stratification.
stratified_bin_size : Optional[float]
    Width of regression bins used for stratification.
used_indexes : Optional[List[int]]
    Original sample indexes to process. ``None`` selects all samples.
train_ref_values_override : Optional[Union[Mapping, Sequence]]
    Optional train reference groups, shared or specified per fold.
valid_ref_values_override : Optional[Union[Mapping, Sequence]]
    Optional validation reference groups, shared or specified per fold.
test_ref_values_override : Optional[Union[Mapping, Sequence]]
    Optional test reference groups specified for one or more folds.

##### Returns

tuple[dict, dict]
    Index folds and corresponding reference-value folds.

#### `data_split_dependent`

```python
def data_split_dependent(data_dicts: List[Dict], split_ref_key: Any, split_mode: Literal['holdout', 'kfold', 'leave_one_out']='kfold', folds: int=5, train_valid_ratio: float=0.9, holdout_test_ratio: float=0.2, use_stratified_split: bool=False, split_valid_by: Literal['index', 'ref_key']='index', focused_target_key: Any='label', is_focused_key_multi_dim: bool=False, target2indexes: Optional[Dict]=None, focused_target_task_type: Literal['c', 'r']='c', stratified_bin_num: Optional[int]=None, stratified_bin_size: Optional[float]=None, used_indexes: Optional[List[int]]=None) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[Any]]]]
```

Split samples within every reference group so groups may appear in all sets.

##### Parameters

data_dicts : List[Dict]
    Ordered sample-level dictionaries indexed by original sample position.
split_ref_key : Any
    Dictionary key defining groups such as speakers, participants, or sessions.
split_mode : Literal['holdout', 'kfold', 'leave_one_out']
    Test-split strategy: holdout, k-fold, or leave-one-out.
folds : int
    Requested number of folds for k-fold splitting.
train_valid_ratio : float
    Fraction of non-test data assigned to training.
holdout_test_ratio : float
    Fraction assigned to test in holdout mode.
use_stratified_split : bool
    Whether scalar targets are used to preserve target proportions.
split_valid_by : Literal['index', 'ref_key']
    Whether independent validation data are separated by indexes or reference groups.
focused_target_key : Any
    Target key used for stratification.
is_focused_key_multi_dim : bool
    Whether the focused target contains more than one value per sample.
target2indexes : Optional[Dict]
    Optional mapping from targets or bins to original sample indexes.
focused_target_task_type : Literal['c', 'r']
    Focused target type: ``"c"`` for classification or ``"r"`` for regression.
stratified_bin_num : Optional[int]
    Number of regression bins used for stratification.
stratified_bin_size : Optional[float]
    Width of regression bins used for stratification.
used_indexes : Optional[List[int]]
    Original sample indexes to process. ``None`` selects all samples.

##### Returns

tuple[dict, dict]
    Index folds and corresponding reference-value folds.

#### `data_split_unconstrained`

```python
def data_split_unconstrained(data_dicts: List[Dict], split_ref_key: Any=None, split_mode: Literal['holdout', 'kfold', 'leave_one_out']='kfold', folds: int=5, train_valid_ratio: float=0.9, holdout_test_ratio: float=0.2, use_stratified_split: bool=False, split_valid_by: Literal['index', 'ref_key']='index', target2indexes: Optional[Dict]=None, focused_target_key: Any='label', is_focused_key_multi_dim: bool=False, focused_target_task_type: Literal['c', 'r']='c', stratified_bin_num: Optional[int]=None, stratified_bin_size: Optional[float]=None, used_indexes: Optional[List[int]]=None) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[Any]]]]
```

Split sample indexes without enforcing reference-group independence.

##### Parameters

data_dicts : List[Dict]
    Ordered sample-level dictionaries indexed by original sample position.
split_ref_key : Any
    Dictionary key defining groups such as speakers, participants, or sessions.
split_mode : Literal['holdout', 'kfold', 'leave_one_out']
    Test-split strategy: holdout, k-fold, or leave-one-out.
folds : int
    Requested number of folds for k-fold splitting.
train_valid_ratio : float
    Fraction of non-test data assigned to training.
holdout_test_ratio : float
    Fraction assigned to test in holdout mode.
use_stratified_split : bool
    Whether scalar targets are used to preserve target proportions.
split_valid_by : Literal['index', 'ref_key']
    Whether independent validation data are separated by indexes or reference groups.
target2indexes : Optional[Dict]
    Optional mapping from targets or bins to original sample indexes.
focused_target_key : Any
    Target key used for stratification.
is_focused_key_multi_dim : bool
    Whether the focused target contains more than one value per sample.
focused_target_task_type : Literal['c', 'r']
    Focused target type: ``"c"`` for classification or ``"r"`` for regression.
stratified_bin_num : Optional[int]
    Number of regression bins used for stratification.
stratified_bin_size : Optional[float]
    Width of regression bins used for stratification.
used_indexes : Optional[List[int]]
    Original sample indexes to process. ``None`` selects all samples.

##### Returns

tuple[dict, dict]
    Index folds and corresponding reference-value folds.

#### `remove_overlapped_seq_split`

```python
def remove_overlapped_seq_split(data_dicts, index_split_folds, anchore_index2seq_indexes, *, split_ref_key='group', is_test_train_no_seq_overlap: bool=True, is_train_valid_no_seq_overlap: bool=True, priority_order=('test', 'train', 'valid'), padding_index: int=-1)
```

Remove lower-priority anchors whose sequence contents overlap protected splits.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
index_split_folds : Any
    Per-fold train, validation, and test original indexes.
anchore_index2seq_indexes : Any
    Mapping from each anchor to all original indexes used by its sequence.
split_ref_key : Any
    Dictionary key defining groups such as speakers, participants, or sessions.
is_test_train_no_seq_overlap : bool
    Whether train/test sequence overlap is prohibited.
is_train_valid_no_seq_overlap : bool
    Whether train/validation sequence overlap is prohibited.
priority_order : Any
    Highest-to-lowest split priority used when removing overlap.
padding_index : int
    Sentinel index inserted for padded positions.

##### Returns

tuple[dict, dict]
    Filtered index folds and rebuilt reference-value folds.

#### `get_target_info_cls`

```python
def get_target_info_cls(data_dicts, target_ref_key, used_indexes=None)
```

Build class statistics, class-ID mappings, and original-index groups.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
target_ref_key : Any
    Dictionary key containing the target value.
used_indexes : Any
    Original sample indexes to process. ``None`` selects all samples.

##### Returns

tuple
    Class statistics, class mappings, and target-to-index mapping.

#### `get_target_info_regression`

```python
def get_target_info_regression(data_dicts, target_ref_key, used_indexes=None, stratified_bin_size=None, stratified_bin_num=None, bin_closed_side: Literal['upper', 'lower']='lower')
```

Bin scalar regression targets and collect bin statistics and indexes.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
target_ref_key : Any
    Dictionary key containing the target value.
used_indexes : Any
    Original sample indexes to process. ``None`` selects all samples.
stratified_bin_size : Any
    Width of regression bins used for stratification.
stratified_bin_num : Any
    Number of regression bins used for stratification.
bin_closed_side : Literal['upper', 'lower']
    Boundary convention for regression bins.

##### Returns

tuple
    Bin statistics, bin ranges, and bin-to-index mapping.

#### `get_target2indexes`

```python
def get_target2indexes(data_dicts, target_ref_key, used_indexes=None, task_type='c', stratified_bin_size: Optional[float]=None, stratified_bin_num: Optional[int]=10)
```

Map each class or regression bin to its original sample indexes.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
target_ref_key : Any
    Dictionary key containing the target value.
used_indexes : Any
    Original sample indexes to process. ``None`` selects all samples.
task_type : Any
    Target type: classification or regression.
stratified_bin_size : Optional[float]
    Width of regression bins used for stratification.
stratified_bin_num : Optional[int]
    Number of regression bins used for stratification.

##### Returns

dict
    Mapping from class values or regression bins to original indexes.

#### `get_stratified_bin_info`

```python
def get_stratified_bin_info(data_dicts, target_ref_key, used_indexes=None, bin_size=None, bin_num=None, target_list=None)
```

Calculate effective regression-bin width and count.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
target_ref_key : Any
    Dictionary key containing the target value.
used_indexes : Any
    Original sample indexes to process. ``None`` selects all samples.
bin_size : Any
    Requested regression-bin width.
bin_num : Any
    Requested number of regression bins.
target_list : Any
    Optional pre-collected scalar targets.

##### Returns

tuple[float, int]
    Effective bin width and number of bins.

#### `generate_config_template`

```python
def generate_config_template(file: str='./data_prep_config.json', cls_target_keys: Union[List, str, None]=None, regression_target_keys: Union[List, str, None]=None, input_keys: Union[List, str, None]=None)
```

Write a JSON configuration template for specified data keys.

##### Parameters

file : str
    Path to the JSON configuration file.
cls_target_keys : Union[List, str, None]
    Classification target key or keys.
regression_target_keys : Union[List, str, None]
    Regression target key or keys.
input_keys : Union[List, str, None]
    Model-input key or keys.

#### `load_config`

```python
def load_config(file: str='./data_prep_config.json')
```

Load a JSON configuration file into configuration objects.

##### Parameters

file : str
    Path to the JSON configuration file.

##### Returns

DataPrepConfig
    Reconstructed data-preparation configuration.

#### `make_config_from_json`

```python
def make_config_from_json(config_json)
```

Reconstruct nested configuration objects from parsed JSON data.

##### Parameters

config_json : Any
    Parsed configuration representation produced by ``to_json``.

##### Returns

DataPrepConfig
    Reconstructed data-preparation configuration.

#### `save_data`

```python
def save_data(collected_data, info_dict, data_prep_config, save_dir='./DataExperiment', overwrite_data=False)
```

Save prepared data and metadata using explicit or configured options.

##### Parameters

collected_data : Any
    Prepared input and target arrays or lists.
info_dict : Any
    Preparation metadata and split information.
data_prep_config : Any
    Configuration object saved with the prepared data.
save_dir : Any
    Destination directory. ``None`` uses the configured store directory.
overwrite_data : Any
    Whether existing serialized data may be replaced.

#### `load_data`

```python
def load_data(data_dir='./DataExperiment')
```

Load prepared data, metadata, and configuration from a directory.

##### Parameters

data_dir : Any
    Directory containing serialized prepared-data files.

##### Returns

tuple
    Prepared data, metadata, and configuration.

#### `get_target_list`

```python
def get_target_list(data_dicts, target_ref_key, used_indexes=None)
```

Collect target values from selected original sample indexes.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
target_ref_key : Any
    Dictionary key containing the target value.
used_indexes : Any
    Original sample indexes to process. ``None`` selects all samples.

##### Returns

list
    Target values in selected-index order.

#### `get_scalar_target_list`

```python
def get_scalar_target_list(data_dicts, target_ref_key, used_indexes=None)
```

Collect and normalize selected target values to Python scalars.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
target_ref_key : Any
    Dictionary key containing the target value.
used_indexes : Any
    Original sample indexes to process. ``None`` selects all samples.

##### Returns

list
    Python scalar targets in selected-index order.

#### `convert_to_python_scalar`

```python
def convert_to_python_scalar(target_value)
```

Convert a scalar-like Python, NumPy, or PyTorch value to a Python scalar.

##### Parameters

target_value : Any
    Scalar or one-element scalar-like value to convert.

##### Returns

object
    Equivalent Python scalar value.

#### `calculate_miu_sigma`

```python
def calculate_miu_sigma(data_dicts, ref_key, index_list)
```

Calculate feature-wise mean and standard deviation over selected samples.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
ref_key : Any
    Dictionary key containing values used in a calculation.
index_list : Any
    Ordered original sample indexes or generic index-like values.

##### Returns

tuple[numpy.ndarray, numpy.ndarray]
    Feature-wise mean and standard deviation.

<details>
<summary><strong>Internal module helpers (25)</strong></summary>

These functions support the public API and are included because they contain source docstrings. They are not the preferred compatibility surface for user code.

#### `_get_test_indexes`

```python
def _get_test_indexes(indexes_list, current_split_group, num_split_group, one_fold_test_ratio, split_mode='kfold')
```

Select the test indexes for one fold and split mode.

##### Parameters

indexes_list : Any
    Ordered indexes eligible for the current test selection.
current_split_group : Any
    Zero-based fold position.
num_split_group : Any
    Total number of split groups.
one_fold_test_ratio : Any
    Holdout ratio used when ``split_mode="holdout"``.
split_mode : Any
    Test-split strategy: holdout, k-fold, or leave-one-out.

##### Returns

list[int]
    Indexes assigned to the requested test fold.

#### `_distribute_indexes_to_folds`

```python
def _distribute_indexes_to_folds(index_list, current_group_index)
```

Assign at most one index to a requested fold position.

##### Parameters

index_list : Any
    Ordered original sample indexes or generic index-like values.
current_group_index : Any
    Zero-based group position.

##### Returns

list[int]
    A one-item list or an empty list.

#### `_summary_seq_index_list`

```python
def _summary_seq_index_list(anchore_index_list, anchore_index2seq_index)
```

Collect unique sequence indexes referenced by a set of anchors.

##### Parameters

anchore_index_list : Any
    Anchor indexes whose sequence contents are summarized.
anchore_index2seq_index : Any
    Mapping from anchor indexes to sequence index lists.

##### Returns

list[int]
    Sorted unique sequence indexes.

#### `_validate_split_parameters`

```python
def _validate_split_parameters(split_mode, folds, train_valid_ratio, holdout_test_ratio)
```

Validate common split modes, fold counts, and ratios.

##### Parameters

split_mode : Any
    Test-split strategy: holdout, k-fold, or leave-one-out.
folds : Any
    Requested number of folds for k-fold splitting.
train_valid_ratio : Any
    Fraction of non-test data assigned to training.
holdout_test_ratio : Any
    Fraction assigned to test in holdout mode.

#### `_normalize_used_indexes`

```python
def _normalize_used_indexes(data_dicts, used_indexes)
```

Resolve and validate the original sample indexes used for preparation.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
used_indexes : Any
    Original sample indexes to process. ``None`` selects all samples.

##### Returns

list[int]
    Validated original sample indexes.

#### `_balanced_chunks`

```python
def _balanced_chunks(values, num_chunks, reverse=False)
```

Partition ordered values into nearly equal chunks.

##### Parameters

values : Any
    Values to process.
num_chunks : Any
    Number of chunks to create.
reverse : Any
    Whether chunk order is reversed after partitioning.

##### Returns

list[list]
    Nearly equal ordered chunks.

#### `_get_holdout_size`

```python
def _get_holdout_size(num_items, ratio)
```

Calculate a nonempty holdout size while retaining training data when possible.

##### Parameters

num_items : Any
    Number of available items.
ratio : Any
    Requested fraction.

##### Returns

int
    Number of items assigned to holdout test data.

#### `_split_train_valid_indexes`

```python
def _split_train_valid_indexes(indexes, train_ratio)
```

Split ordered indexes into nonempty train and validation subsets when possible.

##### Parameters

indexes : Any
    Ordered candidate indexes.
train_ratio : Any
    Fraction assigned to training.

##### Returns

tuple[list, list]
    Training and validation index lists.

#### `_stratified_train_valid_split`

```python
def _stratified_train_valid_split(indexes, target2indexes, train_ratio)
```

Split candidate indexes within each target stratum.

##### Parameters

indexes : Any
    Ordered candidate indexes.
target2indexes : Any
    Optional mapping from targets or bins to original sample indexes.
train_ratio : Any
    Fraction assigned to training.

##### Returns

tuple[list, list]
    Stratified training and validation index lists.

#### `_prepare_target2indexes`

```python
def _prepare_target2indexes(data_dicts, target_key, used_indexes, task_type, target2indexes, stratified_bin_size, stratified_bin_num)
```

Create or filter a target-to-original-index mapping.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
target_key : Any
    Dictionary key containing target values.
used_indexes : Any
    Original sample indexes to process. ``None`` selects all samples.
task_type : Any
    Target type: classification or regression.
target2indexes : Any
    Optional mapping from targets or bins to original sample indexes.
stratified_bin_size : Any
    Width of regression bins used for stratification.
stratified_bin_num : Any
    Number of regression bins used for stratification.

##### Returns

dict
    Target or bin mapping limited to selected indexes.

#### `_normalize_fold_override`

```python
def _normalize_fold_override(override)
```

Normalize split overrides into a list of per-fold value lists.

##### Parameters

override : Any
    User-provided split values in shared or per-fold form.

##### Returns

list[list]
    Per-fold override values.

#### `_get_fold_override`

```python
def _get_fold_override(override, fold)
```

Read an override for one fold, allowing one shared override.

##### Parameters

override : Any
    User-provided split values in shared or per-fold form.
fold : Any
    Zero-based fold identifier.

##### Returns

list or None
    Values assigned to the requested fold.

#### `_resolve_train_valid_ref_values`

```python
def _resolve_train_valid_ref_values(candidate_values, train_ratio, train_override, valid_override)
```

Resolve automatic or user-defined train and validation reference groups.

##### Parameters

candidate_values : Any
    Reference values not assigned to test.
train_ratio : Any
    Fraction assigned to training.
train_override : Any
    Optional explicitly assigned training reference values.
valid_override : Any
    Optional explicitly assigned validation reference values.

##### Returns

tuple[list, list]
    Training and validation reference values.

#### `_validate_ref_values`

```python
def _validate_ref_values(values, available_values, split_name)
```

Ensure override reference values exist in the available groups.

##### Parameters

values : Any
    Values to process.
available_values : Any
    Reference values available for assignment.
split_name : Any
    Human-readable split name used in error messages.

#### `_build_ref_value_split_dict`

```python
def _build_ref_value_split_dict(data_dicts, split_dict, split_ref_key)
```

Derive unique reference values for each split from original indexes.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
split_dict : Any
    One train/validation/test index dictionary.
split_ref_key : Any
    Dictionary key defining groups such as speakers, participants, or sessions.

##### Returns

dict
    Unique reference values for train, validation, and test.

#### `_build_ref_value_split_folds`

```python
def _build_ref_value_split_folds(data_dicts, index_split_folds, split_ref_key)
```

Derive reference-value summaries for every index fold.

##### Parameters

data_dicts : Any
    Ordered sample-level dictionaries indexed by original sample position.
index_split_folds : Any
    Per-fold train, validation, and test original indexes.
split_ref_key : Any
    Dictionary key defining groups such as speakers, participants, or sessions.

##### Returns

dict
    Reference-value summaries keyed by fold.

#### `_normalize_index_split_folds`

```python
def _normalize_index_split_folds(index_split_folds)
```

Normalize list- or mapping-based folds to a consistent integer-keyed mapping.

##### Parameters

index_split_folds : Any
    Per-fold train, validation, and test original indexes.

##### Returns

dict
    Integer-keyed fold mapping with standard split keys.

#### `_validate_split_dict`

```python
def _validate_split_dict(split_dict, allowed_indexes)
```

Ensure one fold is disjoint, complete, and limited to allowed indexes.

##### Parameters

split_dict : Any
    One train/validation/test index dictionary.
allowed_indexes : Any
    Complete set of indexes that must be assigned exactly once.

#### `_flatten`

```python
def _flatten(iterables)
```

Flatten one level of nested iterables.

##### Parameters

iterables : Any
    Nested iterables to flatten by one level.

##### Returns

list
    One-level flattened values.

#### `_deduplicate_preserve_order`

```python
def _deduplicate_preserve_order(values)
```

Remove repeated values while preserving first-occurrence order.

##### Parameters

values : Any
    Values to process.

##### Returns

list
    Unique values in first-occurrence order.

#### `_make_regression_bin_ranges`

```python
def _make_regression_bin_ranges(values, bin_size=None, bin_num=None)
```

Create contiguous bin boundaries spanning observed target values.

##### Parameters

values : Any
    Values to process.
bin_size : Any
    Requested regression-bin width.
bin_num : Any
    Requested number of regression bins.

##### Returns

list[tuple[float, float]]
    Contiguous lower/upper bin boundaries.

#### `_find_bin_id`

```python
def _find_bin_id(value, bin_ranges, bin_closed_side)
```

Find the bin containing one scalar regression target.

##### Parameters

value : Any
    Scalar regression target.
bin_ranges : Any
    Mapping from bin IDs to lower and upper boundaries.
bin_closed_side : Any
    Boundary convention for regression bins.

##### Returns

int
    Identifier of the containing regression bin.

#### `_convert_value_to_bin`

```python
def _convert_value_to_bin(value, bin_ranges, bin_closed_side)
```

Convert a scalar target to the configured boundary representative of its bin.

##### Parameters

value : Any
    Scalar regression target.
bin_ranges : Any
    Mapping from bin IDs to lower and upper boundaries.
bin_closed_side : Any
    Boundary convention for regression bins.

##### Returns

float
    Selected lower or upper boundary representing the target bin.

#### `_init_split_set_dict`

```python
def _init_split_set_dict()
```

Create an empty train/validation/test dictionary.

##### Returns

dict[str, list]
    Empty lists for ``train``, ``valid``, and ``test``.

#### `_unify_to_list`

```python
def _unify_to_list(content)
```

Normalize a scalar, tuple, or NumPy array into a Python list.

##### Parameters

content : Any
    Scalar or collection to normalize into a list.

##### Returns

list
    Normalized Python list.

</details>

---

## `flexmm.experiment`

[View source on GitHub](https://github.com/Xia-code/flexmm/blob/main/flexmm/experiment.py)

Experiment configuration, iteration, data conversion, and evaluation tools.

The module separates experiment orchestration into three concepts:

``ExperimentManager``
    Loads or prepares shared data and creates a fresh experiment iterator each
    time it is iterated.
``ExperimentUnit``
    Contains the train, validation, and test data for one input-combination and
    fold condition.
``RunContext``
    Carries the metadata needed to execute and save that individual condition.

Prepared-data split indexes are expected to refer to original ``data_dicts``
indexes. Before indexing ``collected_data``, the manager converts them to
prepared-data positions through ``info_dict['ori_index2id']`` when available.

### API summary

#### Classes

| Class | Summary |
|---|---|
| [`ExperimentConfig`](#experimentconfig) | Configure experiment combinations, data output, and reproducibility.  |
| [`TorchExperimentConfig`](#torchexperimentconfig) | Extend :class:`ExperimentConfig` with PyTorch DataLoader settings.  |
| [`ExperimentContext`](#experimentcontext) | Carry runtime metadata for one input-combination and fold condition.  |
| [`ExperimentUnit`](#experimentunit) | Represent the data and context of one executable experiment condition.  |
| [`ExperimentManager`](#experimentmanager) | Prepare shared experiment state and iterate over experiment units.  |
| [`ExperimentResultLoader`](#experimentresultloader) | Load saved experiment configuration, preparation config, and results.  |

#### Public functions

| Function | Summary |
|---|---|
| `generate_key_combs()` | Generate every non-empty combination of the given keys.  |
| `load_prepared_data()` | Load data, preparation information, and preparation configuration.  |
| `iter_experiment_units()` | Yield one :class:`ExperimentUnit` per input combination and fold.  |
| `make_data_generator()` | Create an experiment-unit generator.  |
| `make_data_geneartor()` | Deprecated misspelled alias of :func:`make_data_generator`. |
| `collect_combination_data()` | Select specified prepared-data positions for a group of keys.  |
| `perform_zscore()` | Standardize numeric data with zero-variance protection.  |
| `get_input_target_shapes()` | Collect configured input and target shapes available in a dataset.  |
| `make_dataset()` | Create a dictionary-backed PyTorch dataset.  |
| `convert_single_data_to_tensor()` | Convert supported split data to PyTorch tensors.  |
| `make_dataloader()` | Create a PyTorch DataLoader from a non-empty dataset.  |
| `torch_postprocess()` | Convert tensors or array-like values to NumPy evaluation arrays.  |
| `load_exp_config()` | Load a saved experiment configuration JSON file. |
| `make_config_from_json()` | Construct an experiment configuration from saved JSON content.  |
| `compute_cls_metrics()` | Calculate common single-label classification metrics.  |
| `compute_regression_metrics()` | Calculate MAE, MSE, RMSE, and Pearson correlation for regression. |
| `compute_target_key_average_metric()` | Average a metric nested below one target key. |
| `compute_average_metric()` | Average a top-level metric across result dictionaries. |
| `change_average_summary_form()` | Convert average summaries to a selected analysis-friendly format. |
| `change_average_summary_lists()` | Convert combination summaries to parallel combination/metric lists. |

### Classes

### `ExperimentConfig`

```python
class ExperimentConfig
```

Configure experiment combinations, data output, and reproducibility.

#### Parameters

experiment_input_keys : Any or list[Any]
    Candidate input keys used by the experiments.
generate_input_comb : bool, default=True
    If ``True``, generate every non-empty combination of the candidate
    input keys. If ``False``, use all candidate keys as one combination.
experiment_target_keys : Any or list[Any]
    Target keys included in every experiment unit.
input_comb_custom : list[Any or list[Any]], optional
    Explicit input combinations. When supplied, this value overrides
    ``experiment_input_keys`` and ``generate_input_comb``.
input_key_abbr : dict or list, optional
    Short labels used to construct combination directory names. A list is
    aligned with the resolved input-key order; a dictionary must contain
    exactly the resolved input keys.
random_seed : int, optional
    Base random seed. A seed is generated when omitted.
random_seed_scope : list[str]
    Random-number systems to seed. Supported values are ``"random"``,
    ``"numpy"``, and ``"torch"``.
data_level : {"raw", "dataset", "dataloader"}
    Representation yielded for each split.
data_representation : {"original", "pt"}
    Data representation inside a dataset. ``"pt"`` converts supported
    arrays to PyTorch tensors.
load_prepared_data : bool, default=True
    Load prepared data from ``store_dir`` instead of running data
    preparation from ``data_dicts``.
store_dir : str
    Root directory for prepared data, experiment configuration, and result
    files.
debug_flag : int
    User-controlled debugging flag carried in the configuration.

#### Attributes

input_keys : list[Any]
    Resolved unique input keys.
input_combs : list[list[Any]]
    Resolved input combinations.
target_keys : list[Any]
    Resolved target keys.

#### Declared fields

| Field | Type | Default |
|---|---|---|
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

#### Public methods

##### `ExperimentConfig.assert_config`

```python
def ExperimentConfig.assert_config(self) -> None
```

Validate general experiment settings.

###### Raises

ValueError
    If a categorical setting or random-seed scope is invalid.

##### `ExperimentConfig.to_json`

```python
def ExperimentConfig.to_json(self) -> tuple[str, Dict[str, Any]]
```

Convert the dataclass fields to the serializable config format.

###### Returns

tuple[str, dict]
    Class name and dataclass field dictionary.

<details>
<summary><strong>Internal methods (2)</strong></summary>

##### `ExperimentConfig.__post_init__`

```python
def ExperimentConfig.__post_init__(self) -> None
```

Normalize configuration fields and validate the resolved settings.

##### `ExperimentConfig._normalize_input_key_abbr`

```python
def ExperimentConfig._normalize_input_key_abbr(self, abbreviations: Optional[Union[Dict[Any, Any], List[Any]]]) -> Dict[Any, Any]
```

Normalize input-key abbreviations to a complete dictionary.

###### Parameters

abbreviations : dict, list, or None
    User-supplied abbreviations.

###### Returns

dict
    Mapping from every resolved input key to its abbreviation.

</details>

### `TorchExperimentConfig`

```python
class TorchExperimentConfig(ExperimentConfig)
```

Extend :class:`ExperimentConfig` with PyTorch DataLoader settings.

#### Parameters

train_batch_size, valid_batch_size, test_batch_size : int
    Batch sizes used for the corresponding splits.
shuffle_train_data, shuffle_valid_data, shuffle_test_data : bool
    Whether the corresponding DataLoader shuffles its dataset.
use_gpu : bool
    Whether tensor-conversion helpers may place tensors on CUDA. Moving
    tensors to the model device inside the training loop is generally more
    flexible, so the default dataset workflow keeps tensors on CPU.

#### Declared fields

| Field | Type | Default |
|---|---|---|
| `train_batch_size` | `int` | `4` |
| `valid_batch_size` | `int` | `1` |
| `test_batch_size` | `int` | `1` |
| `shuffle_train_data` | `bool` | `True` |
| `shuffle_valid_data` | `bool` | `False` |
| `shuffle_test_data` | `bool` | `False` |
| `use_gpu` | `bool` | `USE_GPU_DEFAULT` |

<details>
<summary><strong>Internal methods (1)</strong></summary>

##### `TorchExperimentConfig.__post_init__`

```python
def TorchExperimentConfig.__post_init__(self) -> None
```

Initialize inherited fields and validate PyTorch-specific values.

</details>

### `ExperimentContext`

```python
class ExperimentContext
```

Carry runtime metadata for one input-combination and fold condition.

#### Parameters

fold : Any
    Fold identifier stored in the prepared split information.
comb_index : int
    Zero-based position of the input combination in the experiment plan.
comb_name : str
    File-system-safe name for the input combination.
input_comb : list[Any]
    Input keys used by this condition.
target_keys : list[Any]
    Target keys included in this condition.
split_indexes : dict[str, list[int]]
    Original ``data_dicts`` indexes assigned to each split.
prepared_split_indexes : dict[str, list[int]]
    Corresponding positions used to index ``collected_data``.
ref_value_splits : dict[str, list[Any]]
    Split reference values, such as speaker or group identifiers.
standardization_info : dict
    Mean, standard deviation, and scope used for each standardized input.
info_dict : dict
    Shared information produced during data preparation.
exp_config : ExperimentConfig
    Experiment configuration.
data_prep_config : Any
    Data-preparation configuration.
output_dir : str
    Recommended output directory for this condition.
seed : int
    Base experiment seed.
user_extras : dict
    Additional user-defined runtime information.

#### Declared fields

| Field | Type | Default |
|---|---|---|
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

#### Public methods

##### `ExperimentContext.as_dict`

```python
def ExperimentContext.as_dict(self) -> Dict[str, Any]
```

Return a shallow dictionary representation of the context.

###### Returns

dict
    Context fields keyed by field name.

### `ExperimentUnit`

```python
class ExperimentUnit
```

Represent the data and context of one executable experiment condition.

#### Parameters

data : dict[str, Any]
    Train, validation, and test data in the configured data level.
context : RunContext
    Runtime information associated with the condition.

#### Declared fields

| Field | Type | Default |
|---|---|---|
| `data` | `Dict[str, Any]` | `_required_` |
| `context` | `ExperimentContext` | `_required_` |

#### Public methods

##### `ExperimentUnit.input_comb`

```python
def ExperimentUnit.input_comb(self) -> List[Any]
```

Return the unit's input-key combination.

##### `ExperimentUnit.fold`

```python
def ExperimentUnit.fold(self) -> Any
```

Return the unit's fold identifier.

##### `ExperimentUnit.info_dict`

```python
def ExperimentUnit.info_dict(self) -> Dict[str, Any]
```

Return shared data-preparation information for compatibility.

##### `ExperimentUnit.as_dict`

```python
def ExperimentUnit.as_dict(self) -> Dict[str, Any]
```

Return the experiment unit as a dictionary.

###### Returns

dict
    Dictionary containing ``data`` and ``context``.

### `ExperimentManager`

```python
class ExperimentManager
```

Prepare shared experiment state and iterate over experiment units.

The manager is re-iterable: every call to ``iter(manager)`` creates a new
generator over all input-combination and fold conditions.

#### Parameters

exp_config : ExperimentConfig
    Experiment-level configuration.
data_dicts : list[dict], optional
    Source sample information and data. Required when
    ``exp_config.load_prepared_data`` is ``False``.
data_prep_config : Any, optional
    Configuration passed to ``data_prep.DataPreparator``. Required when
    prepared data is not loaded.
user_extras : dict, optional
    Additional information copied into every :class:`RunContext`.

#### Constructor and protocol methods

##### `ExperimentManager.__init__`

```python
def ExperimentManager.__init__(self, exp_config: ExperimentConfig, data_dicts: Optional[List[Dict[str, Any]]]=None, data_prep_config: Any=None, user_extras: Optional[Dict[str, Any]]=None) -> None
```

Initialize the manager without loading or preparing data yet.

##### `ExperimentManager.__iter__`

```python
def ExperimentManager.__iter__(self) -> Iterator[ExperimentUnit]
```

Return a fresh iterator over all configured experiment units.

#### Public methods

##### `ExperimentManager.setup`

```python
def ExperimentManager.setup(self) -> 'ExperimentManager'
```

Load or prepare shared data and save the experiment configuration.

###### Returns

ExperimentManager
    The initialized, iterable manager itself.

##### `ExperimentManager.init_random_seed`

```python
def ExperimentManager.init_random_seed(self, random_seed: int) -> None
```

Seed configured random-number systems.

###### Parameters

random_seed : int
    Seed applied to the configured systems.

##### `ExperimentManager.get_prepared_data`

```python
def ExperimentManager.get_prepared_data(self) -> tuple[Dict[str, Any], Dict[str, Any], Any]
```

Load prepared data or execute the data-preparation pipeline.

###### Returns

collected_data : dict
    Prepared arrays or lists keyed by modality and target.
info_dict : dict
    Split, target, shape, and index-mapping information.
data_prep_config : Any
    Configuration used to prepare the data.

##### `ExperimentManager.get_result`

```python
def ExperimentManager.get_result(self, pred: Any, true: Any, task_type: TaskType='c') -> Dict[str, Any]
```

Post-process predictions and calculate task metrics.

###### Parameters

pred, true : Any
    Prediction and target arrays or tensors.
task_type : {"c", "r"}
    Classification or regression task type.

###### Returns

dict
    Metric dictionary.

##### `ExperimentManager.torch_postprocess`

```python
def ExperimentManager.torch_postprocess(self, tensor: Any, *, task_type: TaskType='c')
```

Wrapper for post-process tensor. Convert tensors or array-like values to NumPy evaluation arrays.

###### Parameters

tensor : Any
    PyTorch tensor, NumPy array, or array-like value.
task_type : {"c", "r"}
    Classification or regression task type.

###### Returns

dict
    Metric dictionary.

##### `ExperimentManager.compute_result`

```python
def ExperimentManager.compute_result(pred: Any, true: Any, task_type: TaskType='c') -> Dict[str, Any]
```

Calculate classification or regression metrics.

##### `ExperimentManager.compute_average_result`

```python
def ExperimentManager.compute_average_result(result_dicts: Sequence[Mapping[str, Any]], target_key: Any, metric_key: str) -> Any
```

Average a nested target metric across result dictionaries.

##### `ExperimentManager.save_exp_config`

```python
def ExperimentManager.save_exp_config(self) -> None
```

Save the experiment configuration under ``store_dir``.

##### `ExperimentManager.save_result`

```python
def ExperimentManager.save_result(self, result: Any, context: Optional[ExperimentContext]=None, file_name: str='ExpResult.pkl') -> str
```

Save a result globally or under one run's output directory.

###### Parameters

result : Any
    Pickle-serializable result object.
context : RunContext, optional
    When supplied, save below ``context.output_dir``. Otherwise save
    below the manager's ``store_dir``.
file_name : str
    Output file name.

###### Returns

str
    Saved file path.

<details>
<summary><strong>Internal methods (1)</strong></summary>

##### `ExperimentManager._validate_prepared_data`

```python
def ExperimentManager._validate_prepared_data(self) -> None
```

Validate keys and split metadata required by the experiment plan.

</details>

### `ExperimentResultLoader`

```python
class ExperimentResultLoader
```

Load saved experiment configuration, preparation config, and results.

#### Parameters

result_dir : str
    Directory containing the saved files.
exp_config_file, data_prep_config_file, result_file : str
    File names relative to ``result_dir``.

#### Constructor and protocol methods

##### `ExperimentResultLoader.__init__`

```python
def ExperimentResultLoader.__init__(self, result_dir: str, exp_config_file: str='ExpConfig.json', data_prep_config_file: str='DataPrepConfig.json', result_file: str='ExpResult.pkl') -> None
```

Load all requested experiment artifacts.

### Public functions

#### `generate_key_combs`

```python
def generate_key_combs(keys: Sequence[Any]) -> List[List[Any]]
```

Generate every non-empty combination of the given keys.

##### Parameters

keys : sequence
    Input keys in the desired combination order.

##### Returns

list[list]
    Combinations ordered first by size and then by input order.

#### `load_prepared_data`

```python
def load_prepared_data(data_dir: str='./DataExperiment') -> tuple[Dict[str, Any], Dict[str, Any], Any]
```

Load data, preparation information, and preparation configuration.

##### Parameters

data_dir : str
    Directory created by ``data_prep.save_data``.

##### Returns

collected_data, info_dict, data_prep_config : tuple
    Objects loaded through ``data_prep.load_data``.

##### Raises

FileNotFoundError
    If the directory or required files do not exist.

#### `iter_experiment_units`

```python
def iter_experiment_units(collected_data: Dict[str, Any], info_dict: Dict[str, Any], exp_config: ExperimentConfig, data_prep_config: Any, user_extras: Optional[Dict[str, Any]]=None) -> Iterator[ExperimentUnit]
```

Yield one :class:`ExperimentUnit` per input combination and fold.

Standardization with ``standardize_scope='split'`` uses training data from
the current fold and applies the same statistics to train, validation, and
test data. This avoids validation/test leakage. ``'all'`` uses all prepared
samples and therefore should only be selected deliberately.

##### Parameters

collected_data : dict
    Prepared modality, target, and index data.
info_dict : dict
    Data-preparation information including split folds and index mappings.
exp_config : ExperimentConfig
    Experiment configuration.
data_prep_config : Any
    Preparation configuration containing ``key2config``.
user_extras : dict, optional
    Additional information copied into each run context.

##### Yields

ExperimentUnit
    Data and runtime information for one experiment condition.

#### `make_data_generator`

```python
def make_data_generator(collected_data: Dict[str, Any], info_dict: Dict[str, Any], exp_config: ExperimentConfig, data_prep_config: Any, data_level: Optional[DataLevel]=None, data_representation: Optional[DataRepresentation]=None) -> Iterator[ExperimentUnit]
```

Create an experiment-unit generator.

This compatibility function delegates to :func:`iter_experiment_units`.
Optional data-level arguments are applied to a shallow configuration copy
so the caller's configuration is not modified.

#### `make_data_geneartor`

```python
def make_data_geneartor(*args: Any, **kwargs: Any) -> Iterator[ExperimentUnit]
```

Deprecated misspelled alias of :func:`make_data_generator`.

#### `collect_combination_data`

```python
def collect_combination_data(collected_data: Mapping[Any, Any], keys: Sequence[Any], index_list: Sequence[int]) -> Dict[Any, Any]
```

Select specified prepared-data positions for a group of keys.

##### Parameters

collected_data : mapping
    Prepared data keyed by input, target, or index key.
keys : sequence
    Keys to include in the result.
index_list : sequence[int]
    Positions in the prepared data arrays/lists.

##### Returns

dict
    Selected data keyed by ``keys``.

#### `perform_zscore`

```python
def perform_zscore(data_array: Any, axis: Union[int, tuple[int, ...]]=0) -> np.ndarray
```

Standardize numeric data with zero-variance protection.

##### Parameters

data_array : array-like
    Numeric data.
axis : int or tuple[int, ...]
    Axes used to calculate the mean and standard deviation.

##### Returns

numpy.ndarray
    Standardized data.

#### `get_input_target_shapes`

```python
def get_input_target_shapes(datasets: Mapping[str, Any], info_dict: Mapping[str, Any]) -> Dict[Any, tuple[int, ...]]
```

Collect configured input and target shapes available in a dataset.

##### Parameters

datasets : mapping
    Split datasets or raw split dictionaries containing ``"train"``.
info_dict : mapping
    Data-preparation information containing ``input_shapes`` and
    ``target_info``.

##### Returns

dict
    Shape information keyed by input or target name.

#### `make_dataset`

```python
def make_dataset(single_condition_data: Dict[Any, Any], data_representation: DataRepresentation='original', use_gpu: bool=False) -> TorchDataset
```

Create a dictionary-backed PyTorch dataset.

##### Parameters

single_condition_data : dict
    Data for one train, validation, or test split.
data_representation : {"original", "pt"}
    Keep values unchanged or convert supported values to tensors.
use_gpu : bool
    Place converted tensors on CUDA. CPU tensors are recommended when the
    dataset will be wrapped by a DataLoader.

##### Returns

TorchDataset
    Created dataset.

#### `convert_single_data_to_tensor`

```python
def convert_single_data_to_tensor(single_condition_data: Mapping[Any, Any], *, exclude_keys: Optional[Iterable[Any]]=None, use_gpu: bool=False) -> Dict[Any, Any]
```

Convert supported split data to PyTorch tensors.

Index metadata is converted to ``torch.long`` when possible. Unsupported
Python objects are kept unchanged.

##### Parameters

single_condition_data : mapping
    Split data keyed by modality, target, or metadata key.
exclude_keys : iterable, optional
    Keys copied without conversion.
use_gpu : bool
    Move converted tensors to CUDA when available.

##### Returns

dict
    Tensor-converted data.

#### `make_dataloader`

```python
def make_dataloader(dataset: TorchDataset, batch_size: int, shuffle: bool) -> Any
```

Create a PyTorch DataLoader from a non-empty dataset.

##### Parameters

dataset : TorchDataset
    Dataset to batch.
batch_size : int
    Positive number of samples per batch.
shuffle : bool
    Whether to shuffle sample order.

##### Returns

torch.utils.data.DataLoader
    Created DataLoader.

#### `torch_postprocess`

```python
def torch_postprocess(tensor: Any, *, mode: Literal['raw', 'argmax']='raw', use_gpu: Optional[bool]=None) -> np.ndarray
```

Convert tensors or array-like values to NumPy evaluation arrays.

##### Parameters

tensor : Any
    PyTorch tensor, NumPy array, or array-like value.
mode : {"raw", "argmax"}
    Return raw values or take the final-axis argmax first.
use_gpu : bool, optional
    Deprecated compatibility argument. Device handling is inferred from the
    tensor itself.

##### Returns

numpy.ndarray
    Detached CPU array.

#### `load_exp_config`

```python
def load_exp_config(file: str='./ExpConfig.json') -> ExperimentConfig
```

Load a saved experiment configuration JSON file.

#### `make_config_from_json`

```python
def make_config_from_json(config_json: Sequence[Any]) -> ExperimentConfig
```

Construct an experiment configuration from saved JSON content.

##### Parameters

config_json : sequence
    Two-element sequence containing class name and constructor fields.

##### Returns

ExperimentConfig
    Reconstructed configuration object.

#### `compute_cls_metrics`

```python
def compute_cls_metrics(pred_list: Any, true_list: Any) -> Dict[str, Any]
```

Calculate common single-label classification metrics.

Pearson correlation is returned as ``nan`` when it is undefined, such as
for constant or fewer-than-two-element arrays.

#### `compute_regression_metrics`

```python
def compute_regression_metrics(pred_list: Any, true_list: Any) -> Dict[str, Any]
```

Calculate MAE, MSE, RMSE, and Pearson correlation for regression.

#### `compute_target_key_average_metric`

```python
def compute_target_key_average_metric(result_dicts: Sequence[Mapping[Any, Any]], target_key: Any, metric_key: str) -> Any
```

Average a metric nested below one target key.

#### `compute_average_metric`

```python
def compute_average_metric(result_dicts: Sequence[Mapping[str, Any]], metric_key: str) -> Any
```

Average a top-level metric across result dictionaries.

#### `change_average_summary_form`

```python
def change_average_summary_form(average_summary: Mapping[str, Any], form: Literal['lists']='lists') -> Dict[str, Any]
```

Convert average summaries to a selected analysis-friendly format.

#### `change_average_summary_lists`

```python
def change_average_summary_lists(average_summary: Mapping[str, Any]) -> Dict[str, Any]
```

Convert combination summaries to parallel combination/metric lists.

<details>
<summary><strong>Internal module helpers (17)</strong></summary>

These functions support the public API and are included because they contain source docstrings. They are not the preferred compatibility surface for user code.

#### `_make_dataset_tensor`

```python
def _make_dataset_tensor(single_condition_data: Dict[Any, Any], use_gpu: bool=False) -> TorchDataset
```

Convert split data to tensors and wrap it in :class:`TorchDataset`.

#### `_get_fold_average_summary`

```python
def _get_fold_average_summary(result_dir: Optional[str]=None, result_root: Optional[str]=None, experiment_name: Optional[str]=None, combs: Optional[Union[str, List[str]]]=None, folds: Optional[int]=None) -> Dict[str, Any]
```

Load fold summaries and calculate per-combination averages.

Either ``result_dir`` or both ``result_root`` and ``experiment_name`` must
be supplied.

#### `_unify_to_list`

```python
def _unify_to_list(value: Any) -> List[Any]
```

Convert scalar or iterable configuration input to a plain list.

#### `_unique_in_order`

```python
def _unique_in_order(values: Iterable[Any]) -> List[Any]
```

Return unique values while preserving first-occurrence order.

#### `_normalize_fold_mapping`

```python
def _normalize_fold_mapping(folds: Any) -> Dict[Any, Dict[str, List[int]]]
```

Normalize list- or mapping-form split folds to an ordered dictionary.

#### `_normalize_optional_fold_mapping`

```python
def _normalize_optional_fold_mapping(folds: Any, fold_keys: Iterable[Any]) -> Dict[Any, Dict[str, List[Any]]]
```

Normalize optional reference-value split information.

#### `_translate_split_indexes`

```python
def _translate_split_indexes(split_indexes: Mapping[str, Sequence[int]], info_dict: Mapping[str, Any], collected_data: Mapping[Any, Any]) -> Dict[str, List[int]]
```

Translate original sample indexes to prepared-data positions.

#### `_infer_sample_count`

```python
def _infer_sample_count(collected_data: Mapping[Any, Any]) -> int
```

Infer and validate the shared first-dimension sample count.

#### `_standardize_split_data`

```python
def _standardize_split_data(split_data: Dict[str, Dict[Any, Any]], collected_data: Mapping[Any, Any], input_comb: Sequence[Any], data_prep_config: Any) -> tuple[Dict[str, Dict[Any, Any]], Dict[Any, Dict[str, Any]]]
```

Standardize configured numeric inputs without mutating shared data.

#### `_apply_standardization`

```python
def _apply_standardization(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray
```

Apply z-score normalization while mapping zero-variance axes to zero.

#### `_as_numeric_array`

```python
def _as_numeric_array(value: Any, key: Any) -> np.ndarray
```

Convert a value to a non-object numeric NumPy array.

#### `_convert_data_level`

```python
def _convert_data_level(split_data: Dict[str, Dict[Any, Any]], exp_config: ExperimentConfig) -> Dict[str, Any]
```

Convert raw split dictionaries to the requested experiment data level.

#### `_make_comb_name`

```python
def _make_comb_name(input_comb: Sequence[Any], abbreviations: Mapping[Any, Any]) -> str
```

Create a file-system-safe directory name for an input combination.

#### `_sanitize_path_component`

```python
def _sanitize_path_component(value: str) -> str
```

Replace path-unsafe characters with underscores.

#### `_require_torch`

```python
def _require_torch(caller: str) -> None
```

Raise an informative error when a PyTorch helper is unavailable.

#### `_needs_argmax`

```python
def _needs_argmax(prediction: Any) -> bool
```

Return whether a classification prediction has a class-score axis.

#### `_safe_pearson`

```python
def _safe_pearson(first: np.ndarray, second: np.ndarray) -> float
```

Calculate Pearson correlation and return ``nan`` when undefined.

</details>

---

## `flexmm.info_utils`

[View source on GitHub](https://github.com/Xia-code/flexmm/blob/main/flexmm/info_utils.py)

Utilities for querying and organizing sample-level information records.

The functions in this module expect ``data_dicts`` to be a list of dictionaries.
Each dictionary describes one sample, utterance, frame, or other sequential item.
Dictionary fields may contain identifiers, grouping variables, target values, or
other lightweight sample information.

### API summary

#### Public functions

| Function | Summary |
|---|---|
| `get_ref_value2indexes()` | Map each reference value to its corresponding sample indexes.  |
| `get_ref_value_list()` | Collect one reference value for each processed sample.  |
| `get_ref_value2another()` | Map each reference value to values from another sample field.  |
| `get_turn2ref_value_and_indexes()` | Group consecutive samples with the same reference value into turns.  |
| `get_ref_value2turn_indexes()` | Map each reference value to the sample-index lists of its turns.  |
| `get_ref_value2turns()` | Map each reference value to its zero-based turn IDs.  |
| `get_ref_value2indexes_in_turns()` | Map each reference value to indexes collected through its turns.  |
| `get_ref_value2adjacent_ref_value()` | Summarize cross-reference adjacency by sample index or turn.  |
| `get_interval_split_indexes()` | Split a sequence into approximately equal chronological intervals.  |

### Public functions

#### `get_ref_value2indexes`

```python
def get_ref_value2indexes(data_dicts: List[Dict], ref_key: Any='speaker', used_indexes: Union[List[int], None]=None) -> Dict[Any, List[int]]
```

Map each reference value to its corresponding sample indexes.

##### Parameters

data_dicts : list of dict
    Sample-level information records. Every processed dictionary must
    contain ``ref_key``.
ref_key : Any, optional
    Dictionary key whose values define the reference groups, such as a
    speaker, participant, session, or group identifier.
used_indexes : list of int or None, optional
    Subset of sample indexes to process. If ``None``, all samples are used.

##### Returns

dict
    Mapping from each value of ``ref_key`` to the indexes of samples that
    contain that value. Index order follows ``used_indexes``.

##### Raises

KeyError
    If a processed dictionary does not contain ``ref_key``.
TypeError
    If a reference value is not hashable and therefore cannot be used as a
    dictionary key.

#### `get_ref_value_list`

```python
def get_ref_value_list(data_dicts: List[Dict], ref_key: Any='speaker', used_indexes: Union[List[int], None]=None) -> List[Any]
```

Collect one reference value for each processed sample.

##### Parameters

data_dicts : list of dict
    Sample-level information records. Every processed dictionary must
    contain ``ref_key``.
ref_key : Any, optional
    Dictionary key whose value is collected from each sample.
used_indexes : list of int or None, optional
    Subset of sample indexes to process. If ``None``, all samples are used.

##### Returns

list
    Values of ``ref_key`` in sample order. Repeated values are preserved.

##### Raises

KeyError
    If a processed dictionary does not contain ``ref_key``.

#### `get_ref_value2another`

```python
def get_ref_value2another(data_dicts: List[Dict], ref_key: Any='speaker', another_key: Any='target', used_indexes: Union[List[int], None]=None, *, unique_values: bool=True) -> Dict[Any, List[Any]]
```

Map each reference value to values from another sample field.

This function is useful for checking relationships between two information
fields, such as mapping each speaker to their target values or each group to
its participant identifiers.

##### Parameters

data_dicts : list of dict
    Sample-level information records. Every processed dictionary must
    contain both ``ref_key`` and ``another_key``.
ref_key : Any, optional
    Dictionary key whose values become the keys of the returned mapping.
another_key : Any, optional
    Dictionary key whose values are collected for each reference value.
used_indexes : list of int or None, optional
    Subset of sample indexes to process. If ``None``, all samples are used.
unique_values : bool, optional
    If ``True``, equivalent repeated values are stored only once for each
    reference value. If ``False``, all values are retained, including
    repetitions. Array-like values are supported in either mode.

##### Returns

dict
    Mapping from each value of ``ref_key`` to values of ``another_key``.

##### Raises

KeyError
    If a processed dictionary lacks ``ref_key`` or ``another_key``.

#### `get_turn2ref_value_and_indexes`

```python
def get_turn2ref_value_and_indexes(data_dicts: List[Dict], ref_key: Any='speaker', used_indexes: Union[List[int], None]=None) -> Dict[int, Dict[str, Any]]
```

Group consecutive samples with the same reference value into turns.

A new turn starts when either the value of ``ref_key`` changes or the next
processed sample index is not numerically consecutive to the previous one.
The order supplied by ``used_indexes`` is preserved.

##### Parameters

data_dicts : list of dict
    Ordered sample-level records representing one dialogue, conversation,
    event, or other sequence.
ref_key : Any, optional
    Dictionary key used to determine whether consecutive samples belong to
    the same turn. This is typically a speaker or participant identifier.
used_indexes : list of int or None, optional
    Ordered subset of sample indexes to process. If ``None``, all samples
    are used.

##### Returns

dict
    Mapping from zero-based turn IDs to dictionaries with two fields:
    ``"ref_value"`` contains the turn's reference value, and ``"indexes"``
    contains the sample indexes in that turn.

##### Raises

KeyError
    If a processed dictionary does not contain ``ref_key``.

#### `get_ref_value2turn_indexes`

```python
def get_ref_value2turn_indexes(data_dicts: List[Dict], ref_key: Any='speaker', used_indexes: Union[List[int], None]=None) -> Dict[Any, List[List[int]]]
```

Map each reference value to the sample-index lists of its turns.

##### Parameters

data_dicts : list of dict
    Ordered sample-level records representing a sequence.
ref_key : Any, optional
    Dictionary key used to identify the owner of each turn.
used_indexes : list of int or None, optional
    Ordered subset of sample indexes to process. If ``None``, all samples
    are used.

##### Returns

dict
    Mapping from each reference value to a list of turns, where each turn is
    represented by its sample-index list.

#### `get_ref_value2turns`

```python
def get_ref_value2turns(data_dicts: List[Dict], ref_key: Any='speaker', used_indexes: Union[List[int], None]=None) -> Dict[Any, List[int]]
```

Map each reference value to its zero-based turn IDs.

##### Parameters

data_dicts : list of dict
    Ordered sample-level records representing a sequence.
ref_key : Any, optional
    Dictionary key used to identify the owner of each turn.
used_indexes : list of int or None, optional
    Ordered subset of sample indexes to process. If ``None``, all samples
    are used.

##### Returns

dict
    Mapping from each reference value to the IDs of its turns.

#### `get_ref_value2indexes_in_turns`

```python
def get_ref_value2indexes_in_turns(data_dicts: List[Dict], ref_key: Any='speaker', used_indexes: Union[List[int], None]=None) -> Dict[Any, List[int]]
```

Map each reference value to indexes collected through its turns.

This is the flattened counterpart of :func:`get_ref_value2turn_indexes`.
Turn boundaries are not retained in the returned value.

##### Parameters

data_dicts : list of dict
    Ordered sample-level records representing a sequence.
ref_key : Any, optional
    Dictionary key used to identify the owner of each turn.
used_indexes : list of int or None, optional
    Ordered subset of sample indexes to process. If ``None``, all samples
    are used.

##### Returns

dict
    Mapping from each reference value to all of its sample indexes, ordered
    by turn occurrence.

#### `get_ref_value2adjacent_ref_value`

```python
def get_ref_value2adjacent_ref_value(data_dicts: List[Dict], ref_key: Any='speaker', prev_or_following: Literal['prev', 'following']='prev', adjacent_by: Literal['index', 'turn']='index', used_indexes: Union[List[int], None]=None) -> Dict[Any, Dict[Any, List[int]]]
```

Summarize cross-reference adjacency by sample index or turn.

The top-level key is the reference value of the item of interest. Each
nested key is the reference value of its adjacent item. Returned integers
identify the current sample when ``adjacent_by="index"`` and the current
turn when ``adjacent_by="turn"``.

Only adjacent items with different reference values are included. When a
subset containing gaps is supplied, adjacency is not created across those
non-consecutive gaps.

##### Parameters

data_dicts : list of dict
    Ordered sample-level records representing a sequence.
ref_key : Any, optional
    Dictionary key whose values are compared between adjacent items.
prev_or_following : {"prev", "following"}, optional
    Direction of adjacency relative to the current sample or turn.
adjacent_by : {"index", "turn"}, optional
    Unit used to define adjacency. ``"index"`` compares neighboring sample
    indexes; ``"turn"`` compares neighboring turns.
used_indexes : list of int or None, optional
    Ordered subset of sample indexes to process. If ``None``, all samples
    are used.

##### Returns

dict
    Nested mapping of
    ``current_ref_value -> adjacent_ref_value -> current_indexes_or_turns``.

##### Raises

ValueError
    If ``prev_or_following`` or ``adjacent_by`` is invalid.

#### `get_interval_split_indexes`

```python
def get_interval_split_indexes(data_dicts: List[Dict], ref_key: Any='speaker', interval_num: int=3, interval_split_by: Literal['index', 'turn']='index', used_indexes: Union[List[int], None]=None) -> Dict[int, List[int]]
```

Split a sequence into approximately equal chronological intervals.

When splitting by index, intervals contain sample indexes directly. When
splitting by turn, complete turns are assigned to intervals and then
expanded back into their sample indexes, so a turn is never divided across
two intervals.

##### Parameters

data_dicts : list of dict
    Ordered sample-level records representing a sequence.
ref_key : Any, optional
    Dictionary key used to identify turns when
    ``interval_split_by="turn"``.
interval_num : int, optional
    Number of intervals to create. The returned dictionary always contains
    this many interval keys, although some intervals may be empty when
    there are fewer samples or turns than intervals.
interval_split_by : {"index", "turn"}, optional
    Whether interval boundaries are based on sample indexes or complete
    turns.
used_indexes : list of int or None, optional
    Ordered subset of sample indexes to process. If ``None``, all samples
    are used.

##### Returns

dict
    Mapping from zero-based interval IDs to sample-index lists. Interval
    sizes differ by at most one unit before turn expansion.

##### Raises

ValueError
    If ``interval_num`` is not positive or ``interval_split_by`` is invalid.

<details>
<summary><strong>Internal module helpers (3)</strong></summary>

These functions support the public API and are included because they contain source docstrings. They are not the preferred compatibility surface for user code.

#### `_resolve_used_indexes`

```python
def _resolve_used_indexes(data_dicts: List[Dict], used_indexes: Union[List[int], None]) -> List[int]
```

Resolve and validate the sample indexes to be processed.

##### Parameters

data_dicts : list of dict
    Sample-level information records.
used_indexes : list of int or None
    Indexes to process. If ``None``, all indexes are returned in their
    original order.

##### Returns

list of int
    Validated indexes to process.

##### Raises

TypeError
    If an index is not an integer.
IndexError
    If an index is outside the valid range of ``data_dicts``.

#### `_values_equal`

```python
def _values_equal(left: Any, right: Any) -> bool
```

Compare two values while supporting scalar and array-like objects.

##### Parameters

left : Any
    First value to compare.
right : Any
    Second value to compare.

##### Returns

bool
    ``True`` when the values are equivalent. Array-like comparison results
    are reduced across all elements.

#### `_contains_equivalent`

```python
def _contains_equivalent(values: List[Any], candidate: Any) -> bool
```

Check whether a list contains a value equivalent to a candidate.

##### Parameters

values : list
    Existing values to search.
candidate : Any
    Candidate value, which may be a scalar or an array-like object.

##### Returns

bool
    ``True`` if an equivalent value is already present.

</details>

---

## Documentation maintenance

This file currently documents **160** classes, methods, and module-level functions extracted from the source code.

When signatures or docstrings change, regenerate this file so the documentation remains synchronized with the code. For conceptual workflow explanations and end-to-end examples, see the repository [README](https://github.com/Xia-code/flexmm).
