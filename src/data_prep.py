# -*- coding: utf-8 -*-
"""Prepare sequence-aware multimodal data, targets, and train/validation/test splits.

The module defines configuration objects, data-gathering utilities, target
statistics and conversion helpers, deterministic split strategies, sequence
overlap removal, and serialization helpers. Sample-level descriptive data are
expected as a list of dictionaries whose indexes identify the original samples.
"""

import os
import copy
import json
import pickle
import random
import warnings
import itertools
import numpy as np
from typing import List, Dict, Tuple, Union, Any, Callable, Literal, Mapping, Sequence, Optional
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict

from flexmm import info_utils

warnings.simplefilter("once")

# not all users need to use torch, so here try to import torch, if torch is not available, 
# some torch-related processing in this script will not be executed (this does not influence the main flow of the processing)
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True

global ORI_INDEX_KEY
ORI_INDEX_KEY = "_ORI_INDEX_"
global SEQ_INDEX_KEY
SEQ_INDEX_KEY = "_SEQ_INDEX_"
global INDEX_KEY_LIST
INDEX_KEY_LIST = [ORI_INDEX_KEY, SEQ_INDEX_KEY]

warnings.filterwarnings("once", category=UserWarning)


# configs
@dataclass
class BaseConfig():
    """Common configuration fields for one input or target data key group.

    Notes
    -----
    Instances store validated settings or pipeline state used by this module.
    """
    keys: Union[List, Any]
    # seq-related
    seq_len_before: int = 0
    seq_len_after: int = 0
    step_offset: int = 0
    stride: int = 1
    seq_pos_from_start: int = 0
    seq_pos_from_end: int = 0
    # seq padding-related
    seq_padding: bool = True
    seq_padding_mode: Literal["edge", "constant"] = "constant"
    seq_padding_value: Any = 0
    # dim-related
    keep_batch_seq_dims: bool = True
    squeeze_singleton_dims: bool = True
    # standardization-related
    standardize_data: bool = False
    standardize_method: Literal["zscore", "minmax"] = "zscore"
    standardize_scope: Literal["all", "split"] = "split"
    # data type when converting data to array, if not given, follow the default dtype in np.array for the data
    dtype: Union[None, str] = None
    
    def __post_init__(self):
        """Normalize derived fields and validate the configuration after initialization.
        """
        self.keys = _unify_to_list(self.keys)


@dataclass
class ClassificationTargetConfig(BaseConfig):
    # convert target to id or not after target2id is created
    """Configure one or more scalar classification targets.

    Notes
    -----
    Instances store validated settings or pipeline state used by this module.
    """
    convert_target_to_id: bool = False

    def __post_init__(self):
        """Normalize derived fields and validate the configuration after initialization.
        """
        super().__post_init__()
        self.config_name = "target"
        self.task_type = "c"
        self.is_multi_dim = False
        self.dtype = "int64"


@dataclass
class RegressionTargetConfig(BaseConfig):
    """Configure one or more regression targets and optional stratification bins.

    Notes
    -----
    Instances store validated settings or pipeline state used by this module.
    """
    is_multi_dim: bool = False
    convert_target_to_bin: bool = False
    # if bin size is not given, statistic value range and divide bin num to decide bin size
    stratified_bin_size: Union[float, int, None] = None
    stratified_bin_num: Optional[int] = 10
    bin_closed_side: Literal["upper", "lower"] = "lower"
    
    def __post_init__(self):
        """Normalize derived fields and validate the configuration after initialization.
        """
        super().__post_init__()
        self.config_name = "target"
        self.task_type = "r"
        self.is_multi_dim = self.is_multi_dim


@dataclass
class InputConfig(BaseConfig):
    """Configure one or more model-input data keys.

    Notes
    -----
    Instances store validated settings or pipeline state used by this module.
    """
    is_non_numeric: bool = False
    
    def __post_init__(self):
        """Normalize derived fields and validate the configuration after initialization.
        """
        super().__post_init__()
        self.config_name = "input"


@dataclass
class DataPrepConfig():
    # key identification-related
    # specify the target key that is used for straitified data split, if not given, then the index 0 of target keys will be used
    """Configure sequence construction, target processing, splitting, and persistence.

    Notes
    -----
    Instances store validated settings or pipeline state used by this module.
    """
    focused_target_key: Union[Any, None] = None
    # data split-related
    split_ref_key: Union[Any, None] = "speaker"
    split_dependency: Literal["independent", "dependent", "none"] = "independent"
    independent_split_valid_by: Union[None, Literal["index", "ref_key"]] = "index"
    split_mode: Literal["holdout", "kfold", "leave_one_out"] = "kfold"
    folds: int = 5
    train_valid_ratio: float = 0.9
    holdout_test_ratio: float = 0.2
    use_stratified_split: bool = False
    # sequence data formatting-relatd
    seq_group_mode: Literal["ref_key", "index"] = "ref_key"
    seq_group_key: Union[Any, None] = None # if None, group by split_ref_key; otherwise, group by seq_group_key
    seq_ranges_custom: Union[List[tuple[int, int]], None] = None
    include_seq_inter_ranges: bool = False
    # padding index
    seq_padding_index: int = -1
    # whether remove seq overlap across train, valid, and test sets
    remove_test_train_overlap_range: bool = True
    remove_train_valid_overlap_range: bool = False
    remove_overlap_priority: Union[List, None] = field(default_factory=lambda: ["test", "train", "valid"])
    # input target and modality configs
    data_configs: List = field(default_factory=list)
    # save data-related
    save_prepared_data: bool = True
    overwrite_data: bool = True
    store_dir: str = "./ExperimentStore"
    # receive user-given split dict or ref values
    index_split_dict_override: Union[None, Dict] = None
    train_ref_values_override: Union[None, Dict] = None
    valid_ref_values_override: Union[None, Dict] = None
    test_ref_values_override: Union[None, Dict] = None
    
    def __post_init__(self):
        # make key2config to find config by key conviniently
        """Normalize derived fields and validate the configuration after initialization.
        """
        self.key2config = {}
        for config in self.data_configs:
            for key in config.keys:
                self.key2config[key] = config
        # assert
        self.assert_config()
        # init input_keys and check duplicates
        self.input_keys = self._init_keys_and_check(self.data_configs, config_name="input")
        self.target_keys = self._init_keys_and_check(self.data_configs, config_name="target")
        # init focused_target_key
        if self.focused_target_key is None:
            if self.target_keys:
                self.focused_target_key = self.target_keys[0]
            else:
                raise ValueError(
                    "At least one target key must be configured in `data_configs`."
                )
        if self.focused_target_key not in self.target_keys:
            raise ValueError(
                f"The focused target key {self.focused_target_key!r} is not configured "
                "as a target."
            )
        # if the task type of focused_target_key is regression, but the target's dimension is >= 1, it cannot use stratified_split
        if(self.key2config[self.focused_target_key].is_multi_dim):
            if(self.use_stratified_split):
                with warnings.catch_warnings():
                    warnings.simplefilter("once")
                    warnings.warn(
                        f"The focused target {self.focused_target_key} dim is greater than 2, "
                        "so stratified split cannot be performed, but is_stratified_split == True. "
                        "Automatically change is_stratified_split to False."
                    )
                self.use_stratified_split = False
        # make target task type mappping, for getting task types of target keys easily
        self.target_key2type = {}
        for key in self.target_keys:
            self.target_key2type[key] = self.key2config[key].task_type
        # make dtype mapping, for getting array with dtype
        # defaultly input's dtype is float32, cls task's dtype is int64 (long), regression's dtype is float32
        self.key2dtype = {}
        for config in self.data_configs:
            for key in config.keys:
                self.key2dtype[key] = config.dtype
        # init remove_overlap_priority
        if(self.remove_overlap_priority == None):
            self.remove_overlap_priority = ["test", "train", "valid"]
        # init seq info dict for keys
        self.seq_info = self._make_seq_info()

    def _init_keys_and_check(self, configs, config_name="input"):
        """Collect keys for one config category and reject duplicated keys.

        Parameters
        ----------
        configs : Any
            Data configuration objects to inspect.
        config_name : Any
            Configuration category to collect, usually ``"input"`` or ``"target"``.
        """
        key_list = []
        for config in configs:
            if(config.config_name == config_name):
                key_list += config.keys
        key_counts = Counter(key_list)
        duplicates = [item for item, count in key_counts.items() if count > 1]
        if(len(duplicates) > 0):
            raise(ValueError(f"Duplicated {config_name}_keys {duplicates} were set in configs."))
        else:
            return key_list

    def _make_seq_info(self) -> (Dict, int, int):
        # make sequence length info dict of given keys
        """Build per-key sequence-window settings.
        """
        seq_info = {}
        for config in self.data_configs:
            for key in config.keys:
                seq_info[key] = {
                    "before": config.seq_len_before, 
                    "after": config.seq_len_after, 
                    "stride": config.stride, 
                    "step_offset": config.step_offset, 
                    "seq_pos_from_start": config.seq_pos_from_start, 
                    "seq_pos_from_end": config.seq_pos_from_end
                }
        return seq_info
    
    def assert_config(self):
        """Validate split-related configuration values and resolve compatible defaults.
        """
        if self.folds <= 0:
            raise ValueError("`folds` must be a positive integer.")
        if not 0.0 < self.train_valid_ratio < 1.0:
            raise ValueError("`train_valid_ratio` must be between 0 and 1.")
        if not 0.0 < self.holdout_test_ratio < 1.0:
            raise ValueError("`holdout_test_ratio` must be between 0 and 1.")
        if self.split_dependency not in {"independent", "dependent", "none"}:
            raise ValueError(
                "`split_dependency` must be 'independent', 'dependent', or 'none'."
            )
        if self.split_mode not in {"holdout", "kfold", "leave_one_out"}:
            raise ValueError(
                "`split_mode` must be 'holdout', 'kfold', or 'leave_one_out'."
            )
        if self.independent_split_valid_by not in {None, "index", "ref_key"}:
            raise ValueError(
                "`independent_split_valid_by` must be None, 'index', or 'ref_key'."
            )
        # valid_split_by should not be set when split_dependency is not independent
        if(self.split_dependency != "independent"):
            if(self.independent_split_valid_by is not None):
                raise ValueError(
                    "`independent_split_valid_by` is only applicable when "
                    "`split_dependency == 'independent'`."
                )
        # if folds is set to 1, set split_mode as holdout
        if(self.folds == 1 and 
           self.split_mode != "holdout"):
            with warnings.catch_warnings():
                warnings.simplefilter("once")
                warnings.warn(
                    f"'folds' is set to 1 but 'split_mode' is not holdout, "
                    "set 'split_mode' as holdout."
                )
            self.split_mode = "holdout"
        # if train_ref_values_override or valid_ref_values_override is given, independent_split_valid_by should be "ref_key"
        if(self.train_ref_values_override is not None or  
           self.valid_ref_values_override is not None):
            if(self.independent_split_valid_by != "ref_key"):
                with warnings.catch_warnings():
                    warnings.simplefilter("once")
                    warnings.warn(
                        f"'train_ref_values_override' or 'valid_ref_values_override' is given"
                        "'independent_split_valid_by' should be 'ref_key', but is not, "
                        "automatically set 'independent_split_valid_by' as 'ref_key'."
                    )
                self.independent_split_valid_by = "ref_key"
    
    def to_json(self):
        # format configs in data_configs and DataPrepConfigs separately for saving out, loading function is defined in the later part
        """Convert nested configuration objects into a JSON-serializable structure.

        Returns
        -------
        list
            Serialized configuration entries containing class names and dataclass fields.
        """
        configs = []
        # save the fields of input/target configs
        for config in self.data_configs:
            field_config = asdict(config)
            configs.append((config.__class__.__name__, field_config))
        # save the fields of DataPrepConfig
        config_dict = asdict(self)
        del config_dict["data_configs"]
        del config_dict["index_split_dict_override"]
        configs.append((self.__class__.__name__, config_dict))
        return configs


# Main class for data preparation, receving data_dicts and data_pre_config
class DataPreparator():
    """Run the complete data-preparation pipeline for sample-level dictionaries.

    Notes
    -----
    Instances store validated settings or pipeline state used by this module.
    """
    def __init__(self, data_dicts, data_prep_config, split_postprocess_fn=None):
        # can choose specified indexes for experiment, if not all indexes are used, an ori_index_key will be added to each dict item to save their original index in the original data_dicts
        """Initialize the data preparator and derive sequence-related index metadata.

        Parameters
        ----------
        data_dicts : Any
            Ordered sample-level dictionaries indexed by original sample position.
        data_prep_config : Any
            Configuration object saved with the prepared data.
        split_postprocess_fn : Any
            Optional callable receiving index folds and reference-value folds after built-in processing.
        """
        self.data_dicts = data_dicts
        self.config = data_prep_config
        self.split_postprocess_fn = split_postprocess_fn
        # Step 0. init indexes related info, including seq ranges, used indexes, ori_index2id, and id2ori_index
        self._init_seq_indexes_info()

    def run(self):
        # Step 1. get seq indexes of each key, - for clearly see that the seq_indexes_dict is assigned, not assign it in the get_seq_indexes()
        """Execute sequence gathering, target processing, splitting, and optional saving.

        Returns
        -------
        tuple
            ``(collected_data, info_dict)`` produced by the complete pipeline.
        """
        self.get_seq_indexes()
        # Step 2. gather data of used indexes and make a summary dict, formatting as Dict[input/target/info key][array/list], add original index
        self.gather_data()
        # Step 3. perform target-related processings, including converting targets to scalar, making info dicts of targets, and convert targets to ids or bins
        self.process_target()
        # Step 4. split the data into train/valid/test sets, get data indexes in each set (center indexes regarding to sequences)
        self.split_data()
        # Step 5. make info dict, containing label/value stats overall and in each fold, label2id, id2label, ori_index2id, id2ori_index, etc..
        self.make_info_dict()
        # Step 6. if save data is set, save
        if(self.config.save_prepared_data):
            self.save_data()
        # torch.dataset will be built in experiment, because it needs additional lib (pytorch) than preparing data
        # data preparation will finish by making a data dict, whose keys are input_keys and target_keys, and values of keys are arrays or list of the corresponding modality and targets
        return self.collected_data, self.info_dict
    
    # Step 0 functions
    def _init_seq_indexes_info(self):
        # get target indexes offset based on future indexes use
        """Initialize sequence ranges, used indexes, and original-index mappings.
        """
        self.index_offsets = {}
        for config in self.config.data_configs:
            for key in config.keys:
                self.index_offsets[key] = config.step_offset
        self.max_target_index_offset = max(list(self.index_offsets.values()))
        # get seq_ranges
        self.seq_ranges = self._init_seq_ranges()
        # make used_indexes
        self.used_indexes = self._get_used_indexes_from_ranges()
        # make id of used_indexes <---> ori_indexes for getting the mapping quickly
        self.id2ori_index = {}
        self.ori_index2id = {}
        for i, index in enumerate(self.used_indexes):
            self.id2ori_index[i] = index
            self.ori_index2id[index] = i
    
    def _init_seq_ranges(self):
        """Create half-open sequence ranges from groups or custom boundaries.

        Returns
        -------
        list[tuple[int, int]]
            Sorted half-open sequence ranges.
        """
        if self.config.seq_ranges_custom is None:
            if self.config.seq_group_mode == "ref_key":
                ref_key = (
                    self.config.split_ref_key
                    if self.config.seq_group_key is None
                    else self.config.seq_group_key
                )
                turn_info = info_utils.get_turn2ref_value_and_indexes(
                    self.data_dicts,
                    ref_key=ref_key,
                )
                seq_ranges = [
                    (min(turn["indexes"]), max(turn["indexes"]) + 1)
                    for turn in turn_info.values()
                    if turn["indexes"]
                ]
            elif self.config.seq_group_mode == "index":
                seq_ranges = [(0, len(self.data_dicts))]
            else:
                raise ValueError(
                    "`seq_group_mode` must be either 'ref_key' or 'index'."
                )
        else:
            seq_ranges = [tuple(r) for r in self.config.seq_ranges_custom]
            self._validate_seq_ranges(seq_ranges)
            if self.config.include_seq_inter_ranges:
                merged = self._merge_ranges(seq_ranges)
                covered_ranges = []
                cursor = 0
                for range_start, range_end in merged:
                    if cursor < range_start:
                        covered_ranges.append((cursor, range_start))
                    covered_ranges.append((range_start, range_end))
                    cursor = range_end
                if cursor < len(self.data_dicts):
                    covered_ranges.append((cursor, len(self.data_dicts)))
                seq_ranges = covered_ranges
        if not seq_ranges:
            raise ValueError("No valid sequence range could be created.")
        seq_ranges = sorted(set(seq_ranges), key=lambda item: (item[0], item[1]))
        return seq_ranges

    def _validate_seq_ranges(self, ranges):
        """Validate custom half-open sequence ranges against the dataset length.

        Parameters
        ----------
        ranges : Any
            Half-open ``(start, end)`` index ranges.
        """
        for range_start, range_end in ranges:
            if not 0 <= range_start < range_end <= len(self.data_dicts):
                raise ValueError(
                    "Each sequence range must satisfy "
                    "0 <= start < end <= len(data_dicts)."
                )

    def _merge_ranges(self, ranges):
        """Merge overlapping or touching half-open ranges.

        Parameters
        ----------
        ranges : Any
            Half-open ``(start, end)`` index ranges.

        Returns
        -------
        list[tuple[int, int]]
            Merged half-open ranges.
        """
        if not ranges:
            return []
        # order by start
        ranges = sorted(ranges, key=lambda x: x[0])
        merged = [ranges[0]]
        for start, end in ranges[1:]:
            last_start, last_end = merged[-1]
            if(start <= last_end):  # if two ranges are overlapped
                # merge the later one to the former one
                merged[-1] = (last_start, max(last_end, end))
            else:
                # if not overlapped, create a new range
                merged.append((start, end))
        return merged

    def _get_used_indexes_from_ranges(self):
        # summary indexes in seq_ranges, these indexes are used in data preparation, others are not used
        """Collect unique sample indexes covered by the sequence ranges.

        Returns
        -------
        list[int]
            Unique covered original sample indexes.
        """
        used_indexes = []
        seen = set()
        for seq_range in self.seq_ranges:
            for index in range(seq_range[0], seq_range[1]):
                if index not in seen:
                    used_indexes.append(index)
                    seen.add(index)
        return used_indexes
    
    # Step 1 functions
    # get seq indexes of each key
    def get_seq_indexes(self):
        """Build sequence windows for every configured input and target key.
        """
        self.seq_indexes_dict = {}
        # seq indexes single should remove max_target_index_offset from the last seq to align inputs and targets
        for key in self.config.seq_info:
            seq_indexes_key = []
            for seq_range in self.seq_ranges:
                seq_range_process = [-1, -1]
                seq_range_process[0] = seq_range[0] + self.index_offsets[key]
                seq_range_process[1] = seq_range[1] + self.index_offsets[key]
                seq_indexes_single_range, _ = shift_get_seq_indexes(
                    index_list=list(range(seq_range_process[0], seq_range_process[1], 1)), 
                    seq_len_before=self.config.seq_info[key]["before"],
                    seq_len_after=self.config.seq_info[key]["after"], 
                    stride=self.config.seq_info[key]["stride"], 
                    step_offset=self.config.seq_info[key]["step_offset"], 
                    seq_pos_from_start=self.config.seq_info[key]["seq_pos_from_start"], 
                    seq_pos_from_end=self.config.seq_info[key]["seq_pos_from_end"], 
                    padding=self.config.key2config[key].seq_padding, 
                    padding_index=self.config.seq_padding_index
                )
                seq_indexes_key += seq_indexes_single_range
            self.seq_indexes_dict[key] = seq_indexes_key

        sequence_counts = {
            key: len(sequence_indexes)
            for key, sequence_indexes in self.seq_indexes_dict.items()
        }
        if len(set(sequence_counts.values())) != 1:
            raise ValueError(
                "Configured keys produce different numbers of sequence samples: "
                f"{sequence_counts}. Align sequence filtering settings across keys."
            )
        focused_sequences = self.seq_indexes_dict[self.config.focused_target_key]
        effective_indexes = [anchor for anchor, _ in focused_sequences]
        if len(effective_indexes) != len(set(effective_indexes)):
            raise ValueError(
                "The focused target produced duplicated sequence anchors. Check "
                "whether custom sequence ranges overlap."
            )
        invalid = [
            index for index in effective_indexes
            if index < 0 or index >= len(self.data_dicts)
        ]
        if invalid:
            raise IndexError(
                "Sequence offsets produced anchors outside `data_dicts`: "
                f"{invalid[:5]}."
            )
        self.used_indexes = effective_indexes
        self.id2ori_index = dict(enumerate(self.used_indexes))
        self.ori_index2id = {
            original_index: prepared_id
            for prepared_id, original_index in self.id2ori_index.items()
        }
        
    # Step 2 functions
    # gather data
    def gather_data(self):
        """Gather configured data fields into aligned arrays or Python lists.
        """
        self.collected_data = {}
        for config in self.config.data_configs:
            for key in config.keys:
                self.collected_data[key] = gather_data_single_key(
                    self.data_dicts,
                    key,
                    self.seq_indexes_dict[key], 
                    dtype=self.config.key2dtype[key], 
                    seq_padding_index=self.config.seq_padding_index, 
                    seq_padding_value=config.seq_padding_value, 
                    seq_padding_mode=config.seq_padding_mode, 
                    squeeze_singleton_dims=config.squeeze_singleton_dims, 
                    keep_batch_seq_dims=config.keep_batch_seq_dims
                )
            self.collected_data[ORI_INDEX_KEY] = self.used_indexes
            self.collected_data[SEQ_INDEX_KEY] = [seq[1] for seq in self.seq_indexes_dict[self.config.focused_target_key]]
        # return self.collected_data
    
    # Step 3 functions
    def process_target(self):
        """Create target statistics and apply requested target conversions.
        """
        self.convert_target_to_scalar()
        self.target_info = self.make_target_info_dict()
        self.convert_target_form()
    
    def convert_target_to_scalar(self):
        # convert target to scalar lists for stats and get info
        """Create scalar target lists for statistics and splitting.
        """
        self.target_scalar_lists = {}
        for key in self.config.key2config:
            config = self.config.key2config[key]
            if(config.config_name == "target"):
                if(config.task_type == "c"):
                    self.target_scalar_lists[key] = get_scalar_target_list(self.data_dicts, key, self.used_indexes)
                elif(config.task_type == "r"):
                    if(config.is_multi_dim):
                        # if target is multi dimensions, not process it
                        self.target_scalar_lists[key] = ["_multi_dim_"]
                    else:
                        self.target_scalar_lists[key] = get_scalar_target_list(self.data_dicts, key, self.used_indexes)
    
    def make_target_info_dict(self):
        """Build label mappings, regression bins, statistics, and index groups.

        Returns
        -------
        dict
            Target metadata keyed by target name.
        """
        target_info = {}
        for config in self.config.data_configs:
            if(config.config_name == "target"):
                target_keys = config.keys
                target_type = config.task_type
                for key in target_keys:
                    key_info = {}
                    if(target_type == "c"):
                        (target_stats,
                         target2id,
                         id2target,
                         target2indexes) = get_target_info_cls(
                             self.data_dicts,
                             key, 
                             used_indexes=self.used_indexes, 
                        )
                        key_info["target2id"] = target2id
                        key_info["id2target"] = id2target
                        key_info["target_stats"] = target_stats
                        key_info["target2indexes"] = target2indexes
                        key_info["target_type"] = target_type
                    if(target_type == "r"):
                        if(self.config.key2config[key].is_multi_dim):
                            key_info["target_size"] = self._get_pure_shape(self.data_dicts[0][key])
                            key_info["target_type"] = target_type
                        else:
                            (target_stats,
                             target_bin_ranges,
                             target2indexes) = get_target_info_regression(
                                 self.data_dicts,
                                 key, 
                                 used_indexes=self.used_indexes, 
                                 stratified_bin_size=config.stratified_bin_size,
                                 stratified_bin_num=config.stratified_bin_num, 
                                 bin_closed_side=self.config.key2config[key].bin_closed_side
                            )
                            key_info["target_stats"] = target_stats
                            key_info["target_bin_ranges"] = target_bin_ranges
                            key_info["target2indexes"] = target2indexes
                            key_info["target_type"] = target_type
                    target_info[key] = key_info
        return target_info
    
    def convert_target_form(self):
        """Convert collected targets to class IDs or regression-bin representatives.
        """
        for config in self.config.data_configs:
            if config.config_name != "target":
                continue
            for key in config.keys:
                if self.target_scalar_lists[key] == ["_multi_dim_"]:
                    continue
                if config.task_type == "c" and config.convert_target_to_id:
                    target2id = self.target_info[key]["target2id"]
                    converted = [target2id[value] for value in self.target_scalar_lists[key]]
                    self.collected_data[key] = np.asarray(
                        converted,
                        dtype=self.config.key2dtype[key],
                    )
                elif config.task_type == "r" and config.convert_target_to_bin:
                    bin_ranges = self.target_info[key]["target_bin_ranges"]
                    converted = [
                        _convert_value_to_bin(value, bin_ranges, config.bin_closed_side)
                        for value in self.target_scalar_lists[key]
                    ]
                    self.collected_data[key] = np.asarray(
                        converted,
                        dtype=self.config.key2dtype[key],
                    )

    # Step 5 functions
    def split_data(self):
        """Create train, validation, and test folds according to the configured strategy.
        """
        if self.config.index_split_dict_override is not None:
            self.index_split_folds = _normalize_index_split_folds(
                self.config.index_split_dict_override
            )
            self.ref_value_split_folds = _build_ref_value_split_folds(
                self.data_dicts,
                self.index_split_folds,
                self.config.split_ref_key,
            )
        else:
            focused_config = self.config.key2config[self.config.focused_target_key]
            target2indexes = None
            if not focused_config.is_multi_dim:
                target2indexes = self.target_info[self.config.focused_target_key].get(
                    "target2indexes"
                )
            common_params = dict(
                data_dicts=self.data_dicts,
                split_ref_key=self.config.split_ref_key,
                folds=self.config.folds,
                split_mode=self.config.split_mode,
                train_valid_ratio=self.config.train_valid_ratio,
                holdout_test_ratio=self.config.holdout_test_ratio,
                use_stratified_split=self.config.use_stratified_split,
                focused_target_key=self.config.focused_target_key,
                is_focused_key_multi_dim=focused_config.is_multi_dim,
                focused_target_task_type=focused_config.task_type,
                target2indexes=target2indexes,
                used_indexes=self.used_indexes,
            )
            if focused_config.task_type == "r":
                common_params.update(
                    stratified_bin_size=focused_config.stratified_bin_size,
                    stratified_bin_num=focused_config.stratified_bin_num,
                )
            if self.config.split_dependency == "independent":
                self.index_split_folds, self.ref_value_split_folds = (
                    data_split_independent(
                        **common_params,
                        split_valid_by=self.config.independent_split_valid_by,
                        train_ref_values_override=self.config.train_ref_values_override,
                        valid_ref_values_override=self.config.valid_ref_values_override,
                        test_ref_values_override=self.config.test_ref_values_override,
                    )
                )
            elif self.config.split_dependency == "dependent":
                self.index_split_folds, self.ref_value_split_folds = (
                    data_split_dependent(**common_params)
                )
            else:
                self.index_split_folds, self.ref_value_split_folds = (
                    data_split_unconstrained(**common_params)
                )

        if (
            self.config.remove_test_train_overlap_range
            or self.config.remove_train_valid_overlap_range
        ):
            self.post_split_process()

        if self.split_postprocess_fn is not None:
            processed = self.split_postprocess_fn(
                self.index_split_folds,
                self.ref_value_split_folds,
            )
            if processed is not None:
                self.index_split_folds, self.ref_value_split_folds = processed

    def post_split_process(self):
        """Remove sequence anchors that cause prohibited cross-split overlap.
        """
        focused_sequences = self.seq_indexes_dict[self.config.focused_target_key]
        anchor_index2seq_indexes = defaultdict(set)
        for position, (anchor_index, _) in enumerate(focused_sequences):
            for key_sequences in self.seq_indexes_dict.values():
                if position < len(key_sequences):
                    anchor_index2seq_indexes[anchor_index].update(
                        key_sequences[position][1]
                    )
        anchor_index2seq_indexes = {
            anchor: sorted(indexes)
            for anchor, indexes in anchor_index2seq_indexes.items()
        }
        self.index_split_folds, self.ref_value_split_folds = (
            remove_overlapped_seq_split(
                self.data_dicts,
                self.index_split_folds,
                anchor_index2seq_indexes,
                split_ref_key=self.config.split_ref_key,
                is_test_train_no_seq_overlap=(
                    self.config.remove_test_train_overlap_range
                ),
                is_train_valid_no_seq_overlap=(
                    self.config.remove_train_valid_overlap_range
                ),
                priority_order=self.config.remove_overlap_priority,
                padding_index=self.config.seq_padding_index,
            )
        )

    # Step 6 functions
    def make_info_dict(self):
        """Assemble preparation metadata for saving and downstream experiments.
        """
        self.info_dict = {}
        self.info_dict["index_split_folds"] = self.index_split_folds
        self.info_dict["ref_value_split_folds"] = self.ref_value_split_folds
        # self.info_dict["data_prep_config"] = self.config
        self.info_dict["id2ori_index"] = self.id2ori_index
        self.info_dict["ori_index2id"] = self.ori_index2id
        self.info_dict["target_info"] = self.target_info
        self.info_dict["input_shapes"] = self.get_input_shapes()
        # self.info_dict["zscore_miu_sigma"] = self.get_zscore_miu_sigma()
    
    def get_zscore_miu_sigma(self):
        """Calculate per-fold normalization statistics for configured input keys.

        Returns
        -------
        list[dict]
            Normalization statistics for every fold and split.
        """
        self.zscore_miu_sigma_folds = []
        for fold in sorted(self.index_split_folds):
            index_split_dict = self.index_split_folds[fold]
            miu_sigma_dict = {}
            # only caclulate for input keys
            for config in self.config.data_configs:
                if(config.config_name == "input"):
                    if(config.standardize_data):
                        # save input_key's miu sigma under split_key, unifying the indexing layer of getting miu sigma to getting split indexes when generate experiment data
                        for split_key in index_split_dict:
                            miu_sigma_dict[split_key] = {}
                            # calculate miu sigma for each key in one config
                            for key in config.keys:
                                if(config.standardize_scope == "all"):
                                    miu, sigma = calculate_miu_sigma(self.data_dicts, key, self.used_indexes)
                                else:
                                    miu, sigma = calculate_miu_sigma(self.data_dicts, key, index_split_dict[split_key])
                                miu_sigma_dict[split_key][key] = (miu, sigma)
            self.zscore_miu_sigma_folds.append(miu_sigma_dict)
        return self.zscore_miu_sigma_folds

    def get_input_shapes(self):
        """Infer model-input shapes from the collected data.

        Returns
        -------
        dict
            Mapping from input keys to inferred feature shapes.
        """
        input_shapes = {}
        for key in self.config.input_keys:
            shape = self._get_pure_shape(self.collected_data[key])
            input_shapes[key] = shape
        return input_shapes

    def _get_pure_shape(self, data):
        """Infer the feature shape after removing batch and sequence dimensions.

        Parameters
        ----------
        data : Any
            Value used by ``_get_pure_shape``.

        Returns
        -------
        tuple
            Feature dimensions after batch and sequence dimensions.
        """
        if(isinstance(data, list)):
            if(isinstance(data[0], np.ndarray)):
                shape = data[0].shape[2:]
            else:
                shape = (1, )
        elif(isinstance(data, np.ndarray)):
            if(len(data.shape) == 1):
                shape = (data.shape[-1], )
            if(len(data.shape) == 2):
                shape = (data.shape[-1], )
            if(len(data.shape) >= 3):
                shape = data.shape[2:]
        return shape
    
    # Step 7 functions
    def save_data(self, save_dir=None, overwrite_data=None):
        # the user can flexibly choose to used the save root in the config, or specify a new save root
        """Save prepared data and metadata using explicit or configured options.

        Parameters
        ----------
        save_dir : Any
            Destination directory. ``None`` uses the configured store directory.
        overwrite_data : Any
            Whether existing serialized data may be replaced.
        """
        used_save_dir = self.config.store_dir if save_dir == None else save_dir
        used_overwrite_data = self.config.overwrite_data if overwrite_data == None else overwrite_data
        save_data(
            self.collected_data,
            self.info_dict,
            self.config,
            save_dir=used_save_dir,
            overwrite_data=used_overwrite_data
        )


# ---------- Data, seq, and indexes gathering ---------- #
# gather data regarding to used indexes
def pick_data_by_indexes(data_dicts: List[Dict], used_indexes: List) -> List:
    # collect dicts of used_indexes, add original index to each dict item.
    """Select sample dictionaries by original index and record each source index.

    Parameters
    ----------
    data_dicts : List[Dict]
        Ordered sample-level dictionaries indexed by original sample position.
    used_indexes : List
        Original sample indexes to process. ``None`` selects all samples.

    Returns
    -------
    list[dict]
        Selected sample dictionaries.
    """
    picked_data = []
    for i, index in enumerate(used_indexes):
        picked_data.append(data_dicts[index])
        picked_data[i][ORI_INDEX_KEY] = index
    return picked_data


def gather_data_single_key(
    data_dicts: List[Dict],
    data_key: Any,
    seq_indexes: List, 
    dtype=None, 
    seq_padding_index: int = -1, 
    seq_padding_mode: Literal["constant", "edge"] = "constant", 
    seq_padding_value: Any = 0.0, 
    squeeze_singleton_dims: bool = True, 
    keep_batch_seq_dims: bool = True
) -> np.array:
    # gather data of the index_list, format the data into np.array if original data is int, float, int/float list, or np array; or into list if orignal data are str or list
    # check data type - here does not check the shape of list, so if the data is not in the same shape, error may be raised
    """Gather all sequence windows for one data key.

    Parameters
    ----------
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

    Returns
    -------
    numpy.ndarray or list
        Gathered values for all sequence anchors.
    """
    sample_data = data_dicts[seq_indexes[0][0]][data_key]
    if (TORCH_AVAILABLE):
        if(isinstance(sample_data, torch.Tensor)):
            data_operation = "array-tensor"
    if(isinstance(sample_data, (int, float, np.ndarray))):
        data_operation = "array"
    elif(isinstance(sample_data, list)):
        # if given list, only check the first dim inside, not consider deeper contents
        if(isinstance(sample_data[0], (int, float))):
            data_operation = "array"
        else:
            data_operation = "list-list"
    else:
        data_operation = "list-scalar"
    # gather and make data array/list
    data_list = []
    for (center_index, seq_index_list) in seq_indexes:
        seq_data = gather_data_by_indexes(
            data_dicts,
            data_key,
            used_indexes=seq_index_list,
            sample_data=sample_data,
            data_operation=data_operation, 
            dtype=dtype, 
            seq_padding_index=seq_padding_index, 
            seq_padding_mode=seq_padding_mode, 
            seq_padding_value=seq_padding_value, 
            squeeze_singleton_dims=squeeze_singleton_dims
        )
        if(data_operation.split("-")[0] == "array" and 
           squeeze_singleton_dims):
            seq_data = seq_data.squeeze()
        data_list.append(seq_data)
    if(data_operation.split("-")[0] == "array"):
        if(keep_batch_seq_dims == True):
            # if is_batch_seq_dim_form, then the data form should be [batch, seq, [dims]], add dim if the data form is not satisfied
            for i in range(len(data_list)):
                least_axis = 3
                for _j in range(least_axis - len(data_list[i].shape)):
                    data_list[i] = np.expand_dims(data_list[i], axis=0)
        try:
            data_array = np.array(data_list).squeeze()
            return data_array
        except:
            return data_list
    else:
        return data_list


def gather_data_by_indexes(
    data_dicts,
    data_key,
    used_indexes=None,
    sample_data=None,
    data_operation="array", 
    dtype=None, 
    seq_padding_index=-1, 
    seq_padding_mode: Literal["constant", "edge"] = "constant", 
    seq_padding_value=0, 
    squeeze_singleton_dims=False
):
    """Gather one sequence of values, applying constant or edge padding.

    Parameters
    ----------
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

    Returns
    -------
    numpy.ndarray or list
        Gathered values for one sequence.
    """
    indexes_for_process = list(range(len(data_dicts))) if used_indexes == None else used_indexes
    sample_data_processing = sample_data if sample_data is None else data_dicts[indexes_for_process[0]][data_key]
    # if use array, padding with the sample_data_shape; if use list-list, padding with the len of sample_data
    if(data_operation in ["array", "array-tensor"]):
        sample_data_shape = np.array(sample_data_processing).shape
    # if repeat padding, then copy the first non-padding content as the padding value/array/tensor/etc..
    if(seq_padding_mode == "edge"):
        for index in indexes_for_process:
            if(index != seq_padding_index):
                if(data_operation == "array-tensor"):
                    edge_padding_value = data_dicts[index][data_key].to_numpy()
                else:
                    edge_padding_value = data_dicts[index][data_key]
                break
    # gather and make data array/list for one sequence
    seq_data = []
    for index in indexes_for_process:
        if(data_operation == "array-tensor"):
            append_data = data_dicts[index][data_key].to_numpy()
        else:
            append_data = data_dicts[index][data_key]
        if(index != seq_padding_index):
            seq_data.append(append_data)
        # append padding
        elif(index == seq_padding_index):
            if(seq_padding_mode == "constant"):
                if(data_operation in ["array", "array-tensor"]):
                    seq_data.append(np.full(sample_data_shape, seq_padding_value))
                elif(data_operation == "list-list"):
                    seq_data.append([seq_padding_value] * len(sample_data_processing))
                elif(data_operation == "list-scalar"):
                    seq_data.append(seq_padding_value)
            elif(seq_padding_mode == "edge"):
                if(data_operation in ["array", "array-tensor"]):
                    seq_data.append(np.full(sample_data_shape, edge_padding_value))
                elif(data_operation == "list-list"):
                    seq_data.append([edge_padding_value] * len(sample_data_processing))
                elif(data_operation == "list-scalar"):
                    seq_data.append(edge_padding_value)
    if(data_operation.split("-")[0] == "array"):
        # stack arrays to expand a dim of seq
        seq_array = np.array(seq_data, dtype=dtype)
        if(squeeze_singleton_dims == True):
            seq_array = seq_array.squeeze()
        return seq_array
    elif(data_operation.split("-")[0] == "list"):
        return seq_data


# get sequence indexes
def shift_get_seq_indexes(
    index_list: List,
    seq_len_before: int,
    seq_len_after: int,
    step_offset: int = 0, 
    stride: int = 1, 
    seq_pos_from_start: int = 0, 
    seq_pos_from_end: int = 0, 
    padding: bool = True, 
    padding_index: int = -1
) -> List[Tuple[int, List]]:
    # shift the used_indexes to get sequence indexes lists based on seq_len_before/after
    # the input used_indexes should be a list containing continuous indexes
    # elements (indexes) in the used_indexes may not align to the idx of the used_indexes,
    # so record the mapping of element (index) to sequence indexes by dict with the key of element rather than using a list to make sure the mapping is correct
    """Construct strided context windows around filtered anchor indexes.

    Parameters
    ----------
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

    Returns
    -------
    tuple[list, list]
        Sequence tuples and their anchor indexes.
    """
    if(seq_pos_from_end == 0):
        pos_filtered_index_list = index_list[seq_pos_from_start: ]
    else:
        pos_filtered_index_list = index_list[seq_pos_from_start: -seq_pos_from_end]
    if(padding == True):
        indexes_for_process = [padding_index for i in range(seq_len_before * stride)] + pos_filtered_index_list + [padding_index for i in range(seq_len_after * stride)]
    else:
        indexes_for_process = index_list
    if(len(indexes_for_process) == 0):
        raise (ValueError("The given index_list is empty."))
    # save center index and its window as a tuple (center index, window indexes), so that the user can make multiple windows for a single center index in one experiment regarding to their needs.
    seq_indexes = []
    seq_anchor_indexes = []
    for i in range(len(pos_filtered_index_list)):
        # seq_indexes contains the key of the center index (index of interest), and the index sequence from seq_len_before to center index to seq_len_after
        # adjust by step_offset
        center_i = i + seq_len_before * stride
        strided_seq_indexes = get_strided_seq(
            indexes_for_process, 
            center_i, 
            stride, 
            seq_len_before, 
            seq_len_after
        )
        if(len(strided_seq_indexes) == seq_len_before + seq_len_after + 1):
            seq_indexes.append(
                (
                    indexes_for_process[center_i], 
                    get_strided_seq(
                        indexes_for_process, 
                        center_i, 
                        stride, 
                        seq_len_before, 
                        seq_len_after
                    )
                )
            )
            seq_anchor_indexes.append(pos_filtered_index_list[i])
    return seq_indexes, seq_anchor_indexes


def get_strided_seq(index_list, i, stride, seq_len_before, seq_len_after):
    """Extract a strided window around one position in an index list.

    Parameters
    ----------
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

    Returns
    -------
    list[int]
        Valid positions in the requested strided window.
    """
    indices = (
        [i - j * stride for j in range(seq_len_before, 0, -1)] +
        [i] +
        [i + j * stride for j in range(1, seq_len_after + 1)]
    )
    return [index_list[k] for k in indices if 0 <= k < len(index_list)]


# ---------- Data split-related functions ---------- #
def data_split_independent(
    data_dicts: List[Dict],
    split_ref_key: Any,
    split_mode: Literal["holdout", "kfold", "leave_one_out"] = "kfold",
    folds: int = 5,
    train_valid_ratio: float = 0.9,
    holdout_test_ratio: float = 0.2,
    use_stratified_split: bool = False,
    split_valid_by: Literal["index", "ref_key"] = "index",
    focused_target_key: Any = "label",
    is_focused_key_multi_dim: bool = False,
    target2indexes: Optional[Dict] = None,
    focused_target_task_type: Literal["c", "r"] = "c",
    stratified_bin_num: Optional[int] = None,
    stratified_bin_size: Optional[float] = None,
    used_indexes: Optional[List[int]] = None,
    train_ref_values_override: Optional[Union[Mapping, Sequence]] = None,
    valid_ref_values_override: Optional[Union[Mapping, Sequence]] = None,
    test_ref_values_override: Optional[Union[Mapping, Sequence]] = None,
) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[Any]]]]:
    """Split reference groups independently so test groups never appear in train or validation.

    Parameters
    ----------
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

    Returns
    -------
    tuple[dict, dict]
        Index folds and corresponding reference-value folds.
    """
    _validate_split_parameters(
        split_mode,
        folds,
        train_valid_ratio,
        holdout_test_ratio,
    )
    if split_valid_by not in {"index", "ref_key"}:
        raise ValueError("`split_valid_by` must be 'index' or 'ref_key'.")
    if use_stratified_split and is_focused_key_multi_dim:
        raise ValueError("Stratified splitting does not support multidimensional targets.")

    indexes = _normalize_used_indexes(data_dicts, used_indexes)
    ref_value2indexes = info_utils.get_ref_value2indexes(
        data_dicts,
        ref_key=split_ref_key,
        used_indexes=indexes,
    )
    ref_values = list(ref_value2indexes)
    if not ref_values:
        raise ValueError("No reference values are available for splitting.")

    if test_ref_values_override is not None:
        test_ref_groups = _normalize_fold_override(test_ref_values_override)
    elif split_mode == "holdout":
        test_count = _get_holdout_size(len(ref_values), holdout_test_ratio)
        test_ref_groups = [ref_values[-test_count:]]
    elif split_mode == "leave_one_out":
        test_ref_groups = [[value] for value in reversed(ref_values)]
    else:
        effective_folds = min(folds, len(ref_values))
        if effective_folds < folds:
            warnings.warn(
                f"`folds` was reduced from {folds} to {effective_folds} because "
                "there are fewer reference values than requested folds."
            )
        test_ref_groups = _balanced_chunks(ref_values, effective_folds, reverse=True)

    target_map = _prepare_target2indexes(
        data_dicts,
        focused_target_key,
        indexes,
        focused_target_task_type,
        target2indexes,
        stratified_bin_size,
        stratified_bin_num,
    ) if use_stratified_split else None

    index_split_folds = {}
    ref_value_split_folds = {}
    for fold, test_ref_values in enumerate(test_ref_groups):
        _validate_ref_values(test_ref_values, ref_values, "test")
        test_ref_values = _deduplicate_preserve_order(test_ref_values)
        train_valid_ref_values = [
            value for value in ref_values if value not in set(test_ref_values)
        ]
        split_dict = _init_split_set_dict()
        split_dict["test"] = _flatten(
            ref_value2indexes[value] for value in test_ref_values
        )

        if split_valid_by == "ref_key":
            train_override = _get_fold_override(train_ref_values_override, fold)
            valid_override = _get_fold_override(valid_ref_values_override, fold)
            train_ref_values, valid_ref_values = _resolve_train_valid_ref_values(
                train_valid_ref_values,
                train_valid_ratio,
                train_override,
                valid_override,
            )
            split_dict["train"] = _flatten(
                ref_value2indexes[value] for value in train_ref_values
            )
            split_dict["valid"] = _flatten(
                ref_value2indexes[value] for value in valid_ref_values
            )
        else:
            if train_ref_values_override is not None or valid_ref_values_override is not None:
                raise ValueError(
                    "Reference-value train/validation overrides require "
                    "`split_valid_by='ref_key'`."
                )
            candidate_indexes = _flatten(
                ref_value2indexes[value] for value in train_valid_ref_values
            )
            if use_stratified_split:
                split_dict["train"], split_dict["valid"] = (
                    _stratified_train_valid_split(
                        candidate_indexes,
                        target_map,
                        train_valid_ratio,
                    )
                )
            else:
                # Split each remaining reference group separately so that, when
                # possible, every group contributes to both train and validation.
                for value in train_valid_ref_values:
                    train_part, valid_part = _split_train_valid_indexes(
                        ref_value2indexes[value],
                        train_valid_ratio,
                    )
                    split_dict["train"].extend(train_part)
                    split_dict["valid"].extend(valid_part)

        _validate_split_dict(split_dict, indexes)
        index_split_folds[fold] = split_dict
        ref_value_split_folds[fold] = _build_ref_value_split_dict(
            data_dicts,
            split_dict,
            split_ref_key,
        )

    return index_split_folds, ref_value_split_folds


def data_split_dependent(
    data_dicts: List[Dict],
    split_ref_key: Any,
    split_mode: Literal["holdout", "kfold", "leave_one_out"] = "kfold",
    folds: int = 5,
    train_valid_ratio: float = 0.9,
    holdout_test_ratio: float = 0.2,
    use_stratified_split: bool = False,
    split_valid_by: Literal["index", "ref_key"] = "index",
    focused_target_key: Any = "label",
    is_focused_key_multi_dim: bool = False,
    target2indexes: Optional[Dict] = None,
    focused_target_task_type: Literal["c", "r"] = "c",
    stratified_bin_num: Optional[int] = None,
    stratified_bin_size: Optional[float] = None,
    used_indexes: Optional[List[int]] = None,
) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[Any]]]]:
    """Split samples within every reference group so groups may appear in all sets.

    Parameters
    ----------
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

    Returns
    -------
    tuple[dict, dict]
        Index folds and corresponding reference-value folds.
    """
    _validate_split_parameters(
        split_mode,
        folds,
        train_valid_ratio,
        holdout_test_ratio,
    )
    if use_stratified_split and is_focused_key_multi_dim:
        raise ValueError("Stratified splitting does not support multidimensional targets.")

    indexes = _normalize_used_indexes(data_dicts, used_indexes)
    ref_value2indexes = info_utils.get_ref_value2indexes(
        data_dicts,
        ref_key=split_ref_key,
        used_indexes=indexes,
    )
    if not ref_value2indexes:
        raise ValueError("No reference groups are available for splitting.")

    if split_mode == "holdout":
        split_group_num = 1
    elif split_mode == "leave_one_out":
        group_lengths = {len(group_indexes) for group_indexes in ref_value2indexes.values()}
        if len(group_lengths) != 1:
            raise ValueError(
                "Dependent leave-one-out requires every reference group to have "
                "the same number of samples."
            )
        split_group_num = next(iter(group_lengths))
        if use_stratified_split:
            warnings.warn(
                "Stratification is ignored for dependent leave-one-out because "
                "exactly one sample per reference group is held out."
            )
            use_stratified_split = False
    else:
        split_group_num = folds

    target_map = _prepare_target2indexes(
        data_dicts,
        focused_target_key,
        indexes,
        focused_target_task_type,
        target2indexes,
        stratified_bin_size,
        stratified_bin_num,
    ) if use_stratified_split else None

    index_split_folds = {}
    ref_value_split_folds = {}
    for fold in range(split_group_num):
        split_dict = _init_split_set_dict()
        for ref_value, group_indexes in ref_value2indexes.items():
            if use_stratified_split:
                group_target_map = {
                    target: [index for index in target_indexes if index in set(group_indexes)]
                    for target, target_indexes in target_map.items()
                }
                group_test = []
                for target_indexes in group_target_map.values():
                    group_test.extend(
                        _get_test_indexes(
                            target_indexes,
                            fold,
                            split_group_num,
                            holdout_test_ratio,
                            split_mode,
                        )
                    )
            else:
                group_test = _get_test_indexes(
                    group_indexes,
                    fold,
                    split_group_num,
                    holdout_test_ratio,
                    split_mode,
                )
            group_test = _deduplicate_preserve_order(group_test)
            test_set = set(group_test)
            group_train_valid = [
                index for index in group_indexes if index not in test_set
            ]
            if use_stratified_split:
                group_train, group_valid = _stratified_train_valid_split(
                    group_train_valid,
                    target_map,
                    train_valid_ratio,
                )
            else:
                group_train, group_valid = _split_train_valid_indexes(
                    group_train_valid,
                    train_valid_ratio,
                )
            split_dict["test"].extend(group_test)
            split_dict["train"].extend(group_train)
            split_dict["valid"].extend(group_valid)

        _validate_split_dict(split_dict, indexes)
        index_split_folds[fold] = split_dict
        ref_value_split_folds[fold] = _build_ref_value_split_dict(
            data_dicts,
            split_dict,
            split_ref_key,
        )

    return index_split_folds, ref_value_split_folds


def data_split_unconstrained(
    data_dicts: List[Dict],
    split_ref_key: Any = None,
    split_mode: Literal["holdout", "kfold", "leave_one_out"] = "kfold",
    folds: int = 5,
    train_valid_ratio: float = 0.9,
    holdout_test_ratio: float = 0.2,
    use_stratified_split: bool = False,
    split_valid_by: Literal["index", "ref_key"] = "index",
    target2indexes: Optional[Dict] = None,
    focused_target_key: Any = "label",
    is_focused_key_multi_dim: bool = False,
    focused_target_task_type: Literal["c", "r"] = "c",
    stratified_bin_num: Optional[int] = None,
    stratified_bin_size: Optional[float] = None,
    used_indexes: Optional[List[int]] = None,
) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[Any]]]]:
    """Split sample indexes without enforcing reference-group independence.

    Parameters
    ----------
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

    Returns
    -------
    tuple[dict, dict]
        Index folds and corresponding reference-value folds.
    """
    _validate_split_parameters(
        split_mode,
        folds,
        train_valid_ratio,
        holdout_test_ratio,
    )
    if use_stratified_split and is_focused_key_multi_dim:
        raise ValueError("Stratified splitting does not support multidimensional targets.")

    indexes = _normalize_used_indexes(data_dicts, used_indexes)
    if split_mode == "holdout":
        split_group_num = 1
    elif split_mode == "leave_one_out":
        split_group_num = len(indexes)
        if use_stratified_split:
            warnings.warn(
                "Stratification is ignored for leave-one-out because each test "
                "fold contains exactly one sample."
            )
            use_stratified_split = False
    else:
        split_group_num = min(folds, len(indexes))
        if split_group_num < folds:
            warnings.warn(
                f"`folds` was reduced from {folds} to {split_group_num} because "
                "there are fewer samples than requested folds."
            )

    target_map = _prepare_target2indexes(
        data_dicts,
        focused_target_key,
        indexes,
        focused_target_task_type,
        target2indexes,
        stratified_bin_size,
        stratified_bin_num,
    ) if use_stratified_split else None

    index_split_folds = {}
    ref_value_split_folds = {}
    for fold in range(split_group_num):
        if use_stratified_split:
            test_indexes = []
            for target_indexes in target_map.values():
                test_indexes.extend(
                    _get_test_indexes(
                        target_indexes,
                        fold,
                        split_group_num,
                        holdout_test_ratio,
                        split_mode,
                    )
                )
            test_indexes = _deduplicate_preserve_order(test_indexes)
        else:
            test_indexes = _get_test_indexes(
                indexes,
                fold,
                split_group_num,
                holdout_test_ratio,
                split_mode,
            )
        test_set = set(test_indexes)
        train_valid_indexes = [index for index in indexes if index not in test_set]
        if use_stratified_split:
            train_indexes, valid_indexes = _stratified_train_valid_split(
                train_valid_indexes,
                target_map,
                train_valid_ratio,
            )
        else:
            train_indexes, valid_indexes = _split_train_valid_indexes(
                train_valid_indexes,
                train_valid_ratio,
            )
        split_dict = {
            "train": train_indexes,
            "valid": valid_indexes,
            "test": test_indexes,
        }
        _validate_split_dict(split_dict, indexes)
        index_split_folds[fold] = split_dict
        if split_ref_key is None:
            ref_value_split_folds[fold] = _init_split_set_dict()
        else:
            ref_value_split_folds[fold] = _build_ref_value_split_dict(
                data_dicts,
                split_dict,
                split_ref_key,
            )

    return index_split_folds, ref_value_split_folds


def _get_test_indexes(
    indexes_list,
    current_split_group,
    num_split_group,
    one_fold_test_ratio,
    split_mode="kfold",
):
    """Select the test indexes for one fold and split mode.

    Parameters
    ----------
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

    Returns
    -------
    list[int]
        Indexes assigned to the requested test fold.
    """
    indexes = list(indexes_list)
    if not indexes:
        return []
    if split_mode == "holdout":
        if current_split_group != 0:
            return []
        test_size = _get_holdout_size(len(indexes), one_fold_test_ratio)
        return indexes[-test_size:]
    if split_mode == "leave_one_out":
        position = len(indexes) - 1 - current_split_group
        return [indexes[position]] if position >= 0 else []
    chunks = _balanced_chunks(indexes, num_split_group, reverse=True)
    return chunks[current_split_group] if current_split_group < len(chunks) else []


def _distribute_indexes_to_folds(index_list, current_group_index):
    """Assign at most one index to a requested fold position.

    Parameters
    ----------
    index_list : Any
        Ordered original sample indexes or generic index-like values.
    current_group_index : Any
        Zero-based group position.

    Returns
    -------
    list[int]
        A one-item list or an empty list.
    """
    indexes = list(index_list)
    if 0 <= current_group_index < len(indexes):
        return [indexes[current_group_index]]
    return []


def remove_overlapped_seq_split(
    data_dicts,
    index_split_folds,
    anchore_index2seq_indexes,
    *,
    split_ref_key="group",
    is_test_train_no_seq_overlap: bool = True,
    is_train_valid_no_seq_overlap: bool = True,
    priority_order=("test", "train", "valid"),
    padding_index: int = -1,
):
    """Remove lower-priority anchors whose sequence contents overlap protected splits.

    Parameters
    ----------
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

    Returns
    -------
    tuple[dict, dict]
        Filtered index folds and rebuilt reference-value folds.
    """
    folds = _normalize_index_split_folds(index_split_folds)
    priority_order = list(priority_order)
    if set(priority_order) != {"train", "valid", "test"} or len(priority_order) != 3:
        raise ValueError(
            "`priority_order` must contain 'train', 'valid', and 'test' exactly once."
        )
    if not is_test_train_no_seq_overlap and not is_train_valid_no_seq_overlap:
        return folds, _build_ref_value_split_folds(
            data_dicts,
            folds,
            split_ref_key,
        )

    checked_pairs = set()
    if is_test_train_no_seq_overlap:
        checked_pairs.add(frozenset(("test", "train")))
    if is_train_valid_no_seq_overlap:
        checked_pairs.add(frozenset(("train", "valid")))

    def sequence_set(anchor):
        """Return non-padding sequence indexes used by one anchor."""
        sequence = anchore_index2seq_indexes.get(anchor, [anchor])
        return {index for index in sequence if index != padding_index}

    post_folds = {}
    for fold, split_dict in folds.items():
        kept = _init_split_set_dict()
        kept_sequence_union = {key: set() for key in kept}
        for split_key in priority_order:
            for anchor in split_dict[split_key]:
                anchor_sequence = sequence_set(anchor)
                overlaps = False
                for higher_key in priority_order[: priority_order.index(split_key)]:
                    if frozenset((split_key, higher_key)) not in checked_pairs:
                        continue
                    if anchor_sequence & kept_sequence_union[higher_key]:
                        overlaps = True
                        break
                if not overlaps:
                    kept[split_key].append(anchor)
                    kept_sequence_union[split_key].update(anchor_sequence)
        post_folds[fold] = kept

    return post_folds, _build_ref_value_split_folds(
        data_dicts,
        post_folds,
        split_ref_key,
    )


def _summary_seq_index_list(anchore_index_list, anchore_index2seq_index):
    """Collect unique sequence indexes referenced by a set of anchors.

    Parameters
    ----------
    anchore_index_list : Any
        Anchor indexes whose sequence contents are summarized.
    anchore_index2seq_index : Any
        Mapping from anchor indexes to sequence index lists.

    Returns
    -------
    list[int]
        Sorted unique sequence indexes.
    """
    summary = set()
    for index in anchore_index_list:
        summary.update(anchore_index2seq_index.get(index, [index]))
    return sorted(summary)


def _validate_split_parameters(split_mode, folds, train_valid_ratio, holdout_test_ratio):
    """Validate common split modes, fold counts, and ratios.

    Parameters
    ----------
    split_mode : Any
        Test-split strategy: holdout, k-fold, or leave-one-out.
    folds : Any
        Requested number of folds for k-fold splitting.
    train_valid_ratio : Any
        Fraction of non-test data assigned to training.
    holdout_test_ratio : Any
        Fraction assigned to test in holdout mode.
    """
    if split_mode not in {"holdout", "kfold", "leave_one_out"}:
        raise ValueError("Invalid split mode.")
    if folds <= 0:
        raise ValueError("`folds` must be positive.")
    if not 0.0 < train_valid_ratio < 1.0:
        raise ValueError("`train_valid_ratio` must be between 0 and 1.")
    if not 0.0 < holdout_test_ratio < 1.0:
        raise ValueError("`holdout_test_ratio` must be between 0 and 1.")


def _normalize_used_indexes(data_dicts, used_indexes):
    """Resolve and validate the original sample indexes used for preparation.

    Parameters
    ----------
    data_dicts : Any
        Ordered sample-level dictionaries indexed by original sample position.
    used_indexes : Any
        Original sample indexes to process. ``None`` selects all samples.

    Returns
    -------
    list[int]
        Validated original sample indexes.
    """
    indexes = list(range(len(data_dicts))) if used_indexes is None else list(used_indexes)
    if not indexes:
        raise ValueError("`used_indexes` must contain at least one index.")
    if len(indexes) != len(set(indexes)):
        raise ValueError("`used_indexes` contains duplicated indexes.")
    invalid = [index for index in indexes if index < 0 or index >= len(data_dicts)]
    if invalid:
        raise IndexError(f"Indexes outside `data_dicts` were found: {invalid[:5]}.")
    return indexes


def _balanced_chunks(values, num_chunks, reverse=False):
    """Partition ordered values into nearly equal chunks.

    Parameters
    ----------
    values : Any
        Values to process.
    num_chunks : Any
        Number of chunks to create.
    reverse : Any
        Whether chunk order is reversed after partitioning.

    Returns
    -------
    list[list]
        Nearly equal ordered chunks.
    """
    values = list(values)
    if num_chunks <= 0:
        raise ValueError("`num_chunks` must be positive.")
    quotient, remainder = divmod(len(values), num_chunks)
    chunks = []
    start = 0
    for chunk_index in range(num_chunks):
        size = quotient + (1 if chunk_index < remainder else 0)
        chunks.append(values[start:start + size])
        start += size
    return list(reversed(chunks)) if reverse else chunks


def _get_holdout_size(num_items, ratio):
    """Calculate a nonempty holdout size while retaining training data when possible.

    Parameters
    ----------
    num_items : Any
        Number of available items.
    ratio : Any
        Requested fraction.

    Returns
    -------
    int
        Number of items assigned to holdout test data.
    """
    if num_items <= 0:
        return 0
    test_size = max(1, int(np.ceil(num_items * ratio)))
    if num_items > 1:
        test_size = min(test_size, num_items - 1)
    return test_size


def _split_train_valid_indexes(indexes, train_ratio):
    """Split ordered indexes into nonempty train and validation subsets when possible.

    Parameters
    ----------
    indexes : Any
        Ordered candidate indexes.
    train_ratio : Any
        Fraction assigned to training.

    Returns
    -------
    tuple[list, list]
        Training and validation index lists.
    """
    indexes = list(indexes)
    if len(indexes) <= 1:
        return indexes, []
    split_index = int(len(indexes) * train_ratio)
    split_index = min(max(split_index, 1), len(indexes) - 1)
    return indexes[:split_index], indexes[split_index:]


def _stratified_train_valid_split(indexes, target2indexes, train_ratio):
    """Split candidate indexes within each target stratum.

    Parameters
    ----------
    indexes : Any
        Ordered candidate indexes.
    target2indexes : Any
        Optional mapping from targets or bins to original sample indexes.
    train_ratio : Any
        Fraction assigned to training.

    Returns
    -------
    tuple[list, list]
        Stratified training and validation index lists.
    """
    candidate_set = set(indexes)
    train, valid = [], []
    assigned = set()
    for target_indexes in target2indexes.values():
        selected = [index for index in target_indexes if index in candidate_set]
        train_part, valid_part = _split_train_valid_indexes(selected, train_ratio)
        train.extend(train_part)
        valid.extend(valid_part)
        assigned.update(selected)
    unassigned = [index for index in indexes if index not in assigned]
    if unassigned:
        warnings.warn(
            "Some candidate indexes were absent from `target2indexes`; they were "
            "split without stratification."
        )
        train_part, valid_part = _split_train_valid_indexes(unassigned, train_ratio)
        train.extend(train_part)
        valid.extend(valid_part)
    return _deduplicate_preserve_order(train), _deduplicate_preserve_order(valid)


def _prepare_target2indexes(
    data_dicts,
    target_key,
    used_indexes,
    task_type,
    target2indexes,
    stratified_bin_size,
    stratified_bin_num,
):
    """Create or filter a target-to-original-index mapping.

    Parameters
    ----------
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

    Returns
    -------
    dict
        Target or bin mapping limited to selected indexes.
    """
    if target2indexes is None:
        return get_target2indexes(
            data_dicts,
            target_key,
            used_indexes=used_indexes,
            task_type=task_type,
            stratified_bin_size=stratified_bin_size,
            stratified_bin_num=stratified_bin_num,
        )
    allowed = set(used_indexes)
    return {
        target: [index for index in indexes if index in allowed]
        for target, indexes in target2indexes.items()
    }


def _normalize_fold_override(override):
    """Normalize split overrides into a list of per-fold value lists.

    Parameters
    ----------
    override : Any
        User-provided split values in shared or per-fold form.

    Returns
    -------
    list[list]
        Per-fold override values.
    """
    if isinstance(override, Mapping):
        return [list(override[key]) for key in sorted(override)]
    values = list(override)
    if not values:
        raise ValueError("A split override cannot be empty.")
    first = values[0]
    if isinstance(first, (list, tuple, set, np.ndarray)):
        return [list(group) for group in values]
    return [values]


def _get_fold_override(override, fold):
    """Read an override for one fold, allowing one shared override.

    Parameters
    ----------
    override : Any
        User-provided split values in shared or per-fold form.
    fold : Any
        Zero-based fold identifier.

    Returns
    -------
    list or None
        Values assigned to the requested fold.
    """
    if override is None:
        return None
    groups = _normalize_fold_override(override)
    if len(groups) == 1:
        return groups[0]
    if fold >= len(groups):
        raise ValueError(f"No override values were provided for fold {fold}.")
    return groups[fold]


def _resolve_train_valid_ref_values(
    candidate_values,
    train_ratio,
    train_override,
    valid_override,
):
    """Resolve automatic or user-defined train and validation reference groups.

    Parameters
    ----------
    candidate_values : Any
        Reference values not assigned to test.
    train_ratio : Any
        Fraction assigned to training.
    train_override : Any
        Optional explicitly assigned training reference values.
    valid_override : Any
        Optional explicitly assigned validation reference values.

    Returns
    -------
    tuple[list, list]
        Training and validation reference values.
    """
    candidate_values = list(candidate_values)
    candidate_set = set(candidate_values)
    if train_override is None and valid_override is None:
        return _split_train_valid_indexes(candidate_values, train_ratio)
    train_values = None if train_override is None else list(train_override)
    valid_values = None if valid_override is None else list(valid_override)
    if train_values is not None:
        _validate_ref_values(train_values, candidate_values, "train")
    if valid_values is not None:
        _validate_ref_values(valid_values, candidate_values, "valid")
    if train_values is None:
        valid_set = set(valid_values)
        train_values = [value for value in candidate_values if value not in valid_set]
    elif valid_values is None:
        train_set = set(train_values)
        valid_values = [value for value in candidate_values if value not in train_set]
    if set(train_values) & set(valid_values):
        raise ValueError("Train and validation reference overrides overlap.")
    if set(train_values) | set(valid_values) != candidate_set:
        raise ValueError(
            "Train and validation reference overrides must cover every non-test "
            "reference value exactly once."
        )
    return (
        _deduplicate_preserve_order(train_values),
        _deduplicate_preserve_order(valid_values),
    )


def _validate_ref_values(values, available_values, split_name):
    """Ensure override reference values exist in the available groups.

    Parameters
    ----------
    values : Any
        Values to process.
    available_values : Any
        Reference values available for assignment.
    split_name : Any
        Human-readable split name used in error messages.
    """
    available_set = set(available_values)
    unknown = [value for value in values if value not in available_set]
    if unknown:
        raise KeyError(
            f"Unknown {split_name} reference values were provided: {unknown}."
        )


def _build_ref_value_split_dict(data_dicts, split_dict, split_ref_key):
    """Derive unique reference values for each split from original indexes.

    Parameters
    ----------
    data_dicts : Any
        Ordered sample-level dictionaries indexed by original sample position.
    split_dict : Any
        One train/validation/test index dictionary.
    split_ref_key : Any
        Dictionary key defining groups such as speakers, participants, or sessions.

    Returns
    -------
    dict
        Unique reference values for train, validation, and test.
    """
    result = _init_split_set_dict()
    for split_key, indexes in split_dict.items():
        result[split_key] = _deduplicate_preserve_order(
            data_dicts[index][split_ref_key] for index in indexes
        )
    return result


def _build_ref_value_split_folds(data_dicts, index_split_folds, split_ref_key):
    """Derive reference-value summaries for every index fold.

    Parameters
    ----------
    data_dicts : Any
        Ordered sample-level dictionaries indexed by original sample position.
    index_split_folds : Any
        Per-fold train, validation, and test original indexes.
    split_ref_key : Any
        Dictionary key defining groups such as speakers, participants, or sessions.

    Returns
    -------
    dict
        Reference-value summaries keyed by fold.
    """
    folds = _normalize_index_split_folds(index_split_folds)
    if split_ref_key is None:
        return {fold: _init_split_set_dict() for fold in folds}
    return {
        fold: _build_ref_value_split_dict(data_dicts, split_dict, split_ref_key)
        for fold, split_dict in folds.items()
    }


def _normalize_index_split_folds(index_split_folds):
    """Normalize list- or mapping-based folds to a consistent integer-keyed mapping.

    Parameters
    ----------
    index_split_folds : Any
        Per-fold train, validation, and test original indexes.

    Returns
    -------
    dict
        Integer-keyed fold mapping with standard split keys.
    """
    if isinstance(index_split_folds, Mapping):
        fold_items = sorted(index_split_folds.items())
    else:
        fold_items = list(enumerate(index_split_folds))
    normalized = {}
    for fold, split_dict in fold_items:
        missing = {"train", "valid", "test"} - set(split_dict)
        if missing:
            raise KeyError(f"Fold {fold} is missing split keys: {sorted(missing)}.")
        normalized[int(fold)] = {
            key: list(split_dict[key]) for key in ("train", "valid", "test")
        }
    return normalized


def _validate_split_dict(split_dict, allowed_indexes):
    """Ensure one fold is disjoint, complete, and limited to allowed indexes.

    Parameters
    ----------
    split_dict : Any
        One train/validation/test index dictionary.
    allowed_indexes : Any
        Complete set of indexes that must be assigned exactly once.
    """
    allowed_set = set(allowed_indexes)
    split_sets = {key: set(values) for key, values in split_dict.items()}
    for split_key, indexes in split_dict.items():
        if len(indexes) != len(set(indexes)):
            raise ValueError(f"Duplicated indexes were found in the {split_key} set.")
        unknown = [index for index in indexes if index not in allowed_set]
        if unknown:
            raise IndexError(
                f"The {split_key} set contains indexes outside `used_indexes`: "
                f"{unknown[:5]}."
            )
    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = split_sets[first] & split_sets[second]
        if overlap:
            raise ValueError(
                f"The {first} and {second} sets overlap at indexes "
                f"{sorted(overlap)[:5]}."
            )
    covered = split_sets["train"] | split_sets["valid"] | split_sets["test"]
    if covered != allowed_set:
        missing = sorted(allowed_set - covered)
        raise ValueError(f"Some used indexes were not assigned to a split: {missing[:5]}.")


def _flatten(iterables):
    """Flatten one level of nested iterables.

    Parameters
    ----------
    iterables : Any
        Nested iterables to flatten by one level.

    Returns
    -------
    list
        One-level flattened values.
    """
    return [item for iterable in iterables for item in iterable]


def _deduplicate_preserve_order(values):
    """Remove repeated values while preserving first-occurrence order.

    Parameters
    ----------
    values : Any
        Values to process.

    Returns
    -------
    list
        Unique values in first-occurrence order.
    """
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


# ---------- Functions about target info ---------- #
def get_target_info_cls(
    data_dicts,
    target_ref_key,
    used_indexes=None,
):
    """Build class statistics, class-ID mappings, and original-index groups.

    Parameters
    ----------
    data_dicts : Any
        Ordered sample-level dictionaries indexed by original sample position.
    target_ref_key : Any
        Dictionary key containing the target value.
    used_indexes : Any
        Original sample indexes to process. ``None`` selects all samples.

    Returns
    -------
    tuple
        Class statistics, class mappings, and target-to-index mapping.
    """
    indexes = _normalize_used_indexes(data_dicts, used_indexes)
    target_list = get_scalar_target_list(data_dicts, target_ref_key, indexes)
    target2id = {}
    id2target = {}
    target_stats = {}
    target2indexes = {}
    for original_index, target in zip(indexes, target_list):
        if target not in target2id:
            target_id = len(target2id)
            target2id[target] = target_id
            id2target[target_id] = target
        target_stats[target] = target_stats.get(target, 0) + 1
        target2indexes.setdefault(target, []).append(original_index)
    return target_stats, target2id, id2target, target2indexes


def get_target_info_regression(
    data_dicts,
    target_ref_key,
    used_indexes=None,
    stratified_bin_size=None,
    stratified_bin_num=None,
    bin_closed_side: Literal["upper", "lower"] = "lower",
):
    """Bin scalar regression targets and collect bin statistics and indexes.

    Parameters
    ----------
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

    Returns
    -------
    tuple
        Bin statistics, bin ranges, and bin-to-index mapping.
    """
    indexes = _normalize_used_indexes(data_dicts, used_indexes)
    target_list = get_scalar_target_list(data_dicts, target_ref_key, indexes)
    if not all(isinstance(value, (int, float, np.number)) for value in target_list):
        raise TypeError("Regression targets must be numeric scalars.")
    bin_ranges = _make_regression_bin_ranges(
        target_list,
        stratified_bin_size,
        stratified_bin_num,
    )
    target_stats = {i: 0 for i in range(len(bin_ranges))}
    target2indexes = {i: [] for i in range(len(bin_ranges))}
    target_bin_ranges = {i: bounds for i, bounds in enumerate(bin_ranges)}
    for original_index, target in zip(indexes, target_list):
        bin_id = _find_bin_id(target, target_bin_ranges, bin_closed_side)
        target_stats[bin_id] += 1
        target2indexes[bin_id].append(original_index)
    return target_stats, target_bin_ranges, target2indexes


def get_target2indexes(
    data_dicts,
    target_ref_key,
    used_indexes=None,
    task_type="c",
    stratified_bin_size: Optional[float] = None,
    stratified_bin_num: Optional[int] = 10,
):
    """Map each class or regression bin to its original sample indexes.

    Parameters
    ----------
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

    Returns
    -------
    dict
        Mapping from class values or regression bins to original indexes.
    """
    if task_type == "c":
        _, _, _, target2indexes = get_target_info_cls(
            data_dicts,
            target_ref_key,
            used_indexes=used_indexes,
        )
    elif task_type == "r":
        _, _, target2indexes = get_target_info_regression(
            data_dicts,
            target_ref_key,
            used_indexes=used_indexes,
            stratified_bin_num=stratified_bin_num,
            stratified_bin_size=stratified_bin_size,
        )
    else:
        raise ValueError("`task_type` must be 'c' or 'r'.")
    return target2indexes


def get_stratified_bin_info(
    data_dicts,
    target_ref_key,
    used_indexes=None,
    bin_size=None,
    bin_num=None,
    target_list=None,
):
    """Calculate effective regression-bin width and count.

    Parameters
    ----------
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

    Returns
    -------
    tuple[float, int]
        Effective bin width and number of bins.
    """
    values = (
        get_scalar_target_list(
            data_dicts,
            target_ref_key,
            _normalize_used_indexes(data_dicts, used_indexes),
        )
        if target_list is None
        else list(target_list)
    )
    ranges = _make_regression_bin_ranges(values, bin_size, bin_num)
    if len(ranges) == 1:
        return ranges[0][1] - ranges[0][0], 1
    return ranges[0][1] - ranges[0][0], len(ranges)


def _make_regression_bin_ranges(values, bin_size=None, bin_num=None):
    """Create contiguous bin boundaries spanning observed target values.

    Parameters
    ----------
    values : Any
        Values to process.
    bin_size : Any
        Requested regression-bin width.
    bin_num : Any
        Requested number of regression bins.

    Returns
    -------
    list[tuple[float, float]]
        Contiguous lower/upper bin boundaries.
    """
    if not values:
        raise ValueError("Cannot create bins from an empty target list.")
    target_min = float(min(values))
    target_max = float(max(values))
    if bin_size is not None:
        if bin_size <= 0:
            raise ValueError("`bin_size` must be positive.")
        if target_max == target_min:
            return [(target_min, target_max)]
        bin_num = max(1, int(np.ceil((target_max - target_min) / bin_size)))
        edges = [target_min + i * bin_size for i in range(bin_num)]
        edges.append(target_max)
    elif bin_num is not None:
        if bin_num <= 0:
            raise ValueError("`bin_num` must be positive.")
        if target_max == target_min:
            return [(target_min, target_max)]
        edges = np.linspace(target_min, target_max, bin_num + 1).tolist()
    else:
        raise ValueError("Specify either `bin_size` or `bin_num`.")
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _find_bin_id(value, bin_ranges, bin_closed_side):
    """Find the bin containing one scalar regression target.

    Parameters
    ----------
    value : Any
        Scalar regression target.
    bin_ranges : Any
        Mapping from bin IDs to lower and upper boundaries.
    bin_closed_side : Any
        Boundary convention for regression bins.

    Returns
    -------
    int
        Identifier of the containing regression bin.
    """
    last_bin_id = max(bin_ranges)
    for bin_id, (lower, upper) in bin_ranges.items():
        if lower == upper:
            return bin_id
        if bin_closed_side == "lower":
            if lower <= value < upper or (bin_id == last_bin_id and value == upper):
                return bin_id
        elif bin_closed_side == "upper":
            if lower < value <= upper or (bin_id == 0 and value == lower):
                return bin_id
        else:
            raise ValueError("`bin_closed_side` must be 'lower' or 'upper'.")
    raise ValueError(f"Target value {value!r} does not belong to any bin.")


def _convert_value_to_bin(value, bin_ranges, bin_closed_side):
    """Convert a scalar target to the configured boundary representative of its bin.

    Parameters
    ----------
    value : Any
        Scalar regression target.
    bin_ranges : Any
        Mapping from bin IDs to lower and upper boundaries.
    bin_closed_side : Any
        Boundary convention for regression bins.

    Returns
    -------
    float
        Selected lower or upper boundary representing the target bin.
    """
    bin_id = _find_bin_id(value, bin_ranges, bin_closed_side)
    lower, upper = bin_ranges[bin_id]
    return lower if bin_closed_side == "lower" else upper


# ---------- Data and config save and load ---------- #
# generate and read config template for setting configs by json
def generate_config_template(
    file: str = "./data_prep_config.json",
    cls_target_keys: Union[List, str, None] = None,
    regression_target_keys: Union[List, str, None] = None,
    input_keys: Union[List, str, None] = None
):
    """Write a JSON configuration template for specified data keys.

    Parameters
    ----------
    file : str
        Path to the JSON configuration file.
    cls_target_keys : Union[List, str, None]
        Classification target key or keys.
    regression_target_keys : Union[List, str, None]
        Regression target key or keys.
    input_keys : Union[List, str, None]
        Model-input key or keys.
    """
    config_list = []
    unified_cls_keys = _unify_to_list(
        cls_target_keys) if cls_target_keys != None else []
    unified_regression_keys = _unify_to_list(
        regression_target_keys) if regression_target_keys != None else []
    unified_input_keys = _unify_to_list(
        input_keys) if input_keys != None else []
    if (len(unified_cls_keys) == 0
        and len(unified_regression_keys) == 0
        and len(unified_input_keys) == 0):
        raise (ValueError(
            "cls_target_keys, regression_target_keys, input_keys are all None in generate_config_template()."))
    for config_type, keys in zip(["c", "r", "i"], [unified_cls_keys, unified_regression_keys, unified_input_keys]):
        if(len(keys) > 0):
            if(config_type == "c"):
                config_list.append(ClassificationTargetConfig(keys))
            if(config_type == "r"):
                config_list.append(RegressionTargetConfig(keys))
            if(config_type == "i"):
                config_list.append(InputConfig(keys))
    data_prep_config = DataPrepConfig(data_configs=config_list)
    with open(file, "w") as json_o:
        json.dump(data_prep_config.to_json(), json_o, indent=2)


def load_config(file: str = "./data_prep_config.json"):
    """Load a JSON configuration file into configuration objects.

    Parameters
    ----------
    file : str
        Path to the JSON configuration file.

    Returns
    -------
    DataPrepConfig
        Reconstructed data-preparation configuration.
    """
    with open(file, "r") as json_i:
        config_json = json.load(json_i)
    return make_config_from_json(config_json)


def make_config_from_json(config_json):
    """Reconstruct nested configuration objects from parsed JSON data.

    Parameters
    ----------
    config_json : Any
        Parsed configuration representation produced by ``to_json``.

    Returns
    -------
    DataPrepConfig
        Reconstructed data-preparation configuration.
    """
    data_configs = []
    for config in config_json[:-1]:
        data_configs.append(globals()[config[0]](**config[1]))
    data_prep_config_fields = config_json[-1][1]
    data_prep_config_fields["data_configs"] = data_configs
    data_prep_config = globals()[config_json[-1][0]](**data_prep_config_fields)
    return data_prep_config


# save and load config and data
def save_data(
    collected_data,
    info_dict,
    data_prep_config,
    save_dir="./DataExperiment",
    overwrite_data=False
):
    # the user can flexibly choose to used the save root in the config, or specify a new save root
    """Save prepared data and metadata using explicit or configured options.

    Parameters
    ----------
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
    """
    save_flag = False
    save_data_path = os.path.join(save_dir, "Data.pkl")
    if(os.path.exists(save_data_path)):
        if(overwrite_data):
            save_flag = True
    else:
        save_flag = True
    if(save_flag):
        # save paths
        save_data_path = os.path.join(save_dir, "Data.pkl")
        save_info_path = os.path.join(save_dir, "Info.pkl")
        data_prep_config_path = os.path.join(save_dir, "DataPrepConfig.json")
        if(os.path.exists(save_dir) == False):
            os.makedirs(save_dir)
        with open(save_data_path, "wb") as pkl_o:
            pickle.dump(collected_data, pkl_o)
        with open(save_info_path, "wb") as pkl_o:
            pickle.dump(info_dict, pkl_o)
        with open(data_prep_config_path, "w") as json_o:
            json.dump(data_prep_config.to_json(), json_o, indent=2)


def load_data(data_dir="./DataExperiment"):
    """Load prepared data, metadata, and configuration from a directory.

    Parameters
    ----------
    data_dir : Any
        Directory containing serialized prepared-data files.

    Returns
    -------
    tuple
        Prepared data, metadata, and configuration.
    """
    save_data_path = os.path.join(data_dir, "Data.pkl")
    save_info_path = os.path.join(data_dir, "Info.pkl")
    data_prep_config_path = os.path.join(data_dir, "DataPrepConfig.json")
    # init data config
    data_prep_config = load_config(data_prep_config_path)
    # load collected_data and info_dict
    with open(save_data_path, "rb") as pkl_i:
        collected_data = pickle.load(pkl_i)
    with open(save_info_path, "rb") as pkl_i:
        info_dict = pickle.load(pkl_i)
    return collected_data, info_dict, data_prep_config


# ---------- Other processings ---------- #
def get_target_list(data_dicts, target_ref_key, used_indexes=None):
    """Collect target values from selected original sample indexes.

    Parameters
    ----------
    data_dicts : Any
        Ordered sample-level dictionaries indexed by original sample position.
    target_ref_key : Any
        Dictionary key containing the target value.
    used_indexes : Any
        Original sample indexes to process. ``None`` selects all samples.

    Returns
    -------
    list
        Target values in selected-index order.
    """
    indexes = _normalize_used_indexes(data_dicts, used_indexes)
    return [data_dicts[index][target_ref_key] for index in indexes]


# convert target to scalar for target statistics and processing
def get_scalar_target_list(data_dicts, target_ref_key, used_indexes=None):
    """Collect and normalize selected target values to Python scalars.

    Parameters
    ----------
    data_dicts : Any
        Ordered sample-level dictionaries indexed by original sample position.
    target_ref_key : Any
        Dictionary key containing the target value.
    used_indexes : Any
        Original sample indexes to process. ``None`` selects all samples.

    Returns
    -------
    list
        Python scalar targets in selected-index order.
    """
    indexes = _normalize_used_indexes(data_dicts, used_indexes)
    return [
        convert_to_python_scalar(data_dicts[index][target_ref_key])
        for index in indexes
    ]


def convert_to_python_scalar(target_value):
    """Convert a scalar-like Python, NumPy, or PyTorch value to a Python scalar.

    Parameters
    ----------
    target_value : Any
        Scalar or one-element scalar-like value to convert.

    Returns
    -------
    object
        Equivalent Python scalar value.
    """
    if isinstance(target_value, (int, float, complex, str, bool)):
        return target_value
    if isinstance(target_value, np.generic):
        return target_value.item()
    if isinstance(target_value, (list, tuple)):
        if len(target_value) != 1:
            raise ValueError(
                "A scalar target was expected, but a sequence with more than one "
                "element was provided."
            )
        warnings.warn(
            "A one-element sequence was provided as a scalar target and was "
            "converted automatically."
        )
        return convert_to_python_scalar(target_value[0])
    if isinstance(target_value, np.ndarray):
        if target_value.size != 1:
            raise ValueError(
                "A scalar target was expected, but a NumPy array with more than "
                "one element was provided."
            )
        warnings.warn(
            "A one-element NumPy array was provided as a scalar target and was "
            "converted automatically."
        )
        return target_value.item()
    if TORCH_AVAILABLE and isinstance(target_value, torch.Tensor):
        if target_value.numel() != 1:
            raise ValueError(
                "A scalar target was expected, but a tensor with more than one "
                "element was provided."
            )
        warnings.warn(
            "A one-element tensor was provided as a scalar target and was "
            "converted automatically."
        )
        return target_value.item()
    raise TypeError(
        f"Unsupported scalar target type: {type(target_value).__name__}."
    )


# calcualte miu and sigma for given key over given indexes, for getting z-score miu and sigma for overall or for each train/valid/test in fold
def calculate_miu_sigma(data_dicts, ref_key, index_list):
    """Calculate feature-wise mean and standard deviation over selected samples.

    Parameters
    ----------
    data_dicts : Any
        Ordered sample-level dictionaries indexed by original sample position.
    ref_key : Any
        Dictionary key containing values used in a calculation.
    index_list : Any
        Ordered original sample indexes or generic index-like values.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Feature-wise mean and standard deviation.
    """
    collected_list = []
    for index in index_list:
        collected_list.append(data_dicts[index][ref_key])
    collected_array = np.concatenate(collected_list, axis=0)
    miu = np.mean(collected_array, axis=0)
    sigma = np.std(collected_array, axis=0)
    return miu, sigma


def _init_split_set_dict():
    """Create an empty train/validation/test dictionary.

    Returns
    -------
    dict[str, list]
        Empty lists for ``train``, ``valid``, and ``test``.
    """
    split_dict = {}
    for key in ["train", "valid", "test"]:
        split_dict[key] = []
    return split_dict


def _unify_to_list(content):
    """Normalize a scalar, tuple, or NumPy array into a Python list.

    Parameters
    ----------
    content : Any
        Scalar or collection to normalize into a list.

    Returns
    -------
    list
        Normalized Python list.
    """
    if isinstance(content, dict):
        raise TypeError("The given key specification must not be a dictionary.")
    if isinstance(content, list):
        return content
    if isinstance(content, np.ndarray):
        return content.tolist()
    if isinstance(content, tuple):
        return list(content)
    return [content]
