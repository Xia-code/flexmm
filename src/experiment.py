# -*- coding: utf-8 -*-
"""Experiment configuration, iteration, data conversion, and evaluation tools.

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
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import pickle
import random
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None
    TORCH_AVAILABLE = False
    USE_GPU_DEFAULT = False
else:
    TORCH_AVAILABLE = True
    USE_GPU_DEFAULT = torch.cuda.is_available()

from flexmm import data_prep


DataLevel = Literal["raw", "dataset", "dataloader"]
DataRepresentation = Literal["original", "pt"]
TaskType = Literal["c", "r"]


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Configure experiment combinations, data output, and reproducibility.

    Parameters
    ----------
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

    Attributes
    ----------
    input_keys : list[Any]
        Resolved unique input keys.
    input_combs : list[list[Any]]
        Resolved input combinations.
    target_keys : list[Any]
        Resolved target keys.
    """

    experiment_input_keys: Union[Any, List[Any]] = field(default_factory=list)
    generate_input_comb: bool = True
    experiment_target_keys: Union[Any, List[Any]] = field(default_factory=list)
    input_comb_custom: Optional[List[Union[Any, List[Any]]]] = None
    input_key_abbr: Optional[Union[Dict[Any, Any], List[Any]]] = None
    random_seed: Optional[int] = None
    random_seed_scope: List[str] = field(
        default_factory=lambda: ["random", "numpy", "torch"]
    )
    data_level: DataLevel = "raw"
    data_representation: DataRepresentation = "original"
    load_prepared_data: bool = True
    store_dir: str = "./ExperimentStore"
    debug_flag: int = 0

    def __post_init__(self) -> None:
        """Normalize configuration fields and validate the resolved settings."""
        self.target_keys = _unify_to_list(self.experiment_target_keys)

        if self.input_comb_custom is not None:
            self.input_combs = [
                _unify_to_list(combination)
                for combination in self.input_comb_custom
            ]
            if not self.input_combs or any(len(combination) == 0 for combination in self.input_combs):
                raise ValueError("`input_comb_custom` must contain non-empty combinations.")
            self.input_keys = _unique_in_order(
                key for combination in self.input_combs for key in combination
            )
        else:
            self.input_keys = _unify_to_list(self.experiment_input_keys)
            if not self.input_keys:
                raise ValueError("At least one experiment input key must be configured.")
            if self.generate_input_comb:
                self.input_combs = generate_key_combs(self.input_keys)
            else:
                self.input_combs = [list(self.input_keys)]

        if not self.target_keys:
            raise ValueError("At least one experiment target key must be configured.")

        if self.random_seed is None:
            self.random_seed = random.SystemRandom().randint(0, 99_999)

        self.input_key_abbr = self._normalize_input_key_abbr(self.input_key_abbr)
        self.assert_config()

    def _normalize_input_key_abbr(
        self,
        abbreviations: Optional[Union[Dict[Any, Any], List[Any]]],
    ) -> Dict[Any, Any]:
        """Normalize input-key abbreviations to a complete dictionary.

        Parameters
        ----------
        abbreviations : dict, list, or None
            User-supplied abbreviations.

        Returns
        -------
        dict
            Mapping from every resolved input key to its abbreviation.
        """
        if abbreviations is None:
            return {key: index for index, key in enumerate(self.input_keys)}

        if isinstance(abbreviations, Mapping):
            missing = [key for key in self.input_keys if key not in abbreviations]
            extra = [key for key in abbreviations if key not in self.input_keys]
            if missing or extra:
                raise KeyError(
                    "`input_key_abbr` must contain exactly the resolved input keys. "
                    f"Missing={missing}, extra={extra}."
                )
            return dict(abbreviations)

        if isinstance(abbreviations, list):
            if len(abbreviations) != len(self.input_keys):
                raise ValueError(
                    "List-form `input_key_abbr` must have the same length as "
                    f"the resolved input keys ({len(self.input_keys)})."
                )
            return dict(zip(self.input_keys, abbreviations))

        raise TypeError("`input_key_abbr` must be a dictionary, list, or None.")

    def assert_config(self) -> None:
        """Validate general experiment settings.

        Raises
        ------
        ValueError
            If a categorical setting or random-seed scope is invalid.
        """
        if self.data_level not in {"raw", "dataset", "dataloader"}:
            raise ValueError(
                "`data_level` must be 'raw', 'dataset', or 'dataloader', "
                f"but {self.data_level!r} was given."
            )
        if self.data_representation not in {"original", "pt"}:
            raise ValueError(
                "`data_representation` must be 'original' or 'pt', "
                f"but {self.data_representation!r} was given."
            )
        allowed_scopes = {"random", "numpy", "torch"}
        invalid_scopes = [
            scope for scope in self.random_seed_scope if scope not in allowed_scopes
        ]
        if invalid_scopes:
            raise ValueError(
                f"Unsupported random seed scopes: {invalid_scopes}. "
                f"Supported scopes are {sorted(allowed_scopes)}."
            )

    def to_json(self) -> tuple[str, Dict[str, Any]]:
        """Convert the dataclass fields to the serializable config format.

        Returns
        -------
        tuple[str, dict]
            Class name and dataclass field dictionary.
        """
        return self.__class__.__name__, asdict(self)


@dataclass
class TorchExperimentConfig(ExperimentConfig):
    """Extend :class:`ExperimentConfig` with PyTorch DataLoader settings.

    Parameters
    ----------
    train_batch_size, valid_batch_size, test_batch_size : int
        Batch sizes used for the corresponding splits.
    shuffle_train_data, shuffle_valid_data, shuffle_test_data : bool
        Whether the corresponding DataLoader shuffles its dataset.
    use_gpu : bool
        Whether tensor-conversion helpers may place tensors on CUDA. Moving
        tensors to the model device inside the training loop is generally more
        flexible, so the default dataset workflow keeps tensors on CPU.
    """

    train_batch_size: int = 4
    valid_batch_size: int = 1
    test_batch_size: int = 1
    shuffle_train_data: bool = True
    shuffle_valid_data: bool = False
    shuffle_test_data: bool = False
    use_gpu: bool = USE_GPU_DEFAULT

    def __post_init__(self) -> None:
        """Initialize inherited fields and validate PyTorch-specific values."""
        super().__post_init__()
        for name in ("train_batch_size", "valid_batch_size", "test_batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        if self.data_level in {"dataset", "dataloader"} and not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required when `data_level` is 'dataset' or 'dataloader'."
            )
        if self.data_representation == "pt" and not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for `data_representation='pt'`.")
        if self.use_gpu and not USE_GPU_DEFAULT:
            warnings.warn(
                "`use_gpu=True` was requested, but CUDA is not available. "
                "Tensor conversion will remain on CPU.",
                UserWarning,
            )
            self.use_gpu = False


# ---------------------------------------------------------------------------
# Experiment orchestration data structures
# ---------------------------------------------------------------------------
@dataclass
class ExperimentContext:
    """Carry runtime metadata for one input-combination and fold condition.

    Parameters
    ----------
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
    """

    fold: Any
    comb_index: int
    comb_name: str
    input_comb: List[Any]
    target_keys: List[Any]
    split_indexes: Dict[str, List[int]]
    prepared_split_indexes: Dict[str, List[int]]
    ref_value_splits: Dict[str, List[Any]]
    standardization_info: Dict[Any, Dict[str, Any]]
    info_dict: Dict[str, Any]
    exp_config: ExperimentConfig
    data_prep_config: Any
    output_dir: str
    seed: int
    user_extras: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow dictionary representation of the context.

        Returns
        -------
        dict
            Context fields keyed by field name.
        """
        return {
            "fold": self.fold,
            "comb_index": self.comb_index,
            "comb_name": self.comb_name,
            "input_comb": self.input_comb,
            "target_keys": self.target_keys,
            "split_indexes": self.split_indexes,
            "prepared_split_indexes": self.prepared_split_indexes,
            "ref_value_splits": self.ref_value_splits,
            "standardization_info": self.standardization_info,
            "info_dict": self.info_dict,
            "exp_config": self.exp_config,
            "data_prep_config": self.data_prep_config,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "user_extras": self.user_extras,
        }


@dataclass
class ExperimentUnit:
    """Represent the data and context of one executable experiment condition.

    Parameters
    ----------
    data : dict[str, Any]
        Train, validation, and test data in the configured data level.
    context : RunContext
        Runtime information associated with the condition.
    """

    data: Dict[str, Any]
    context: ExperimentContext

    @property
    def input_comb(self) -> List[Any]:
        """Return the unit's input-key combination."""
        return self.context.input_comb

    @property
    def fold(self) -> Any:
        """Return the unit's fold identifier."""
        return self.context.fold

    @property
    def info_dict(self) -> Dict[str, Any]:
        """Return shared data-preparation information for compatibility."""
        return self.context.info_dict

    def as_dict(self) -> Dict[str, Any]:
        """Return the experiment unit as a dictionary.

        Returns
        -------
        dict
            Dictionary containing ``data`` and ``context``.
        """
        return {"data": self.data, "context": self.context.as_dict()}


class ExperimentManager:
    """Prepare shared experiment state and iterate over experiment units.

    The manager is re-iterable: every call to ``iter(manager)`` creates a new
    generator over all input-combination and fold conditions.

    Parameters
    ----------
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
    """

    def __init__(
        self,
        exp_config: ExperimentConfig,
        data_dicts: Optional[List[Dict[str, Any]]] = None,
        data_prep_config: Any = None,
        user_extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the manager without loading or preparing data yet."""
        self.exp_config = exp_config
        self.data_dicts = data_dicts
        self.data_prep_config = data_prep_config
        self.user_extras = {} if user_extras is None else dict(user_extras)
        self.collected_data: Optional[Dict[str, Any]] = None
        self.info_dict: Optional[Dict[str, Any]] = None
        self._is_setup = False

        if not self.exp_config.load_prepared_data:
            if self.data_dicts is None or self.data_prep_config is None:
                raise ValueError(
                    "`data_dicts` and `data_prep_config` are required when "
                    "`load_prepared_data=False`."
                )

        self.init_random_seed(self.exp_config.random_seed)

    def setup(self) -> "ExperimentManager":
        """Load or prepare shared data and save the experiment configuration.

        Returns
        -------
        ExperimentManager
            The initialized, iterable manager itself.
        """
        if self._is_setup:
            return self
        (
            self.collected_data,
            self.info_dict,
            self.data_prep_config,
        ) = self.get_prepared_data()
        self._validate_prepared_data()
        self.save_exp_config()
        self._is_setup = True
        return self

    def __iter__(self) -> Iterator[ExperimentUnit]:
        """Return a fresh iterator over all configured experiment units."""
        self.setup()
        assert self.collected_data is not None
        assert self.info_dict is not None
        return iter_experiment_units(
            collected_data=self.collected_data,
            info_dict=self.info_dict,
            exp_config=self.exp_config,
            data_prep_config=self.data_prep_config,
            user_extras=self.user_extras,
        )

    def init_random_seed(self, random_seed: int) -> None:
        """Seed configured random-number systems.

        Parameters
        ----------
        random_seed : int
            Seed applied to the configured systems.
        """
        if "random" in self.exp_config.random_seed_scope:
            random.seed(random_seed)
        if "numpy" in self.exp_config.random_seed_scope:
            np.random.seed(random_seed)
        if "torch" in self.exp_config.random_seed_scope and TORCH_AVAILABLE:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

    def get_prepared_data(self) -> tuple[Dict[str, Any], Dict[str, Any], Any]:
        """Load prepared data or execute the data-preparation pipeline.

        Returns
        -------
        collected_data : dict
            Prepared arrays or lists keyed by modality and target.
        info_dict : dict
            Split, target, shape, and index-mapping information.
        data_prep_config : Any
            Configuration used to prepare the data.
        """
        if self.exp_config.load_prepared_data:
            return load_prepared_data(self.exp_config.store_dir)

        preparator = data_prep.DataPreparator(
            self.data_dicts,
            self.data_prep_config,
        )
        result = preparator.run()
        if isinstance(result, tuple) and len(result) >= 2:
            collected_data, info_dict = result[:2]
        else:
            collected_data = preparator.collected_data
            info_dict = preparator.info_dict
        return collected_data, info_dict, self.data_prep_config

    def _validate_prepared_data(self) -> None:
        """Validate keys and split metadata required by the experiment plan."""
        assert self.collected_data is not None
        assert self.info_dict is not None
        required_keys = self.exp_config.input_keys + self.exp_config.target_keys
        missing = [key for key in required_keys if key not in self.collected_data]
        if missing:
            raise KeyError(f"Prepared data is missing configured keys: {missing}.")
        if "index_split_folds" not in self.info_dict:
            raise KeyError("`info_dict` does not contain `index_split_folds`.")

    def get_result(
        self,
        pred: Any,
        true: Any,
        task_type: TaskType = "c",
    ) -> Dict[str, Any]:
        """Post-process predictions and calculate task metrics.

        Parameters
        ----------
        pred, true : Any
            Prediction and target arrays or tensors.
        task_type : {"c", "r"}
            Classification or regression task type.

        Returns
        -------
        dict
            Metric dictionary.
        """
        pred_mode = "argmax" if task_type == "c" and _needs_argmax(pred) else "raw"
        pred_processed = torch_postprocess(pred, mode=pred_mode)
        true_processed = torch_postprocess(true, mode="raw")
        return self.compute_result(pred_processed, true_processed, task_type)
    
    def torch_postprocess(
        self, 
        tensor: Any,
        *,
        task_type: TaskType = "c",
    ):
        """Wrapper for post-process tensor. Convert tensors or array-like values to NumPy evaluation arrays.

        Parameters
        ----------
        tensor : Any
            PyTorch tensor, NumPy array, or array-like value.
        task_type : {"c", "r"}
            Classification or regression task type.

        Returns
        -------
        dict
            Metric dictionary.
        """
        mode = "argmax" if task_type == "c" else "raw"
        return torch_postprocess(tensor, mode=mode, use_gpu=tensor.is_cuda)
        
    @staticmethod
    def compute_result(
        pred: Any,
        true: Any,
        task_type: TaskType = "c",
    ) -> Dict[str, Any]:
        """Calculate classification or regression metrics."""
        if task_type == "c":
            return compute_cls_metrics(pred, true)
        if task_type == "r":
            return compute_regression_metrics(pred.squeeze(), true.squeeze())
        raise ValueError("`task_type` must be 'c' or 'r'.")

    @staticmethod
    def compute_average_result(
        result_dicts: Sequence[Mapping[str, Any]],
        target_key: Any,
        metric_key: str,
    ) -> Any:
        """Average a nested target metric across result dictionaries."""
        return compute_target_key_average_metric(
            result_dicts,
            target_key,
            metric_key,
        )

    def save_exp_config(self) -> None:
        """Save the experiment configuration under ``store_dir``."""
        os.makedirs(self.exp_config.store_dir, exist_ok=True)
        file_path = os.path.join(self.exp_config.store_dir, "ExpConfig.json")
        with open(file_path, "w", encoding="utf-8") as json_o:
            json.dump(self.exp_config.to_json(), json_o, indent=2, ensure_ascii=False)

    def save_result(
        self,
        result: Any,
        context: Optional[ExperimentContext] = None,
        file_name: str = "ExpResult.pkl",
    ) -> str:
        """Save a result globally or under one run's output directory.

        Parameters
        ----------
        result : Any
            Pickle-serializable result object.
        context : RunContext, optional
            When supplied, save below ``context.output_dir``. Otherwise save
            below the manager's ``store_dir``.
        file_name : str
            Output file name.

        Returns
        -------
        str
            Saved file path.
        """
        output_dir = self.exp_config.store_dir if context is None else context.output_dir
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "wb") as pkl_o:
            pickle.dump(result, pkl_o)
        return file_path


# ---------------------------------------------------------------------------
# Experiment-unit generation
# ---------------------------------------------------------------------------
def generate_key_combs(keys: Sequence[Any]) -> List[List[Any]]:
    """Generate every non-empty combination of the given keys.

    Parameters
    ----------
    keys : sequence
        Input keys in the desired combination order.

    Returns
    -------
    list[list]
        Combinations ordered first by size and then by input order.
    """
    key_combs: List[List[Any]] = []
    for comb_num in range(1, len(keys) + 1):
        key_combs.extend(
            list(combination)
            for combination in itertools.combinations(keys, comb_num)
        )
    return key_combs


def load_prepared_data(
    data_dir: str = "./DataExperiment",
) -> tuple[Dict[str, Any], Dict[str, Any], Any]:
    """Load data, preparation information, and preparation configuration.

    Parameters
    ----------
    data_dir : str
        Directory created by ``data_prep.save_data``.

    Returns
    -------
    collected_data, info_dict, data_prep_config : tuple
        Objects loaded through ``data_prep.load_data``.

    Raises
    ------
    FileNotFoundError
        If the directory or required files do not exist.
    """
    required_files = ("Data.pkl", "Info.pkl", "DataPrepConfig.json")
    missing = [
        file_name
        for file_name in required_files
        if not os.path.isfile(os.path.join(data_dir, file_name))
    ]
    if missing:
        raise FileNotFoundError(
            f"Prepared data is incomplete in {data_dir!r}; missing files: {missing}."
        )
    return data_prep.load_data(data_dir)


def iter_experiment_units(
    collected_data: Dict[str, Any],
    info_dict: Dict[str, Any],
    exp_config: ExperimentConfig,
    data_prep_config: Any,
    user_extras: Optional[Dict[str, Any]] = None,
) -> Iterator[ExperimentUnit]:
    """Yield one :class:`ExperimentUnit` per input combination and fold.

    Standardization with ``standardize_scope='split'`` uses training data from
    the current fold and applies the same statistics to train, validation, and
    test data. This avoids validation/test leakage. ``'all'`` uses all prepared
    samples and therefore should only be selected deliberately.

    Parameters
    ----------
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

    Yields
    ------
    ExperimentUnit
        Data and runtime information for one experiment condition.
    """
    folds = _normalize_fold_mapping(info_dict["index_split_folds"])
    ref_folds = _normalize_optional_fold_mapping(
        info_dict.get("ref_value_split_folds"),
        folds.keys(),
    )
    user_extras_process = {} if user_extras is None else dict(user_extras)

    for comb_index, input_comb in enumerate(exp_config.input_combs):
        comb_name = _make_comb_name(input_comb, exp_config.input_key_abbr)
        selected_keys = _unique_in_order(
            list(input_comb) + list(exp_config.target_keys) + list(data_prep.INDEX_KEY_LIST)
        )

        for fold, original_split_indexes in folds.items():
            prepared_split_indexes = _translate_split_indexes(
                original_split_indexes,
                info_dict,
                collected_data,
            )
            raw_split_data = {
                split_key: collect_combination_data(
                    collected_data,
                    selected_keys,
                    index_list,
                )
                for split_key, index_list in prepared_split_indexes.items()
            }
            standardized_split_data, standardization_info = _standardize_split_data(
                raw_split_data,
                collected_data,
                input_comb,
                data_prep_config,
            )
            unit_data = _convert_data_level(
                standardized_split_data,
                exp_config,
            )
            output_dir = os.path.join(
                exp_config.store_dir,
                comb_name,
                f"fold_{fold}",
            )
            context = ExperimentContext(
                fold=fold,
                comb_index=comb_index,
                comb_name=comb_name,
                input_comb=list(input_comb),
                target_keys=list(exp_config.target_keys),
                split_indexes={
                    key: list(value)
                    for key, value in original_split_indexes.items()
                },
                prepared_split_indexes=prepared_split_indexes,
                ref_value_splits={
                    key: list(value)
                    for key, value in ref_folds.get(fold, {}).items()
                },
                standardization_info=standardization_info,
                info_dict=info_dict,
                exp_config=exp_config,
                data_prep_config=data_prep_config,
                output_dir=output_dir,
                seed=exp_config.random_seed,
                user_extras=dict(user_extras_process),
            )
            yield ExperimentUnit(data=unit_data, context=context)


def make_data_generator(
    collected_data: Dict[str, Any],
    info_dict: Dict[str, Any],
    exp_config: ExperimentConfig,
    data_prep_config: Any,
    data_level: Optional[DataLevel] = None,
    data_representation: Optional[DataRepresentation] = None,
) -> Iterator[ExperimentUnit]:
    """Create an experiment-unit generator.

    This compatibility function delegates to :func:`iter_experiment_units`.
    Optional data-level arguments are applied to a shallow configuration copy
    so the caller's configuration is not modified.
    """
    if data_level is None and data_representation is None:
        config = exp_config
    else:
        config_fields = asdict(exp_config)
        if data_level is not None:
            config_fields["data_level"] = data_level
        if data_representation is not None:
            config_fields["data_representation"] = data_representation
        config = exp_config.__class__(**config_fields)
    return iter_experiment_units(
        collected_data,
        info_dict,
        config,
        data_prep_config,
    )


def make_data_geneartor(*args: Any, **kwargs: Any) -> Iterator[ExperimentUnit]:
    """Deprecated misspelled alias of :func:`make_data_generator`."""
    warnings.warn(
        "`make_data_geneartor` is deprecated; use `make_data_generator`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_data_generator(*args, **kwargs)


def collect_combination_data(
    collected_data: Mapping[Any, Any],
    keys: Sequence[Any],
    index_list: Sequence[int],
) -> Dict[Any, Any]:
    """Select specified prepared-data positions for a group of keys.

    Parameters
    ----------
    collected_data : mapping
        Prepared data keyed by input, target, or index key.
    keys : sequence
        Keys to include in the result.
    index_list : sequence[int]
        Positions in the prepared data arrays/lists.

    Returns
    -------
    dict
        Selected data keyed by ``keys``.
    """
    selected: Dict[Any, Any] = {}
    for key in keys:
        if key not in collected_data:
            raise KeyError(f"Key {key!r} is not present in `collected_data`.")
        source = collected_data[key]
        if isinstance(source, np.ndarray):
            selected[key] = source[list(index_list)]
        elif TORCH_AVAILABLE and isinstance(source, torch.Tensor):
            selected[key] = source[list(index_list)]
        elif isinstance(source, (list, tuple)):
            selected[key] = [source[index] for index in index_list]
        else:
            raise TypeError(
                f"Unsupported collected-data type for key {key!r}: "
                f"{type(source).__name__}."
            )
    return selected


def perform_zscore(
    data_array: Any,
    axis: Union[int, tuple[int, ...]] = 0,
) -> np.ndarray:
    """Standardize numeric data with zero-variance protection.

    Parameters
    ----------
    data_array : array-like
        Numeric data.
    axis : int or tuple[int, ...]
        Axes used to calculate the mean and standard deviation.

    Returns
    -------
    numpy.ndarray
        Standardized data.
    """
    array = _as_numeric_array(data_array, "data_array")
    mean = np.mean(array, axis=axis, keepdims=True)
    std = np.std(array, axis=axis, keepdims=True)
    return _apply_standardization(array, mean, std)


def get_input_target_shapes(
    datasets: Mapping[str, Any],
    info_dict: Mapping[str, Any],
) -> Dict[Any, tuple[int, ...]]:
    """Collect configured input and target shapes available in a dataset.

    Parameters
    ----------
    datasets : mapping
        Split datasets or raw split dictionaries containing ``"train"``.
    info_dict : mapping
        Data-preparation information containing ``input_shapes`` and
        ``target_info``.

    Returns
    -------
    dict
        Shape information keyed by input or target name.
    """
    train_data = datasets["train"]
    if TORCH_AVAILABLE and DataLoader is not None and isinstance(train_data, DataLoader):
        train_data = train_data.dataset
    if isinstance(train_data, TorchDataset):
        keys = list(train_data.data_dict)
    elif isinstance(train_data, Mapping):
        keys = list(train_data)
    else:
        raise TypeError(
            "The training split must be a raw mapping, TorchDataset, or DataLoader."
        )
    input_shapes = info_dict.get("input_shapes", info_dict.get("input_shape", {}))
    target_info = info_dict.get("target_info", {})
    shapes: Dict[Any, tuple[int, ...]] = {}
    for key in keys:
        if key in input_shapes:
            shapes[key] = tuple(input_shapes[key])
        elif key in target_info:
            target_size = target_info[key].get("target_size", 1)
            if isinstance(target_size, int):
                shapes[key] = (target_size,)
            else:
                shapes[key] = tuple(target_size)
    return shapes


# ---------------------------------------------------------------------------
# PyTorch dataset and tensor helpers
# ---------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class TorchDataset(torch.utils.data.Dataset):
        """Dictionary-backed PyTorch dataset with aligned values per key.

        Parameters
        ----------
        data_dict : dict
            Arrays, tensors, or lists with the same first-dimension length.
        """

        def __init__(self, data_dict: Dict[Any, Any]) -> None:
            """Validate and store aligned data containers."""
            if not data_dict:
                raise ValueError("`data_dict` must not be empty.")
            lengths = {key: len(value) for key, value in data_dict.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Dataset values have inconsistent lengths: {lengths}.")
            self.data_dict = data_dict
            self.data_len = next(iter(lengths.values()))

        def __getitem__(self, index: int) -> Dict[Any, Any]:
            """Return all keyed values at one dataset position."""
            return {key: value[index] for key, value in self.data_dict.items()}

        def __len__(self) -> int:
            """Return the number of samples."""
            return self.data_len

else:

    class TorchDataset:
        """Placeholder that reports the missing optional PyTorch dependency."""

        def __init__(self, data_dict: Dict[Any, Any]) -> None:
            """Raise an informative dependency error."""
            raise ImportError("PyTorch is required to create `TorchDataset`.")


def make_dataset(
    single_condition_data: Dict[Any, Any],
    data_representation: DataRepresentation = "original",
    use_gpu: bool = False,
) -> TorchDataset:
    """Create a dictionary-backed PyTorch dataset.

    Parameters
    ----------
    single_condition_data : dict
        Data for one train, validation, or test split.
    data_representation : {"original", "pt"}
        Keep values unchanged or convert supported values to tensors.
    use_gpu : bool
        Place converted tensors on CUDA. CPU tensors are recommended when the
        dataset will be wrapped by a DataLoader.

    Returns
    -------
    TorchDataset
        Created dataset.
    """
    _require_torch("make_dataset")
    if data_representation == "original":
        return TorchDataset(single_condition_data)
    if data_representation == "pt":
        return _make_dataset_tensor(single_condition_data, use_gpu=use_gpu)
    raise ValueError("`data_representation` must be 'original' or 'pt'.")


def convert_single_data_to_tensor(
    single_condition_data: Mapping[Any, Any],
    *,
    exclude_keys: Optional[Iterable[Any]] = None,
    use_gpu: bool = False,
) -> Dict[Any, Any]:
    """Convert supported split data to PyTorch tensors.

    Index metadata is converted to ``torch.long`` when possible. Unsupported
    Python objects are kept unchanged.

    Parameters
    ----------
    single_condition_data : mapping
        Split data keyed by modality, target, or metadata key.
    exclude_keys : iterable, optional
        Keys copied without conversion.
    use_gpu : bool
        Move converted tensors to CUDA when available.

    Returns
    -------
    dict
        Tensor-converted data.
    """
    _require_torch("convert_single_data_to_tensor")
    excluded = set() if exclude_keys is None else set(exclude_keys)
    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    tensor_data: Dict[Any, Any] = {}

    for key, value in single_condition_data.items():
        if key in excluded:
            tensor_data[key] = value
            continue

        if isinstance(value, torch.Tensor):
            tensor_data[key] = value.to(device)
        elif isinstance(value, np.ndarray) and value.dtype != object:
            tensor = torch.as_tensor(value)
            if np.issubdtype(value.dtype, np.integer):
                tensor = tensor.long()
            elif np.issubdtype(value.dtype, np.floating):
                tensor = tensor.float()
            tensor_data[key] = tensor.to(device)
        elif isinstance(value, list) and value and all(
            isinstance(item, np.ndarray) and item.dtype != object for item in value
        ):
            converted = []
            for item in value:
                tensor = torch.as_tensor(item)
                if np.issubdtype(item.dtype, np.integer):
                    tensor = tensor.long()
                elif np.issubdtype(item.dtype, np.floating):
                    tensor = tensor.float()
                converted.append(tensor.to(device))
            tensor_data[key] = converted
        elif key in data_prep.INDEX_KEY_LIST:
            try:
                tensor_data[key] = torch.as_tensor(value, dtype=torch.long, device=device)
            except (TypeError, ValueError):
                tensor_data[key] = value
        else:
            tensor_data[key] = value
    return tensor_data


def _make_dataset_tensor(
    single_condition_data: Dict[Any, Any],
    use_gpu: bool = False,
) -> TorchDataset:
    """Convert split data to tensors and wrap it in :class:`TorchDataset`."""
    return TorchDataset(
        convert_single_data_to_tensor(single_condition_data, use_gpu=use_gpu)
    )


def make_dataloader(
    dataset: TorchDataset,
    batch_size: int,
    shuffle: bool,
) -> Any:
    """Create a PyTorch DataLoader from a non-empty dataset.

    Parameters
    ----------
    dataset : TorchDataset
        Dataset to batch.
    batch_size : int
        Positive number of samples per batch.
    shuffle : bool
        Whether to shuffle sample order.

    Returns
    -------
    torch.utils.data.DataLoader
        Created DataLoader.
    """
    _require_torch("make_dataloader")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    if len(dataset) == 0:
        raise ValueError("Cannot create a DataLoader from an empty dataset.")
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def torch_postprocess(
    tensor: Any,
    *,
    mode: Literal["raw", "argmax"] = "raw",
    use_gpu: Optional[bool] = None,
) -> np.ndarray:
    """Convert tensors or array-like values to NumPy evaluation arrays.

    Parameters
    ----------
    tensor : Any
        PyTorch tensor, NumPy array, or array-like value.
    mode : {"raw", "argmax"}
        Return raw values or take the final-axis argmax first.
    use_gpu : bool, optional
        Deprecated compatibility argument. Device handling is inferred from the
        tensor itself.

    Returns
    -------
    numpy.ndarray
        Detached CPU array.
    """
    del use_gpu
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        value = tensor.detach()
        if mode == "argmax":
            value = torch.argmax(value, dim=-1)
        elif mode != "raw":
            raise ValueError("`mode` must be 'raw' or 'argmax'.")
        return value.cpu().numpy()

    value = np.asarray(tensor)
    if mode == "argmax":
        return np.argmax(value, axis=-1)
    if mode != "raw":
        raise ValueError("`mode` must be 'raw' or 'argmax'.")
    return value


# ---------------------------------------------------------------------------
# Saved configuration and result loading
# ---------------------------------------------------------------------------
class ExperimentResultLoader:
    """Load saved experiment configuration, preparation config, and results.

    Parameters
    ----------
    result_dir : str
        Directory containing the saved files.
    exp_config_file, data_prep_config_file, result_file : str
        File names relative to ``result_dir``.
    """

    def __init__(
        self,
        result_dir: str,
        exp_config_file: str = "ExpConfig.json",
        data_prep_config_file: str = "DataPrepConfig.json",
        result_file: str = "ExpResult.pkl",
    ) -> None:
        """Load all requested experiment artifacts."""
        self.result_dir = result_dir
        self.exp_config = load_exp_config(
            os.path.join(result_dir, exp_config_file)
        )
        self.data_prep_config = data_prep.load_config(
            os.path.join(result_dir, data_prep_config_file)
        )
        with open(os.path.join(result_dir, result_file), "rb") as pkl_i:
            self.result = pickle.load(pkl_i)


def load_exp_config(file: str = "./ExpConfig.json") -> ExperimentConfig:
    """Load a saved experiment configuration JSON file."""
    with open(file, "r", encoding="utf-8") as json_i:
        return make_config_from_json(json.load(json_i))


def make_config_from_json(config_json: Sequence[Any]) -> ExperimentConfig:
    """Construct an experiment configuration from saved JSON content.

    Parameters
    ----------
    config_json : sequence
        Two-element sequence containing class name and constructor fields.

    Returns
    -------
    ExperimentConfig
        Reconstructed configuration object.
    """
    class_map = {
        "ExperimentConfig": ExperimentConfig,
        "TorchExperimentConfig": TorchExperimentConfig,
    }
    if len(config_json) != 2 or config_json[0] not in class_map:
        raise ValueError("Invalid or unsupported experiment configuration JSON.")
    return class_map[config_json[0]](**config_json[1])


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def compute_cls_metrics(pred_list: Any, true_list: Any) -> Dict[str, Any]:
    """Calculate common single-label classification metrics.

    Pearson correlation is returned as ``nan`` when it is undefined, such as
    for constant or fewer-than-two-element arrays.
    """
    pred = np.asarray(pred_list).reshape(-1)
    true = np.asarray(true_list).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {pred.shape} vs {true.shape}."
        )
    return {
        "acc": accuracy_score(true, pred),
        "f1_macro": f1_score(true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(true, pred, average="weighted", zero_division=0),
        "precision": precision_score(true, pred, average="macro", zero_division=0),
        "recall": recall_score(true, pred, average="macro", zero_division=0),
        "pearson_correlation": _safe_pearson(true, pred),
        "true_list": true,
        "pred_list": pred,
        "confusion_matrix": confusion_matrix(true, pred),
    }


def compute_regression_metrics(pred_list: Any, true_list: Any) -> Dict[str, Any]:
    """Calculate MAE, MSE, RMSE, and Pearson correlation for regression."""
    pred = np.asarray(pred_list)
    true = np.asarray(true_list)
    if pred.shape != true.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {pred.shape} vs {true.shape}."
        )
    mse = mean_squared_error(true, pred)
    return {
        "mae": mean_absolute_error(true, pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "pearson_correlation": _safe_pearson(true.reshape(-1), pred.reshape(-1)),
        "true_list": true,
        "pred_list": pred,
    }


def compute_target_key_average_metric(
    result_dicts: Sequence[Mapping[Any, Any]],
    target_key: Any,
    metric_key: str,
) -> Any:
    """Average a metric nested below one target key."""
    if not result_dicts:
        raise ValueError("`result_dicts` must not be empty.")
    values = [result[target_key][metric_key] for result in result_dicts]
    return np.mean(values, axis=0)


def compute_average_metric(
    result_dicts: Sequence[Mapping[str, Any]],
    metric_key: str,
) -> Any:
    """Average a top-level metric across result dictionaries."""
    if not result_dicts:
        raise ValueError("`result_dicts` must not be empty.")
    values = [result[metric_key] for result in result_dicts]
    return np.mean(values, axis=0)


# ---------------------------------------------------------------------------
# Fold-summary loading and formatting
# ---------------------------------------------------------------------------
def _get_fold_average_summary(
    result_dir: Optional[str] = None,
    result_root: Optional[str] = None,
    experiment_name: Optional[str] = None,
    combs: Optional[Union[str, List[str]]] = None,
    folds: Optional[int] = None,
) -> Dict[str, Any]:
    """Load fold summaries and calculate per-combination averages.

    Either ``result_dir`` or both ``result_root`` and ``experiment_name`` must
    be supplied.
    """
    if result_dir is not None:
        process_dir = result_dir
    elif result_root is not None and experiment_name is not None:
        process_dir = os.path.join(result_root, experiment_name)
    else:
        raise ValueError(
            "Provide `result_dir`, or provide both `result_root` and "
            "`experiment_name`."
        )

    if combs is None:
        comb_dirs = sorted(
            path
            for path in glob.glob(os.path.join(process_dir, "Comb_*"))
            if os.path.isdir(path)
        )
    elif isinstance(combs, str):
        comb_dirs = [os.path.join(process_dir, combs)]
    elif isinstance(combs, list) and all(isinstance(item, str) for item in combs):
        comb_dirs = [os.path.join(process_dir, item) for item in combs]
    else:
        raise TypeError("`combs` must be a string, list of strings, or None.")

    if not comb_dirs:
        raise FileNotFoundError(f"No combination directories were found in {process_dir!r}.")

    average_summary: Dict[str, Any] = {}
    for comb_dir in comb_dirs:
        if folds is None:
            fold_dirs = sorted(
                path
                for path in glob.glob(os.path.join(comb_dir, "fold_*"))
                if os.path.isdir(path)
            )
        else:
            if folds <= 0:
                raise ValueError("`folds` must be positive.")
            fold_dirs = [os.path.join(comb_dir, f"fold_{index}") for index in range(folds)]

        if not fold_dirs:
            raise FileNotFoundError(f"No fold directories were found in {comb_dir!r}.")

        metric_sums: Dict[Any, Dict[str, float]] = {}
        fold_counts: Dict[Any, int] = {}
        comb_value: Any = None

        for fold_dir in fold_dirs:
            json_file = os.path.join(fold_dir, "Final_summary.json")
            if not os.path.isfile(json_file):
                raise FileNotFoundError(json_file)
            with open(json_file, "r", encoding="utf-8") as json_i:
                result_summary = json.load(json_i)
            comb_value = result_summary[0]["comb"]
            for summary in result_summary[1]["Test"]:
                target_key = summary["key"]
                metric_sums.setdefault(target_key, {})
                fold_counts[target_key] = fold_counts.get(target_key, 0) + 1
                for metric, value in summary["metrics"].items():
                    metric_sums[target_key][metric] = (
                        metric_sums[target_key].get(metric, 0.0) + value
                    )

        averages = {
            target_key: {
                metric: value / fold_counts[target_key]
                for metric, value in metrics.items()
            }
            for target_key, metrics in metric_sums.items()
        }
        average_summary[os.path.basename(comb_dir)] = {
            "comb": comb_value,
            "aver_summary": averages,
        }
    return average_summary


def change_average_summary_form(
    average_summary: Mapping[str, Any],
    form: Literal["lists"] = "lists",
) -> Dict[str, Any]:
    """Convert average summaries to a selected analysis-friendly format."""
    if form == "lists":
        return change_average_summary_lists(average_summary)
    raise ValueError("Currently only `form='lists'` is supported.")


def change_average_summary_lists(
    average_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Convert combination summaries to parallel combination/metric lists."""
    comb_list: List[Any] = []
    metric_lists: Dict[Any, Dict[str, List[Any]]] = {}
    for comb_summary in average_summary.values():
        comb_list.append(comb_summary["comb"])
        for target_key, metrics in comb_summary["aver_summary"].items():
            metric_lists.setdefault(target_key, {})
            for metric, value in metrics.items():
                metric_lists[target_key].setdefault(metric, []).append(value)
    return {"comb": comb_list, "metric_lists": metric_lists}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _unify_to_list(value: Any) -> List[Any]:
    """Convert scalar or iterable configuration input to a plain list."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _unique_in_order(values: Iterable[Any]) -> List[Any]:
    """Return unique values while preserving first-occurrence order."""
    unique: List[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _normalize_fold_mapping(folds: Any) -> Dict[Any, Dict[str, List[int]]]:
    """Normalize list- or mapping-form split folds to an ordered dictionary."""
    if isinstance(folds, Mapping):
        items = folds.items()
    elif isinstance(folds, list):
        items = enumerate(folds)
    else:
        raise TypeError("`index_split_folds` must be a mapping or list.")

    normalized: Dict[Any, Dict[str, List[int]]] = {}
    for fold, split_dict in items:
        if not isinstance(split_dict, Mapping):
            raise TypeError(f"Fold {fold!r} must contain a split mapping.")
        required = {"train", "valid", "test"}
        missing = required - set(split_dict)
        if missing:
            raise KeyError(f"Fold {fold!r} is missing split keys: {sorted(missing)}.")
        normalized[fold] = {
            split_key: list(indexes)
            for split_key, indexes in split_dict.items()
        }
    return normalized


def _normalize_optional_fold_mapping(
    folds: Any,
    fold_keys: Iterable[Any],
) -> Dict[Any, Dict[str, List[Any]]]:
    """Normalize optional reference-value split information."""
    if folds in (None, {} , []):
        return {fold: {} for fold in fold_keys}
    if isinstance(folds, Mapping):
        source = folds
    elif isinstance(folds, list):
        source = dict(enumerate(folds))
    else:
        raise TypeError("`ref_value_split_folds` must be a mapping, list, or None.")
    return {
        fold: {
            split_key: list(values)
            for split_key, values in source.get(fold, {}).items()
        }
        for fold in fold_keys
    }


def _translate_split_indexes(
    split_indexes: Mapping[str, Sequence[int]],
    info_dict: Mapping[str, Any],
    collected_data: Mapping[Any, Any],
) -> Dict[str, List[int]]:
    """Translate original sample indexes to prepared-data positions."""
    mapping = info_dict.get("ori_index2id")
    sample_count = _infer_sample_count(collected_data)
    translated: Dict[str, List[int]] = {}

    for split_key, indexes in split_indexes.items():
        if mapping is not None:
            missing = [index for index in indexes if index not in mapping]
            if missing:
                raise KeyError(
                    f"Split {split_key!r} contains original indexes absent from "
                    f"`ori_index2id`: {missing[:10]}."
                )
            positions = [mapping[index] for index in indexes]
        else:
            positions = list(indexes)
        invalid = [index for index in positions if index < 0 or index >= sample_count]
        if invalid:
            raise IndexError(
                f"Split {split_key!r} contains prepared positions outside "
                f"[0, {sample_count}): {invalid[:10]}."
            )
        translated[split_key] = positions
    return translated


def _infer_sample_count(collected_data: Mapping[Any, Any]) -> int:
    """Infer and validate the shared first-dimension sample count."""
    lengths = {
        key: len(value)
        for key, value in collected_data.items()
        if hasattr(value, "__len__")
    }
    if not lengths:
        raise ValueError("No sized values were found in `collected_data`.")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Collected-data values have inconsistent lengths: {lengths}.")
    return next(iter(lengths.values()))


def _standardize_split_data(
    split_data: Dict[str, Dict[Any, Any]],
    collected_data: Mapping[Any, Any],
    input_comb: Sequence[Any],
    data_prep_config: Any,
) -> tuple[Dict[str, Dict[Any, Any]], Dict[Any, Dict[str, Any]]]:
    """Standardize configured numeric inputs without mutating shared data."""
    processed = {
        split_key: dict(values)
        for split_key, values in split_data.items()
    }
    standardization_info: Dict[Any, Dict[str, Any]] = {}

    for key in input_comb:
        if not hasattr(data_prep_config, "key2config") or key not in data_prep_config.key2config:
            continue
        config = data_prep_config.key2config[key]
        if not getattr(config, "standardize_data", False):
            continue

        scope = getattr(config, "standardize_scope", "split")
        if scope == "all":
            reference = _as_numeric_array(collected_data[key], key)
            source = "all"
        elif scope == "split":
            if "train" not in processed or len(processed["train"][key]) == 0:
                raise ValueError(
                    f"Cannot calculate train-fold standardization for key {key!r} "
                    "because the training split is empty."
                )
            reference = _as_numeric_array(processed["train"][key], key)
            source = "train"
        else:
            raise ValueError(
                f"Unsupported standardization scope {scope!r} for key {key!r}."
            )

        mean = np.mean(reference, axis=0, keepdims=True)
        std = np.std(reference, axis=0, keepdims=True)
        for split_key in processed:
            array = _as_numeric_array(processed[split_key][key], key)
            processed[split_key][key] = _apply_standardization(array, mean, std)
        standardization_info[key] = {
            "mean": mean,
            "std": std,
            "scope": scope,
            "source": source,
        }
    return processed, standardization_info


def _apply_standardization(
    array: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply z-score normalization while mapping zero-variance axes to zero."""
    safe_std = np.where(std == 0, 1.0, std)
    return (array - mean) / safe_std


def _as_numeric_array(value: Any, key: Any) -> np.ndarray:
    """Convert a value to a non-object numeric NumPy array."""
    array = np.asarray(value)
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Standardization requires numeric data for key {key!r}.")
    return array


def _convert_data_level(
    split_data: Dict[str, Dict[Any, Any]],
    exp_config: ExperimentConfig,
) -> Dict[str, Any]:
    """Convert raw split dictionaries to the requested experiment data level."""
    if exp_config.data_level == "raw":
        return split_data

    converted: Dict[str, Any] = {}
    for split_key, values in split_data.items():
        # Keep DataLoader datasets on CPU; batches should be moved in training.
        dataset = make_dataset(
            values,
            data_representation=exp_config.data_representation,
            use_gpu=False,
        )
        if exp_config.data_level == "dataset":
            converted[split_key] = dataset
            continue

        batch_attr = f"{split_key}_batch_size"
        shuffle_attr = f"shuffle_{split_key}_data"
        if not hasattr(exp_config, batch_attr) or not hasattr(exp_config, shuffle_attr):
            raise TypeError(
                "`data_level='dataloader'` requires batch-size and shuffle "
                "settings, normally provided by `TorchExperimentConfig`."
            )
        converted[split_key] = make_dataloader(
            dataset,
            batch_size=getattr(exp_config, batch_attr),
            shuffle=getattr(exp_config, shuffle_attr),
        )
    return converted


def _make_comb_name(
    input_comb: Sequence[Any],
    abbreviations: Mapping[Any, Any],
) -> str:
    """Create a file-system-safe directory name for an input combination."""
    tags = [str(abbreviations.get(key, key)) for key in input_comb]
    sanitized = [_sanitize_path_component(tag) for tag in tags]
    return "Comb_" + "-".join(sanitized)


def _sanitize_path_component(value: str) -> str:
    """Replace path-unsafe characters with underscores."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "unnamed"


def _require_torch(caller: str) -> None:
    """Raise an informative error when a PyTorch helper is unavailable."""
    if not TORCH_AVAILABLE:
        raise ImportError(f"PyTorch is required for `{caller}`.")


def _needs_argmax(prediction: Any) -> bool:
    """Return whether a classification prediction has a class-score axis."""
    shape = tuple(prediction.shape) if hasattr(prediction, "shape") else np.asarray(prediction).shape
    return len(shape) >= 2 and shape[-1] > 1


def _safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate Pearson correlation and return ``nan`` when undefined."""
    first = np.asarray(first).reshape(-1)
    second = np.asarray(second).reshape(-1)
    if len(first) < 2 or len(second) < 2:
        return float("nan")
    if np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(pearsonr(first, second)[0])
