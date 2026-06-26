# -*- coding: utf-8 -*-
"""Utilities for querying and organizing sample-level information records.

The functions in this module expect ``data_dicts`` to be a list of dictionaries.
Each dictionary describes one sample, utterance, frame, or other sequential item.
Dictionary fields may contain identifiers, grouping variables, target values, or
other lightweight sample information.
"""

from typing import Any, Dict, List, Literal, Union


def _resolve_used_indexes(
    data_dicts: List[Dict],
    used_indexes: Union[List[int], None],
) -> List[int]:
    """Resolve and validate the sample indexes to be processed.

    Parameters
    ----------
    data_dicts : list of dict
        Sample-level information records.
    used_indexes : list of int or None
        Indexes to process. If ``None``, all indexes are returned in their
        original order.

    Returns
    -------
    list of int
        Validated indexes to process.

    Raises
    ------
    TypeError
        If an index is not an integer.
    IndexError
        If an index is outside the valid range of ``data_dicts``.
    """
    if used_indexes is None:
        return list(range(len(data_dicts)))

    indexes_for_process = list(used_indexes)
    for index in indexes_for_process:
        if not isinstance(index, int):
            raise TypeError("All values in used_indexes must be integers.")
        if index < 0 or index >= len(data_dicts):
            raise IndexError(
                f"Index {index} is outside the valid range "
                f"[0, {len(data_dicts) - 1}]."
            )
    return indexes_for_process


def _values_equal(left: Any, right: Any) -> bool:
    """Compare two values while supporting scalar and array-like objects.

    Parameters
    ----------
    left : Any
        First value to compare.
    right : Any
        Second value to compare.

    Returns
    -------
    bool
        ``True`` when the values are equivalent. Array-like comparison results
        are reduced across all elements.
    """
    try:
        comparison = left == right
    except Exception:
        return False

    if isinstance(comparison, bool):
        return comparison

    try:
        return bool(comparison.all())
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        return all(comparison)
    except (TypeError, ValueError):
        pass

    try:
        return bool(comparison)
    except (TypeError, ValueError):
        return False


def _contains_equivalent(values: List[Any], candidate: Any) -> bool:
    """Check whether a list contains a value equivalent to a candidate.

    Parameters
    ----------
    values : list
        Existing values to search.
    candidate : Any
        Candidate value, which may be a scalar or an array-like object.

    Returns
    -------
    bool
        ``True`` if an equivalent value is already present.
    """
    return any(_values_equal(value, candidate) for value in values)


def get_ref_value2indexes(
    data_dicts: List[Dict],
    ref_key: Any = "speaker",
    used_indexes: Union[List[int], None] = None,
) -> Dict[Any, List[int]]:
    """Map each reference value to its corresponding sample indexes.

    Parameters
    ----------
    data_dicts : list of dict
        Sample-level information records. Every processed dictionary must
        contain ``ref_key``.
    ref_key : Any, optional
        Dictionary key whose values define the reference groups, such as a
        speaker, participant, session, or group identifier.
    used_indexes : list of int or None, optional
        Subset of sample indexes to process. If ``None``, all samples are used.

    Returns
    -------
    dict
        Mapping from each value of ``ref_key`` to the indexes of samples that
        contain that value. Index order follows ``used_indexes``.

    Raises
    ------
    KeyError
        If a processed dictionary does not contain ``ref_key``.
    TypeError
        If a reference value is not hashable and therefore cannot be used as a
        dictionary key.
    """
    ref_value2indexes: Dict[Any, List[int]] = {}
    indexes_for_process = _resolve_used_indexes(data_dicts, used_indexes)

    for index in indexes_for_process:
        ref_value = data_dicts[index][ref_key]
        ref_value2indexes.setdefault(ref_value, []).append(index)

    return ref_value2indexes


def get_ref_value_list(
    data_dicts: List[Dict],
    ref_key: Any = "speaker",
    used_indexes: Union[List[int], None] = None,
) -> List[Any]:
    """Collect one reference value for each processed sample.

    Parameters
    ----------
    data_dicts : list of dict
        Sample-level information records. Every processed dictionary must
        contain ``ref_key``.
    ref_key : Any, optional
        Dictionary key whose value is collected from each sample.
    used_indexes : list of int or None, optional
        Subset of sample indexes to process. If ``None``, all samples are used.

    Returns
    -------
    list
        Values of ``ref_key`` in sample order. Repeated values are preserved.

    Raises
    ------
    KeyError
        If a processed dictionary does not contain ``ref_key``.
    """
    indexes_for_process = _resolve_used_indexes(data_dicts, used_indexes)
    return [data_dicts[index][ref_key] for index in indexes_for_process]


def get_ref_value2another(
    data_dicts: List[Dict],
    ref_key: Any = "speaker",
    another_key: Any = "target",
    used_indexes: Union[List[int], None] = None,
    *,
    unique_values: bool = True,
) -> Dict[Any, List[Any]]:
    """Map each reference value to values from another sample field.

    This function is useful for checking relationships between two information
    fields, such as mapping each speaker to their target values or each group to
    its participant identifiers.

    Parameters
    ----------
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

    Returns
    -------
    dict
        Mapping from each value of ``ref_key`` to values of ``another_key``.

    Raises
    ------
    KeyError
        If a processed dictionary lacks ``ref_key`` or ``another_key``.
    """
    ref_value2another: Dict[Any, List[Any]] = {}
    indexes_for_process = _resolve_used_indexes(data_dicts, used_indexes)

    for index in indexes_for_process:
        dict_item = data_dicts[index]
        ref_value = dict_item[ref_key]
        another_value = dict_item[another_key]
        values = ref_value2another.setdefault(ref_value, [])

        if not unique_values or not _contains_equivalent(values, another_value):
            values.append(another_value)

    return ref_value2another


def get_turn2ref_value_and_indexes(
    data_dicts: List[Dict],
    ref_key: Any = "speaker",
    used_indexes: Union[List[int], None] = None,
) -> Dict[int, Dict[str, Any]]:
    """Group consecutive samples with the same reference value into turns.

    A new turn starts when either the value of ``ref_key`` changes or the next
    processed sample index is not numerically consecutive to the previous one.
    The order supplied by ``used_indexes`` is preserved.

    Parameters
    ----------
    data_dicts : list of dict
        Ordered sample-level records representing one dialogue, conversation,
        event, or other sequence.
    ref_key : Any, optional
        Dictionary key used to determine whether consecutive samples belong to
        the same turn. This is typically a speaker or participant identifier.
    used_indexes : list of int or None, optional
        Ordered subset of sample indexes to process. If ``None``, all samples
        are used.

    Returns
    -------
    dict
        Mapping from zero-based turn IDs to dictionaries with two fields:
        ``"ref_value"`` contains the turn's reference value, and ``"indexes"``
        contains the sample indexes in that turn.

    Raises
    ------
    KeyError
        If a processed dictionary does not contain ``ref_key``.
    """
    indexes_for_process = _resolve_used_indexes(data_dicts, used_indexes)
    if not indexes_for_process:
        return {}

    turn2ref_value_and_indexes: Dict[int, Dict[str, Any]] = {}
    current_turn = 0
    first_index = indexes_for_process[0]
    current_ref_value = data_dicts[first_index][ref_key]
    turn_indexes = [first_index]
    previous_index = first_index

    for index in indexes_for_process[1:]:
        ref_value = data_dicts[index][ref_key]
        is_continuous = index == previous_index + 1
        is_same_ref_value = ref_value == current_ref_value

        if is_continuous and is_same_ref_value:
            turn_indexes.append(index)
        else:
            turn2ref_value_and_indexes[current_turn] = {
                "ref_value": current_ref_value,
                "indexes": turn_indexes,
            }
            current_turn += 1
            current_ref_value = ref_value
            turn_indexes = [index]

        previous_index = index

    turn2ref_value_and_indexes[current_turn] = {
        "ref_value": current_ref_value,
        "indexes": turn_indexes,
    }
    return turn2ref_value_and_indexes


def get_ref_value2turn_indexes(
    data_dicts: List[Dict],
    ref_key: Any = "speaker",
    used_indexes: Union[List[int], None] = None,
) -> Dict[Any, List[List[int]]]:
    """Map each reference value to the sample-index lists of its turns.

    Parameters
    ----------
    data_dicts : list of dict
        Ordered sample-level records representing a sequence.
    ref_key : Any, optional
        Dictionary key used to identify the owner of each turn.
    used_indexes : list of int or None, optional
        Ordered subset of sample indexes to process. If ``None``, all samples
        are used.

    Returns
    -------
    dict
        Mapping from each reference value to a list of turns, where each turn is
        represented by its sample-index list.
    """
    turn2ref_value_and_indexes = get_turn2ref_value_and_indexes(
        data_dicts,
        ref_key=ref_key,
        used_indexes=used_indexes,
    )
    ref_value2turn_indexes: Dict[Any, List[List[int]]] = {}

    for turn_info in turn2ref_value_and_indexes.values():
        ref_value = turn_info["ref_value"]
        ref_value2turn_indexes.setdefault(ref_value, []).append(
            turn_info["indexes"]
        )

    return ref_value2turn_indexes


def get_ref_value2turns(
    data_dicts: List[Dict],
    ref_key: Any = "speaker",
    used_indexes: Union[List[int], None] = None,
) -> Dict[Any, List[int]]:
    """Map each reference value to its zero-based turn IDs.

    Parameters
    ----------
    data_dicts : list of dict
        Ordered sample-level records representing a sequence.
    ref_key : Any, optional
        Dictionary key used to identify the owner of each turn.
    used_indexes : list of int or None, optional
        Ordered subset of sample indexes to process. If ``None``, all samples
        are used.

    Returns
    -------
    dict
        Mapping from each reference value to the IDs of its turns.
    """
    turn2ref_value_and_indexes = get_turn2ref_value_and_indexes(
        data_dicts,
        ref_key=ref_key,
        used_indexes=used_indexes,
    )
    ref_value2turns: Dict[Any, List[int]] = {}

    for turn, turn_info in turn2ref_value_and_indexes.items():
        ref_value = turn_info["ref_value"]
        ref_value2turns.setdefault(ref_value, []).append(turn)

    return ref_value2turns


def get_ref_value2indexes_in_turns(
    data_dicts: List[Dict],
    ref_key: Any = "speaker",
    used_indexes: Union[List[int], None] = None,
) -> Dict[Any, List[int]]:
    """Map each reference value to indexes collected through its turns.

    This is the flattened counterpart of :func:`get_ref_value2turn_indexes`.
    Turn boundaries are not retained in the returned value.

    Parameters
    ----------
    data_dicts : list of dict
        Ordered sample-level records representing a sequence.
    ref_key : Any, optional
        Dictionary key used to identify the owner of each turn.
    used_indexes : list of int or None, optional
        Ordered subset of sample indexes to process. If ``None``, all samples
        are used.

    Returns
    -------
    dict
        Mapping from each reference value to all of its sample indexes, ordered
        by turn occurrence.
    """
    ref_value2turn_indexes = get_ref_value2turn_indexes(
        data_dicts,
        ref_key=ref_key,
        used_indexes=used_indexes,
    )
    ref_value2indexes_in_turns: Dict[Any, List[int]] = {}

    for ref_value, turn_index_lists in ref_value2turn_indexes.items():
        ref_value2indexes_in_turns[ref_value] = [
            index
            for turn_indexes in turn_index_lists
            for index in turn_indexes
        ]

    return ref_value2indexes_in_turns


def get_ref_value2adjacent_ref_value(
    data_dicts: List[Dict],
    ref_key: Any = "speaker",
    prev_or_following: Literal["prev", "following"] = "prev",
    adjacent_by: Literal["index", "turn"] = "index",
    used_indexes: Union[List[int], None] = None,
) -> Dict[Any, Dict[Any, List[int]]]:
    """Summarize cross-reference adjacency by sample index or turn.

    The top-level key is the reference value of the item of interest. Each
    nested key is the reference value of its adjacent item. Returned integers
    identify the current sample when ``adjacent_by="index"`` and the current
    turn when ``adjacent_by="turn"``.

    Only adjacent items with different reference values are included. When a
    subset containing gaps is supplied, adjacency is not created across those
    non-consecutive gaps.

    Parameters
    ----------
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

    Returns
    -------
    dict
        Nested mapping of
        ``current_ref_value -> adjacent_ref_value -> current_indexes_or_turns``.

    Raises
    ------
    ValueError
        If ``prev_or_following`` or ``adjacent_by`` is invalid.
    """
    if prev_or_following not in {"prev", "following"}:
        raise ValueError("prev_or_following must be 'prev' or 'following'.")
    if adjacent_by not in {"index", "turn"}:
        raise ValueError("adjacent_by must be 'index' or 'turn'.")

    index_offset = -1 if prev_or_following == "prev" else 1

    if adjacent_by == "index":
        indexes_for_process = _resolve_used_indexes(data_dicts, used_indexes)
        index2ref_value = {
            index: data_dicts[index][ref_key]
            for index in indexes_for_process
        }
        result: Dict[Any, Dict[Any, List[int]]] = {
            ref_value: {} for ref_value in index2ref_value.values()
        }

        for index, ref_value in index2ref_value.items():
            adjacent_index = index + index_offset
            if adjacent_index not in index2ref_value:
                continue

            adjacent_ref_value = index2ref_value[adjacent_index]
            if adjacent_ref_value == ref_value:
                continue

            result[ref_value].setdefault(adjacent_ref_value, []).append(index)

        return result

    turn2info = get_turn2ref_value_and_indexes(
        data_dicts,
        ref_key=ref_key,
        used_indexes=used_indexes,
    )
    result = {
        turn_info["ref_value"]: {}
        for turn_info in turn2info.values()
    }

    for turn, turn_info in turn2info.items():
        adjacent_turn = turn + index_offset
        if adjacent_turn not in turn2info:
            continue

        adjacent_info = turn2info[adjacent_turn]
        if prev_or_following == "prev":
            is_continuous = (
                adjacent_info["indexes"][-1] + 1
                == turn_info["indexes"][0]
            )
        else:
            is_continuous = (
                turn_info["indexes"][-1] + 1
                == adjacent_info["indexes"][0]
            )
        if not is_continuous:
            continue

        ref_value = turn_info["ref_value"]
        adjacent_ref_value = adjacent_info["ref_value"]
        if adjacent_ref_value == ref_value:
            continue

        result[ref_value].setdefault(adjacent_ref_value, []).append(turn)

    return result


def get_interval_split_indexes(
    data_dicts: List[Dict],
    ref_key: Any = "speaker",
    interval_num: int = 3,
    interval_split_by: Literal["index", "turn"] = "index",
    used_indexes: Union[List[int], None] = None,
) -> Dict[int, List[int]]:
    """Split a sequence into approximately equal chronological intervals.

    When splitting by index, intervals contain sample indexes directly. When
    splitting by turn, complete turns are assigned to intervals and then
    expanded back into their sample indexes, so a turn is never divided across
    two intervals.

    Parameters
    ----------
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

    Returns
    -------
    dict
        Mapping from zero-based interval IDs to sample-index lists. Interval
        sizes differ by at most one unit before turn expansion.

    Raises
    ------
    ValueError
        If ``interval_num`` is not positive or ``interval_split_by`` is invalid.
    """
    if interval_num <= 0:
        raise ValueError("interval_num must be a positive integer.")
    if interval_split_by not in {"index", "turn"}:
        raise ValueError("interval_split_by must be 'index' or 'turn'.")

    turn2ref_value_and_indexes = None
    if interval_split_by == "index":
        units = _resolve_used_indexes(data_dicts, used_indexes)
    else:
        turn2ref_value_and_indexes = get_turn2ref_value_and_indexes(
            data_dicts,
            ref_key=ref_key,
            used_indexes=used_indexes,
        )
        units = list(turn2ref_value_and_indexes.keys())

    base_size, remainder = divmod(len(units), interval_num)
    interval2indexes: Dict[int, List[int]] = {}
    start = 0

    for interval in range(interval_num):
        interval_size = base_size + (1 if interval < remainder else 0)
        selected_units = units[start:start + interval_size]
        start += interval_size

        if interval_split_by == "index":
            interval2indexes[interval] = selected_units
        else:
            interval_indexes: List[int] = []
            for turn in selected_units:
                interval_indexes.extend(
                    turn2ref_value_and_indexes[turn]["indexes"]
                )
            interval2indexes[interval] = interval_indexes

    return interval2indexes
