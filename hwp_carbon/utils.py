from __future__ import annotations

import warnings
from typing import Iterator, Any, Sized, Sequence

import pandas as pd
import numpy as np


def pandas_to_carbonnetwork_init_data(pandas_dict: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Format data to create a CarbonNetwork"""
    init_data = {'pools': {},
                 'arcs': {}}

    for df_name in ['pools', 'arcs', 'substitution_factor', 'arcs_factors']:
        if df_name not in pandas_dict:
            raise ValueError(f'"pandas_dict" must have "{df_name}" key')

    for pinfo in pandas_dict['pools'].to_dict('records'):
        init_data['pools'][pinfo['pool_name']] = {'half_life': pinfo['half_life'],
                                                  'name': pinfo['pool_name'],
                                                  'long_name': pinfo['long_name']
                                                  if not np.isnan(pinfo['long_name'])
                                                  else pinfo['pool_name']}

    for pool_name, sdata in pandas_dict['substitution_factor'].set_index('pool_name').to_dict('index').items():
        if pool_name not in init_data['pools']:
            raise ValueError(f'Pool {pool_name} is defined in "substitution_factor" but not in "pools"')
        init_data['pools'][pool_name]['substitution_factor'] = np.array(list(sdata.values()))

    for arc, arc_data in pandas_dict['arcs'].set_index(['src_pool', 'dst_pool']).to_dict('index').items():
        init_data['arcs'][arc] = arc_data
        for key, value in init_data['arcs'][arc].items():
            if np.isnan(value):
                match key:
                    case 'delay': _value = 0
                    case 'recycling': _value = False
                    case _: raise ValueError(f'{key} must have a value in "arcs"')
                init_data['arcs'][arc][key] = _value

    for arc, factors in pandas_dict['arcs_factors'].set_index(['src_pool', 'dst_pool']).to_dict('index').items():
        if arc not in init_data['arcs']:
            raise ValueError(f'Arc {arc} is defined in "arcs_factors" but not in "arcs"')
        init_data['arcs'][arc]['factor'] = np.array(list(factors.values()))

    return init_data


def excel_to_carbonnetwork_init_data(excel_path: str) -> dict[str, dict]:
    """Reads an Excel file correctly formatted and transform the data to create a CarbonNetwork"""
    # Remove warning when we load the file (DataValidation is not supported but we don't care much)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        pd_data = pd.read_excel(excel_path, sheet_name=None)  # dict[sheet_name, pd.DataFrame]

    for df_name in ['pools', 'arcs', 'substitution_factor', 'arcs_factors']:
        if df_name not in pd_data:
            raise ValueError(f'The Excel file must have a "{df_name}" sheet')

    init_data = pandas_to_carbonnetwork_init_data(pd_data)
    return init_data


def adjust_array(sequence: np.ndarray, wanted_length: int = None, keep_left: bool = True,
                 interpolate_nan: bool = True, value_to_add: Any = np.nan) -> np.ndarray:
    """Slice or append values to a np.ndarray to reach wanted length

    Args:
        sequence: Array to adjust
        wanted_length: length of the array returned
        keep_left: if True and array is sliced (reduce length), will keep beginning values.
                If False, will keep ending of the array.
        interpolate_nan: if True will replace nan using linear interpolation.
        value_to_add: If interpolate_nan is False, must

    Returns:
        np.ndarray

    """
    if not interpolate_nan and np.isnan(value_to_add):
        raise ValueError(f"if interpolate_nan is False, value_to_add must be a number")
    elif interpolate_nan and not np.isnan(value_to_add) and len(sequence) < wanted_length:
        raise ValueError(f"if interpolate_nan is True, value_to_add must not be used")

    def interpolate_nan(array: np.ndarray) -> np.ndarray:
        """Replace nan values with linear interpolation. if nan is not between two values, repeat first or last value"""
        mask = np.isnan(array)
        array[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), array[~mask])
        return array

    if wanted_length is None or len(sequence) == wanted_length:
        pass
    elif len(sequence) > wanted_length:  # len(sequence) > wanted_length
        if keep_left:
            sequence = sequence[:wanted_length]
        else:
            sequence = sequence[len(sequence) - wanted_length:]
    elif len(sequence) < wanted_length:
        sequence = np.append(sequence, [value_to_add] * (wanted_length - len(sequence)))

    any_nan = any([np.isnan(value) for value in sequence])
    if any_nan and interpolate_nan:
        sequence = interpolate_nan(sequence)

    return sequence


def cyclic(graph: dict[Any, Sequence[Any]]) -> tuple[bool, str]:
    """Return True if the directed graph has a cycle.
    The graph must be represented as a dictionary mapping vertices to
    iterables of neighbouring vertices. For example:

    >>> cyclic({1: (2,), 2: (3,), 3: (1,)})
    True
    >>> cyclic({1: (2,), 2: (3,), 3: (4,)})
    False

    https://codereview.stackexchange.com/questions/86021/check-if-a-directed-graph-contains-a-cycle
    """
    visited = set()
    path = [object()]
    path_set = set(path)
    stack = [iter(graph)]
    while stack:
        for v in stack[-1]:
            if v in path_set:
                return True, f'Loop on node {v}'
            elif v not in visited:
                visited.add(v)
                path.append(v)
                path_set.add(v)
                stack.append(iter(graph.get(v, ())))
                break
        else:
            path_set.remove(path.pop())
            stack.pop()
    return False, f'No loop found'

