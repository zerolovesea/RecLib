"""
Data utilities package for RecLib

This package provides data processing and manipulation utilities.

Date: create on 13/11/2025
Author:
    Yang Zhou, zyaztec@gmail.com
"""

from recforge.data.data_utils import (
    collate_fn,
    get_column_data,
    split_dict_random,
    build_eval_candidates,
)

# For backward compatibility, keep utils accessible
from recforge.data import data_utils

__all__ = [
    'collate_fn',
    'get_column_data',
    'split_dict_random',
    'build_eval_candidates',
    'data_utils',
]
