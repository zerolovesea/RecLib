"""
Data utilities package for NextRec

This package provides data processing and manipulation utilities.

Date: create on 13/11/2025
Author:
    Yang Zhou, zyaztec@gmail.com
"""

from nextrec.data.data_utils import (
    collate_fn,
    get_column_data,
    split_dict_random,
    build_eval_candidates,
)

# For backward compatibility, keep utils accessible
from nextrec.data import data_utils

__all__ = [
    'collate_fn',
    'get_column_data',
    'split_dict_random',
    'build_eval_candidates',
    'data_utils',
]
