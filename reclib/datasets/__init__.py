"""
Dataset utilities for RecLib.

This module provides utilities for downloading and loading common recommendation datasets.
"""

from .base import BaseDataset, DatasetConfig
from .registry import register_dataset, get_dataset, list_datasets
from .movielens import MovieLens100K, MovieLens1M, MovieLens25M
from .criteo import CriteoDataset

__all__ = [
    "BaseDataset",
    "DatasetConfig",
    "register_dataset",
    "get_dataset",
    "list_datasets",
    "MovieLens100K",
    "MovieLens1M",
    "MovieLens25M",
    "CriteoDataset",
]
