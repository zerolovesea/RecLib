"""
Dataset registry for managing available datasets.
"""

from typing import Dict, Type, List
from .base import BaseDataset

_DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {}


def register_dataset(name: str):
    """
    Decorator to register a dataset class.
    
    Usage:
        @register_dataset("movielens-100k")
        class MovieLens100K(BaseDataset):
            ...
    """
    def decorator(cls: Type[BaseDataset]):
        if name in _DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered!")
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name: str, **kwargs) -> BaseDataset:
    """
    Get a dataset by name.
    
    Args:
        name: Name of the dataset.
        **kwargs: Additional arguments passed to the dataset constructor.
    
    Returns:
        Instance of the requested dataset.
    
    Example:
        >>> dataset = get_dataset("movielens-100k", root="./data")
        >>> df = dataset.load()
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' not found. Available datasets: {list_datasets()}"
        )
    return _DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> List[str]:
    """List all available datasets."""
    return sorted(_DATASET_REGISTRY.keys())
