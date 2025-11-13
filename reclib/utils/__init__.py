"""
Utilities package for RecLib

This package provides various utility functions organized by functionality:
- optimizer: Optimizer and scheduler utilities
- initializer: Parameter initialization utilities
- embedding: Embedding dimension calculation
- common: Common utility functions

For backward compatibility, all functions are also available at the package level.

Date: create on 13/11/2025
Author:
    Yang Zhou, zyaztec@gmail.com
"""

# Import from submodules
from reclib.utils.optimizer import get_optimizer_fn, get_scheduler_fn
from reclib.utils.initializer import get_initializer_fn
from reclib.utils.embedding import get_auto_embedding_dim
from reclib.utils.common import get_task_type

# Backward compatibility - keep tools module accessible
from reclib.utils import optimizer, initializer, embedding, common

__all__ = [
    # Optimizer and scheduler
    'get_optimizer_fn',
    'get_scheduler_fn',
    
    # Initializer
    'get_initializer_fn',
    
    # Embedding
    'get_auto_embedding_dim',
    
    # Common
    'get_task_type',
    
    # Submodules
    'optimizer',
    'initializer',
    'embedding',
    'common',
]
