"""
NextRec - A Unified Deep Learning Framework for Recommender Systems
===================================================================

NextRec provides a comprehensive suite of recommendation models including:
- Ranking models (CTR prediction)
- Matching models (retrieval)
- Multi-task learning models
- Generative recommendation models

Quick Start
-----------
>>> from nextrec.basic.features import DenseFeature, SparseFeature
>>> from nextrec.models.ranking.deepfm import DeepFM
>>>
>>> # Define features
>>> dense_features = [DenseFeature('age')]
>>> sparse_features = [SparseFeature('category', vocab_size=100, embedding_dim=16)]
>>>
>>> # Build model
>>> model = DeepFM(
...     dense_features=dense_features,
...     sparse_features=sparse_features,
...     targets=['label']
... )
>>>
>>> # Train model
>>> model.fit(train_data=df_train, valid_data=df_valid)
"""

from nextrec.__version__ import __version__

__all__ = [
    "__version__",
]

# Package metadata
__author__ = "zerolovesea"
__email__ = "zyaztec@gmail.com"
__license__ = "Apache 2.0"
__url__ = "https://github.com/zerolovesea/NextRec"
