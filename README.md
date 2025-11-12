# RecLib

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/Version-0.0.1-orange.svg)

[中文版](README_zh.md)

**A Unified, Efficient, and Extensible Deep Learning Framework for Recommender Systems**

</div>


## Introduction

RecLib is a modern recommender-system framework built on PyTorch that brings researchers and engineers a unified experience for modeling, training, and evaluation. Its modular design, rich model zoo, data processing utilities, and production-ready training components allow you to cover ranking, matching, multi-task, and generative recommendation workloads quickly.

### Key Features

- **Multi-Scenario Coverage**: Ranking (CTR/CVR), matching, multi-task learning, plus cutting-edge generative recommenders such as TIGER and HSTU with continuous additions to the model zoo.
- **Unified Feature Engineering Pipeline**: Dense/Sparse/Sequence feature abstractions, a persistable `DataProcessor`, and an optimized `RecDataLoader` connect definition, processing, and loading stages.
- **Efficient Training & Evaluation**: Standardized training engine with multiple optimizers, LR schedulers, early stopping, checkpoints, and colorized logs for instant feedback.
- **Built-in Datasets & Baselines**: One-line download for MovieLens, Amazon, Criteo, Avazu, etc., complete with preprocessing helpers and dataset splits.
- **Developer-Friendly Experience**: Modular architecture, comprehensive tutorials, GPU/MPS acceleration, and hooks for monitoring/visualization.

---

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         RecLib Framework                         │
└─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
        ┌───────▼────────┐                 ┌────────▼────────┐
        │  Data Pipeline │                 │   Model Layer   │
        └───────┬────────┘                 └────────┬────────┘
                │                                   │
    ┌───────────┼───────────┐          ┌───────────┼──────────┐
    │           │           │          │           │          │
┌───▼───┐  ┌───▼────┐  ┌──▼───┐  ┌───▼────┐  ┌───▼───┐  ┌──▼──────┐
│Feature│  │Dataset │  │Data  │  │Ranking │  │Matching│  │Multi-   │
│Process│  │Loader  │  │Batch │  │Models  │  │Models  │  │Task     │
└───┬───┘  └───┬────┘  └──┬───┘  └───┬────┘  └───┬───┘  └──┬──────┘
    │          │          │          │           │         │
    └──────────┴──────────┴──────────┴───────────┴─────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     Training Engine       │
                    ├───────────────────────────┤
                    │  • Optimizers             │
                    │  • Loss Functions         │
                    │  • LR Scheduling          │
                    │  • Early Stopping         │
                    │  • Checkpointing          │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Evaluation Module      │
                    ├───────────────────────────┤
                    │  • Classification Metrics │
                    │  • Ranking Metrics        │
                    │  • Custom Metrics         │
                    └───────────────────────────┘
```

---

## Quick Start

### Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0

### Installation

RecLib supports two mainstream installation methods: **UV (recommended)** and traditional pip/source installation.

#### Method 1: UV (Recommended)

UV is a fast, modern Python package manager that offers superior dependency resolution and installation speed.

```bash
# 1. Clone the repository
git clone https://github.com/zerolovesea/RecLib.git
cd RecLib

# 2. Install UV (if needed)
pip install uv

# 3. Sync dependencies and create the virtual environment
python -m uv sync
# - Creates a .venv virtual environment
# - Installs all dependencies from pyproject.toml
# - Sets up the project in editable mode

# 4. Activate the environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# 5. (Optional) Install dev extras
python -m uv sync --extra dev
```

Afterwards, run the tests to verify the setup:

```bash
python -m uv run pytest test/ -v
```

#### Method 2: pip / source install

```bash
# Clone the repository
git clone https://github.com/zerolovesea/RecLib.git
cd RecLib

# Install in editable mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt

# (Optional) Install testing/dev requirements
pip install -r test_requirements.txt
```

Then execute `pytest test/ -v` or `python -m pytest` to ensure everything passes.

### 10-Minute Tutorial

This example walks through DeepFM for CTR prediction and ties together feature definition, preprocessing, loading, and training:

```python
import pandas as pd

from reclib.models.ranking.deepfm import DeepFM
from reclib.basic.dataloader import RecDataLoader
from reclib.data.preprocessor import DataProcessor
from reclib.basic.features import DenseFeature, SparseFeature, SequenceFeature

# 1. Prepare raw data
df = pd.read_csv('your_data.csv')
target = 'label'

dense_cols = ['age', 'income']
sparse_cols = ['gender', 'city', 'category']
sequence_cols = ['his_click_item']

# 2. Build a preprocessing pipeline
processor = DataProcessor()

for feat in dense_cols:
    processor.add_numeric_feature(feat, scaler='standard')
for feat in sparse_cols:
    processor.add_sparse_feature(feat, encode_method='hash', hash_size=1000)  # label encoding is supported as well
for feat in sequence_cols:
    processor.add_sequence_feature(
        feat,
        encode_method='hash',
        hash_size=5000,
        max_len=20,
        pad_value=0,
        truncate='post',
        separator=','
    )

# 3. Fit and persist the processor
processor.fit(df)
processor.save('./processor/example_deepfm_processor.pkl')
df_transformed = processor.transform(df, return_dict=True)

# 4. Define feature metadata
dense_features = [DenseFeature(feat) for feat in dense_cols]
vocab_sizes = processor.get_vocab_sizes()

sparse_features = [
    SparseFeature(feat, vocab_size=vocab_sizes.get(feat, 1000), embedding_dim=10)
    for feat in sparse_cols
]
sequence_features = [
    SequenceFeature(feat, vocab_size=vocab_sizes.get(feat, 5000), max_len=50, embedding_dim=10, padding_idx=0, combiner='mean')
    for feat in sequence_cols
]

# 5. Build the DataLoader (optional, models also accept dict/DataFrame)
dataloader = RecDataLoader(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    target=target,
)

# 6. Instantiate the model
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    mlp_params={"dims": [256, 128], "activation": "relu", "dropout": 0.5},
    target=target,
    device='cuda',  # or 'mps' / 'cpu'
    model_id="deepfm_with_processor",
    embedding_l1_reg=1e-6,
    dense_l1_reg=1e-5,
    embedding_l2_reg=1e-5,
    dense_l2_reg=1e-4,
)

# 7. Configure the training hyperparameters
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    loss="bce"
)

# 8. Train
model.fit(
    train_data=df_transformed,
    metrics=['auc', 'recall', 'precision'],
    epochs=10,
    batch_size=512,
    shuffle=True,
    verbose=1
)

# 9. Predict
preds = model.predict(df_transformed, batch_size=512)
```

### Running with UV

UV lets you run commands without manually activating the virtual environment:

```bash
# Run an example script
python -m uv run python tutorials/example_deepfm.py

# Run the unit tests
python -m uv run pytest test/ -v

# Launch Jupyter
python -m uv run jupyter notebook
```

### More Examples

The `tutorials/` directory contains end-to-end samples for common tasks:

- `example_deepfm.py` - Complete DeepFM ranking example
- `example_ranking_din.py` - DIN (Deep Interest Network) example
- `example_match_dssm.py` - DSSM matching model example
- `example_multitask.py` - Multi-task learning example
- `example_datasets.py` - Dataset download and usage examples
- `example_dataloader.py` - DataLoader usage example
- `example_dataloader_integration.py` - DataLoader + training integration walkthrough
- `data_generator.py` - Synthetic data generator for quick experiments

### Dataset Example

RecLib ships a unified dataset interface for downloading and preparing popular benchmarks:

```python
from reclib.datasets import get_dataset, list_datasets

# List available datasets
print(list_datasets())
# ['movielens-100k', 'movielens-1m', 'movielens-25m',
#  'criteo', 'amazon-books', 'amazon-electronics', 'avazu']

# Download and load MovieLens 100K
dataset = get_dataset("movielens-100k", root="./data", download=True)
dataset.info()

# Load data with user/item side features
df = dataset.load(include_features=True)
print(df.head())

# Preprocess
from reclib.datasets.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df = preprocessor.create_binary_labels(df, rating_col='rating', threshold=4.0)
df = preprocessor.encode_categorical(df, ['user_id', 'item_id', 'gender'])
train_df, valid_df, test_df = preprocessor.split_by_ratio(df, ratios=(0.7, 0.15, 0.15))
```

---

## Supported Models

### Ranking Models

| Model | Paper | Year | Status |
|------|------|------|------|
| **DeepFM** | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction | IJCAI 2017 | Supported |
| **Wide&Deep** | Wide & Deep Learning for Recommender Systems | DLRS 2016 | Supported |
| **xDeepFM** | xDeepFM: Combining Explicit and Implicit Feature Interactions | KDD 2018 | Supported |
| **AutoInt** | AutoInt: Automatic Feature Interaction Learning | CIKM 2019 | Supported |
| **DCN** | Deep & Cross Network for Ad Click Predictions | ADKDD 2017 | Supported |
| **DIN** | Deep Interest Network for Click-Through Rate Prediction | KDD 2018 | Supported |
| **DIEN** | Deep Interest Evolution Network for Click-Through Rate Prediction | AAAI 2019 | Supported |

### Matching Models

| Model | Paper | Year | Status |
|------|------|------|------|
| **DSSM** | Learning Deep Structured Semantic Models | CIKM 2013 | Supported |
| **YouTube DNN** | Deep Neural Networks for YouTube Recommendations | RecSys 2016 | Supported |
| **MIND** | Multi-Interest Network with Dynamic Routing | CIKM 2019 | Supported |
| **SDM** | Sequential Deep Matching Model | - | Supported |

### Multi-Task Models

| Model | Paper | Year | Status |
|------|------|------|------|
| **MMOE** | Modeling Task Relationships in Multi-task Learning | KDD 2018 | Supported |
| **PLE** | Progressive Layered Extraction | RecSys 2020 | Supported |
| **ESMM** | Entire Space Multi-Task Model | SIGIR 2018 | Supported |
| **ShareBottom** | Multitask Learning | - | Supported |

### Generative Models

| Model | Paper | Year | Status |
|------|------|------|------|
| **TIGER** | Recommender Systems with Generative Retrieval | NeurIPS 2023 | In Progress |
| **HSTU** | Hierarchical Sequential Transduction Units | - | In Progress |

---

## Core Capabilities

### Feature Definition

RecLib exposes three unified feature abstractions so you can manage embeddings and interaction logic consistently:

```python
from reclib.basic.features import DenseFeature, SparseFeature, SequenceFeature

dense_feat = DenseFeature(
    feature_name='age',
    feature_dim=1
)

sparse_feat = SparseFeature(
    feature_name='category',
    vocab_size=100,
    embedding_dim=16,
    embedding_name='category_emb'
)

sequence_feat = SequenceFeature(
    feature_name='click_history',
    vocab_size=1000,
    embedding_dim=32,
    pooling='mean',
    max_length=50
)
```

### Custom Model

Extend `BaseModel` to bring your own architectures while reusing training, logging, and evaluation plumbing:

```python
from reclib.basic.model import BaseModel
import torch
import torch.nn as nn

class YourModel(BaseModel):
    @property
    def model_name(self):
        return "YourModel"
    
    @property  
    def task_type(self):
        return "binary"
    
    def __init__(self, dense_features, sparse_features, **kwargs):
        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            **kwargs
        )
        
        self.dnn = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        dense_input = x['dense']
        sparse_input = self.embedding(x['sparse'])
        combined = torch.cat([dense_input, sparse_input], dim=-1)
        output = self.dnn(combined)
        return output
```

### Training & Evaluation

```python
# Training
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=['auc', 'logloss'],
    epochs=20,
    batch_size=1024,
    shuffle=True,
    verbose=1,
    early_stop_patience=5
)

# Prediction
predictions = model.predict(test_df, batch_size=1024)

# Evaluation
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_df['label'], predictions)
print(f"Test AUC: {auc:.6f}")
```

---

## Contributing

We welcome contributions of all kinds!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

> Please run `pytest test/ -v` (or `python -m pytest`) before submitting a PR.

### Code Standards

- Follow the PEP 8 Python style guide
- Add unit tests for new functionality
- Update relevant documentation

### Reporting Bugs

When filing issues on [GitHub Issues](https://github.com/zerolovesea/RecLib/issues), please include:

- Bug description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment info (Python version, PyTorch version, etc.)

---

## License

This project is licensed under the [Apache 2.0 License](./LICENSE).

---

## Contact

- **GitHub Issues**: [Submit Issues](https://github.com/zerolovesea/RecLib/issues)
- **Email**: zyaztec@gmail.com

---

## Acknowledgments

RecLib draws inspiration from these excellent projects:

- [FuxiCTR](https://github.com/reczoo/FuxiCTR) - Configurable, tunable, and reproducible CTR prediction library
- [RecBole](https://github.com/RUCAIBox/RecBole) - Unified, comprehensive, and efficient recommendation library
- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec) - Large-scale recommendation algorithm library

Thanks to every contributor in the open-source community!

---

<div align="center">

**[Back to Top](#reclib)**

Made with care by the RecLib Team

</div>
