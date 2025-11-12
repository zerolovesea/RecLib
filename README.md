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

RecLib is a modern deep learning framework for recommender systems built on PyTorch, designed for both researchers and engineers. The framework provides a clear modular architecture, rich model library, and flexible feature engineering capabilities, supporting various recommendation scenarios.

### Key Features

- **Multi-Task Scenario Support**
  - Ranking Models (CTR Prediction): DeepFM, DCN, xDeepFM, DIN, DIEN, WideDeep, AutoInt, etc.
  - Matching Models: DSSM, MIND, YouTube DNN, SDM, etc.
  - Multi-Task Learning: MMOE, PLE, ESMM, ShareBottom
  - Generative Recommendation: TIGER, HSTU and other cutting-edge models

- **Flexible Feature Engineering**
  - Unified feature definition interface: DenseFeature, SparseFeature, SequenceFeature
  - Automated embedding layer management
  - Support for various feature interaction methods

- **Common Dataset Support**
  - MovieLens (100K, 1M, 25M): Movie rating datasets
  - Amazon Reviews (Books, Electronics, Movies): Product review datasets
  - Criteo: Display advertising click-through rate dataset
  - Avazu: Mobile advertising click dataset
  - One-click download, loading, and preprocessing

- **Efficient Training Pipeline**
  - Built-in DataLoader with batch processing optimization
  - Multiple optimizers and learning rate scheduling strategies
  - Early Stopping and model checkpoint management
  - GPU/MPS acceleration support

- **Comprehensive Evaluation System**
  - Classification metrics: AUC, LogLoss, Accuracy, etc.
  - Ranking metrics: NDCG, MRR, Recall, etc.
  - Custom metric extension interface

- **Developer-Friendly Experience**
  - Colorized logging for clear training progress
  - Modular design for easy model extension
  - Rich example code and tutorials

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
                    │  • Optimizer              │
                    │  • Loss Functions         │
                    │  • Learning Rate Scheduler│
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

### Installation

RecLib supports two installation methods: **UV (Recommended)** and traditional pip.

#### Method 1: Using UV (Recommended)

UV is a fast, modern Python package manager. It provides better dependency resolution and faster installation.

```bash
# 1. Clone the repository
git clone https://github.com/zerolovesea/RecLib.git
cd RecLib

# 2. Install UV (if not already installed)
pip install uv

# 3. Sync dependencies and create virtual environment
python -m uv sync

# This will:
# - Create a .venv virtual environment
# - Install all dependencies from pyproject.toml
# - Set up the project in development mode

# 4. Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# 5. (Optional) Install development dependencies
python -m uv sync --extra dev
# This includes pytest, jupyter, matplotlib, and other dev tools
```

#### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/zerolovesea/RecLib.git
cd RecLib

# Install dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r test_requirements.txt
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0

### 10-Minute Tutorial

Here's a quick example using DeepFM for CTR prediction:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from reclib.models.ranking.deepfm import DeepFM
from reclib.basic.features import DenseFeature, SparseFeature

# 1. Prepare Data
df = pd.read_csv('your_data.csv')
target = 'label'

# Feature columns
dense_cols = ['age', 'income']
sparse_cols = ['gender', 'city', 'category']

# Encode sparse features
for col in sparse_cols:
    lbe = LabelEncoder()
    df[col] = lbe.fit_transform(df[col].astype(str))

# 2. Define Features
dense_features = [DenseFeature(name) for name in dense_cols]
sparse_features = [
    SparseFeature(name, vocab_size=df[name].nunique(), embedding_dim=16)
    for name in sparse_cols
]

# 3. Build Model
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    mlp_params={"dims": [256, 128], "activation": "relu", "dropout": 0.3},
    targets=[target],
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    device='cuda',  # or 'mps' (Mac) / 'cpu'
    model_id="deepfm_exp"
)

# 4. Train Model
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=['auc', 'logloss'],
    epochs=10,
    batch_size=512,
    shuffle=True
)

# 5. Make Predictions
predictions = model.predict(valid_df, batch_size=512)
```

### Running with UV

If you installed RecLib with UV, you can run scripts directly without activating the virtual environment:

```bash
# Run Python scripts
python -m uv run python tutorials/example_deepfm.py

# Run tests
python -m uv run pytest test/ -v

# Run Jupyter notebook
python -m uv run jupyter notebook

# Or activate the virtual environment first
source .venv/bin/activate
python tutorials/example_deepfm.py
pytest test/ -v
```

### More Examples

Check the `tutorials/` directory for more examples:

- `example_deepfm.py` - Complete DeepFM ranking model example
- `example_ranking_din.py` - DIN (Deep Interest Network) example
- `example_match_dssm.py` - DSSM matching model example
- `example_multitask.py` - Multi-task learning example
- `example_datasets.py` - Dataset download and usage examples
- `example_dataloader.py` - DataLoader usage example
- `example_dataloader_integration.py` - DataLoader integration example
- `data_generator.py` - Test data generation utility

### Using Datasets

RecLib provides a unified interface for common recommendation datasets:

```python
from reclib.datasets import get_dataset, list_datasets

# List all available datasets
print(list_datasets())
# ['movielens-100k', 'movielens-1m', 'movielens-25m', 
#  'criteo', 'amazon-books', 'amazon-electronics', 'avazu']

# Download and load MovieLens 100K dataset
dataset = get_dataset("movielens-100k", root="./data", download=True)
dataset.info()  # View dataset information

# Load data (with user and movie features)
df = dataset.load(include_features=True)
print(df.head())

# Data preprocessing
from reclib.datasets.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df = preprocessor.create_binary_labels(df, rating_col='rating', threshold=4.0)
df = preprocessor.encode_categorical(df, ['user_id', 'item_id', 'gender'])
train_df, valid_df, test_df = preprocessor.split_by_ratio(df, ratios=(0.7, 0.15, 0.15))
```

For detailed documentation, see [reclib/datasets/README.md](reclib/datasets/README.md)

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
| **TIGER** | Recommender Systems with Generative Retrieval | NeurIPS 2023 |  |
| **HSTU** | Hierarchical Sequential Transduction Units | - |  |

---

## Core Features

### Feature Definition

RecLib provides three types of features:

```python
from reclib.basic.features import DenseFeature, SparseFeature, SequenceFeature

# Dense feature - for continuous numerical values
dense_feat = DenseFeature(
    feature_name='age',
    feature_dim=1
)

# Sparse feature - for categorical variables
sparse_feat = SparseFeature(
    feature_name='category',
    vocab_size=100,
    embedding_dim=16,
    embedding_name='category_emb'  # Optional, for parameter sharing
)

# Sequence feature - for user behavior sequences
sequence_feat = SequenceFeature(
    feature_name='click_history',
    vocab_size=1000,
    embedding_dim=32,
    pooling='mean',  # 'mean', 'sum', 'max'
    max_length=50
)
```

### Custom Model

Inherit from `BaseModel` to create a new model:

```python
from reclib.basic.model import BaseModel
import torch.nn as nn

class YourModel(BaseModel):
    @property
    def model_name(self):
        return "YourModel"
    
    @property  
    def task_type(self):
        return "binary"  # or "regression", "multi_class"
    
    def __init__(self, dense_features, sparse_features, **kwargs):
        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            **kwargs
        )
        
        # Define model structure
        self.dnn = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        # Implement forward propagation
        dense_input = x['dense']
        sparse_input = self.embedding(x['sparse'])
        
        combined = torch.cat([dense_input, sparse_input], dim=-1)
        output = self.dnn(combined)
        return output
```

### Training and Evaluation

```python
# Training
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=['auc', 'logloss'],  # Multiple metrics evaluation
    epochs=20,
    batch_size=1024,
    shuffle=True,
    verbose=1,  # Logging verbosity
    early_stop_patience=5  # Early Stopping
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

### Code Standards

- Follow PEP 8 Python code style
- Add unit tests for new features
- Update relevant documentation

### Reporting Bugs

When submitting issues on the [Issues](https://github.com/zerolovesea/RecLib/issues) page, please include:

- Bug description
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment information (Python version, PyTorch version, etc.)

---

## License

This project is licensed under the [Apache 2.0 License](./LICENSE).

---

## Contact

- **GitHub Issues**: [Submit Issues](https://github.com/zerolovesea/RecLib/issues)
- **Email**: zyaztec@gmail.com

---

## Acknowledgments

RecLib's development was inspired by the following excellent projects:

- [FuxiCTR](https://github.com/reczoo/FuxiCTR) - A configurable, tunable, and reproducible library for CTR prediction
- [RecBole](https://github.com/RUCAIBox/RecBole) - A unified, comprehensive and efficient recommendation library
- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec) - Large-scale recommendation algorithm library

Thanks to all contributors in the open-source community!

---

<div align="center">

**[Back to Top](#reclib)**

Made with care by RecLib Team

</div>
