# RecLib

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/Version-0.0.1-orange.svg)

[English Version](README.md)

**统一、高效、可扩展的推荐系统框架**

</div>


## 简介

RecLib 是一个基于 PyTorch 构建的现代推荐系统框架，为研究人员与工程团队提供统一的建模、训练与评估体验。框架采用模块化设计，内置丰富的模型实现、数据处理工具和工程化训练组件，可快速覆盖多种推荐场景。

### 核心特性

- **多场景推荐能力**：同时覆盖排序（CTR/CVR）、召回、多任务学习以及 TIGER、HSTU 等生成式推荐模型，持续扩充模型库。
- **统一的特征工程与数据流水线**：提供 Dense/Sparse/Sequence 特征定义、可持久化的 DataProcessor、批处理优化的 RecDataLoader，贯穿“定义-处理-加载”全链路。
- **高效训练与评估**：标准化训练引擎内置多种优化器、学习率调度、早停、模型检查点与彩色日志，开箱即用。
- **内置数据集与基线**：一行代码即可下载 MovieLens、Amazon、Criteo、Avazu 等常用数据集，并提供预处理和切分工具。
- **友好的工程体验**：模块化设计便于扩展，配套教程齐全，并支持 GPU/MPS 加速与可视化监控。

---

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         RecLib 框架                              │
└─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
        ┌───────▼────────┐                 ┌────────▼────────┐
        │    数据流水线   │                 │    模型层       │
        └───────┬────────┘                 └────────┬────────┘
                │                                   │
    ┌───────────┼───────────┐          ┌───────────┼──────────┐
    │           │           │          │           │          │
┌───▼───┐  ┌───▼────┐  ┌──▼───┐  ┌───▼────┐  ┌───▼───┐  ┌──▼──────┐
│特征   │  │数据集  │  │数据  │  │排序    │  │召回   │  │多任务   │
│处理   │  │加载器  │  │批次  │  │模型    │  │模型   │  │模型     │
└───┬───┘  └───┬────┘  └──┬───┘  └───┬────┘  └───┬───┘  └──┬──────┘
    │          │          │          │           │         │
    └──────────┴──────────┴──────────┴───────────┴─────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │       训练引擎            │
                    ├───────────────────────────┤
                    │  • 优化器                 │
                    │  • 损失函数               │
                    │  • 学习率调度             │
                    │  • 早停机制               │
                    │  • 模型检查点             │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │       评估模块            │
                    ├───────────────────────────┤
                    │  • 分类指标               │
                    │  • 排序指标               │
                    │  • 自定义指标             │
                    └───────────────────────────┘
```

---

## 快速开始

### 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0

### 安装

RecLib 提供两种主流的安装方式：**UV（推荐）** 与传统 pip/source 安装。

#### 方法一：使用 UV（推荐）

UV 是一款高速、现代化的 Python 包管理器，能够带来更快的依赖解析与安装体验。

```bash
# 1. 克隆仓库
git clone https://github.com/zerolovesea/RecLib.git
cd RecLib

# 2. 安装 UV（若尚未安装）
pip install uv

# 3. 同步依赖并创建虚拟环境
python -m uv sync
# - 创建 .venv 虚拟环境
# - 从 pyproject.toml 安装所有依赖
# - 以开发模式安装项目

# 4. 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows

# 5.（可选）安装开发依赖
python -m uv sync --extra dev
```

完成后可以运行测试验证环境：

```bash
python -m uv run pytest test/ -v
```

#### 方法二：使用 pip/source 安装

```bash
# 克隆仓库
git clone https://github.com/zerolovesea/RecLib.git
cd RecLib

# 以可编辑模式安装
pip install -e .

# 或仅安装依赖
pip install -r requirements.txt

# （可选）安装测试/开发依赖
pip install -r test_requirements.txt
```

完成后运行 `pytest test/ -v` 或 `python -m pytest` 即可执行单元测试。

### 10分钟教程

以下示例展示了如何使用 DeepFM 进行 CTR 预测，并串联起「特征定义 → 数据处理 → 数据加载 → 模型训练」的完整流程：

```python
import pandas as pd

from reclib.models.ranking.deepfm import DeepFM
from reclib.basic.dataloader import RecDataLoader
from reclib.data.preprocessor import DataProcessor
from reclib.basic.features import DenseFeature, SparseFeature, SequenceFeature

# 1. 准备原始数据
df = pd.read_csv('your_data.csv')
target = 'label'

dense_cols = ['age', 'income']
sparse_cols = ['gender', 'city', 'category']
sequence_cols = ['his_click_item']

# 2. 构建特征处理器
processor = DataProcessor()

for feat in dense_cols:
    processor.add_numeric_feature(feat, scaler='standard')
for feat in sparse_cols:
    processor.add_sparse_feature(feat, encode_method='hash', hash_size=1000)  # 也支持 label encoding
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

# 3. 拟合并持久化预处理流程
processor.fit(df)
processor.save('./processor/example_deepfm_processor.pkl')
df_transformed = processor.transform(df, return_dict=True)

# 4. 初始化特征描述
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

# 5. 构建 DataLoader（可选，也可直接传入 DataFrame/Dict）
dataloader = RecDataLoader(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    target=target,
)

# 6. 初始化模型与超参数
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    sequence_features=sequence_features,
    mlp_params={"dims": [256, 128], "activation": "relu", "dropout": 0.5},
    target=target,
    device='mps',
    model_id="deepfm_with_processor",
    embedding_l1_reg=1e-6,
    dense_l1_reg=1e-5,
    embedding_l2_reg=1e-5,
    dense_l2_reg=1e-4,
)

# 7. 配置训练参数
model.compile(
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    loss="bce"
)

# 8. 开始训练
model.fit(
    train_data=df_transformed,  # 支持 dict/DataFrame/RecDataLoader
    metrics=['auc', 'recall', 'precision'],
    epochs=10,
    batch_size=512,
    shuffle=True,
    verbose=1
)

# 9. 推理
preds = model.predict(df_transformed, batch_size=512)
```

### 使用 UV 运行脚本

借助 UV 可以在无需手动激活虚拟环境的情况下直接运行脚本：

```bash
# 运行示例脚本
python -m uv run python tutorials/example_deepfm.py

# 运行单元测试
python -m uv run pytest test/ -v

# 运行 Jupyter Notebook
python -m uv run jupyter notebook
```

### 更多示例

`tutorials/` 目录提供了覆盖排序、召回、多任务、数据处理等场景的示例：

- `example_deepfm.py` - 完整的 DeepFM 排序模型示例
- `example_ranking_din.py` - DIN 深度兴趣网络示例
- `example_match_dssm.py` - DSSM 召回模型示例
- `example_multitask.py` - 多任务学习示例
- `example_datasets.py` - 数据集下载与使用示例
- `example_dataloader.py` - DataLoader 使用示例
- `example_dataloader_integration.py` - DataLoader 与训练流程集成示例
- `data_generator.py` - 测试数据生成工具

### 数据集示例

RecLib 提供统一的数据集接口，支持一键下载与使用：

```python
from reclib.datasets import get_dataset, list_datasets

# 列出所有可用数据集
print(list_datasets())
# ['movielens-100k', 'movielens-1m', 'movielens-25m',
#  'criteo', 'amazon-books', 'amazon-electronics', 'avazu']

# 下载并加载 MovieLens 100K
dataset = get_dataset("movielens-100k", root="./data", download=True)
dataset.info()

# 载入带有用户/物品特征的数据
df = dataset.load(include_features=True)
print(df.head())

# 数据预处理示例
from reclib.datasets.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df = preprocessor.create_binary_labels(df, rating_col='rating', threshold=4.0)
df = preprocessor.encode_categorical(df, ['user_id', 'item_id', 'gender'])
train_df, valid_df, test_df = preprocessor.split_by_ratio(df, ratios=(0.7, 0.15, 0.15))
```

---

## 支持的模型

### 排序模型

| 模型 | 论文 | 年份 | 状态 |
|------|------|------|------|
| **DeepFM** | DeepFM: A Factorization-Machine based Neural Network for CTR Prediction | IJCAI 2017 | 已支持 |
| **Wide&Deep** | Wide & Deep Learning for Recommender Systems | DLRS 2016 | 已支持 |
| **xDeepFM** | xDeepFM: Combining Explicit and Implicit Feature Interactions | KDD 2018 | 已支持 |
| **AutoInt** | AutoInt: Automatic Feature Interaction Learning | CIKM 2019 | 已支持 |
| **DCN** | Deep & Cross Network for Ad Click Predictions | ADKDD 2017 | 已支持 |
| **DIN** | Deep Interest Network for Click-Through Rate Prediction | KDD 2018 | 已支持 |
| **DIEN** | Deep Interest Evolution Network for Click-Through Rate Prediction | AAAI 2019 | 已支持 |

### 召回模型

| 模型 | 论文 | 年份 | 状态 |
|------|------|------|------|
| **DSSM** | Learning Deep Structured Semantic Models | CIKM 2013 | 已支持 |
| **YouTube DNN** | Deep Neural Networks for YouTube Recommendations | RecSys 2016 | 已支持 |
| **MIND** | Multi-Interest Network with Dynamic Routing | CIKM 2019 | 已支持 |
| **SDM** | Sequential Deep Matching Model | - | 已支持 |

### 多任务模型

| 模型 | 论文 | 年份 | 状态 |
|------|------|------|------|
| **MMOE** | Modeling Task Relationships in Multi-task Learning | KDD 2018 | 已支持 |
| **PLE** | Progressive Layered Extraction | RecSys 2020 | 已支持 |
| **ESMM** | Entire Space Multi-Task Model | SIGIR 2018 | 已支持 |
| **ShareBottom** | Multitask Learning | - | 已支持 |

### 生成式模型

| 模型 | 论文 | 年份 | 状态 |
|------|------|------|------|
| **TIGER** | Recommender Systems with Generative Retrieval | NeurIPS 2023 | 开发中 |
| **HSTU** | Hierarchical Sequential Transduction Units | - | 开发中 |

---

## 核心能力

### 特征定义

RecLib 提供 Dense、Sparse、Sequence 三类特征描述，可统一管理 Embedding 与特征交互配置：

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

### 自定义模型

通过继承 `BaseModel` 可以快速扩展新模型，并复用训练、日志、评估等基础设施：

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

### 训练与评估

```python
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

predictions = model.predict(test_df, batch_size=1024)

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_df['label'], predictions)
print(f"Test AUC: {auc:.6f}")
```

---

## 贡献指南

我们欢迎任何形式的贡献！

### 如何贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

> 在提交 PR 之前，请运行 `pytest test/ -v` 或 `python -m pytest` 确保所有测试通过。

### 代码规范

- 遵循 PEP 8 Python 代码风格
- 为新增功能补充单元测试
- 同步更新相关文档

### 报告错误

在 [Issues](https://github.com/zerolovesea/RecLib/issues) 页面提交问题时，请包含：

- 错误描述
- 重现步骤
- 期望行为
- 实际行为
- 环境信息（Python 版本、PyTorch 版本等）

---

## 许可证

本项目采用 [Apache 2.0 许可证](./LICENSE)。

---

## 联系方式

- **GitHub Issues**: [提交问题](https://github.com/zerolovesea/RecLib/issues)
- **邮箱**: zyaztec@gmail.com

---

## 致谢

RecLib 的开发受到以下优秀项目的启发：

- [FuxiCTR](https://github.com/reczoo/FuxiCTR) - 可配置、可调优、可复现的 CTR 预测库
- [RecBole](https://github.com/RUCAIBox/RecBole) - 统一、全面、高效的推荐库
- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec) - 大规模推荐算法库

感谢开源社区的所有贡献者！

---

<div align="center">

**[返回顶部](#reclib)**

</div>
