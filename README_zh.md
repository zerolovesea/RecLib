# RecLib

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/Version-0.0.1-orange.svg)

**统一、高效、可扩展的推荐系统深度学习框架**

</div>

---

## 简介

RecLib 是一个基于 PyTorch 构建的现代推荐系统深度学习框架，专为研究人员和工程师设计。该框架提供清晰的模块化架构、丰富的模型库和灵活的特征工程能力，支持多种推荐场景。

### 核心特性

- **多任务场景支持**
  - 排序模型（CTR预测）：DeepFM、DCN、xDeepFM、DIN、DIEN、WideDeep、AutoInt 等
  - 召回模型：DSSM、MIND、YouTube DNN、SDM 等
  - 多任务学习：MMOE、PLE、ESMM、ShareBottom
  - 生成式推荐：TIGER、HSTU 等前沿模型

- **灵活的特征工程**
  - 统一的特征定义接口：DenseFeature、SparseFeature、SequenceFeature
  - 自动化的 Embedding 层管理
  - 支持多种特征交互方式

- **常用数据集支持**
  - MovieLens (100K, 1M, 25M): 电影评分数据
  - Amazon Reviews (Books, Electronics, Movies): 商品评论数据
  - Criteo: 广告点击率预测数据
  - Avazu: 移动广告点击数据
  - 一键下载、加载和预处理

- **高效的训练流程**
  - 内置 DataLoader，支持批处理优化
  - 多种优化器和学习率调度策略
  - Early Stopping 和模型检查点管理
  - GPU/MPS 加速支持

- **完善的评估体系**
  - 分类指标：AUC、LogLoss、Accuracy 等
  - 排序指标：NDCG、MRR、Recall 等
  - 自定义指标扩展接口

- **开发者友好**
  - 彩色日志输出，清晰展示训练进度
  - 模块化设计，易于扩展模型
  - 丰富的示例代码和教程

---

## 框架架构

### 目录结构

```
RecLib/
├── reclib/
│   ├── basic/              # 基础组件
│   │   ├── model.py        # 模型基类
│   │   ├── features.py     # 特征定义
│   │   ├── layers.py       # 通用网络层
│   │   ├── activation.py   # 激活函数
│   │   ├── callback.py     # 回调函数
│   │   └── loggers.py      # 日志工具
│   ├── models/             # 模型库
│   │   ├── ranking/        # 排序模型
│   │   ├── match/          # 召回模型
│   │   ├── multi_task/     # 多任务模型
│   │   └── generative/     # 生成式模型
│   ├── datasets/           # 数据集处理
│   ├── metrics/            # 评估指标
│   ├── loss/               # 损失函数
│   ├── trainer/            # 训练器
│   └── utils/              # 工具函数
├── tutorials/              # 教程示例
├── tests/                  # 单元测试
└── docs/                   # 文档
```

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

### 安装

RecLib 支持两种安装方式：**UV（推荐）** 和传统的 pip 方式。

#### 方法一：使用 UV（推荐）

UV 是一个快速、现代的 Python 包管理器，提供更好的依赖解析和更快的安装速度。

```bash
# 1. 克隆仓库
git clone https://github.com/zerolovesea/RecLib.git
cd RecLib

# 2. 安装 UV（如果尚未安装）
pip install uv

# 3. 同步依赖并创建虚拟环境
python -m uv sync

# 这将会：
# - 创建一个 .venv 虚拟环境
# - 从 pyproject.toml 安装所有依赖
# - 以开发模式设置项目

# 4. 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows

# 5. （可选）安装开发依赖
python -m uv sync --extra dev
# 这包括 pytest、jupyter、matplotlib 和其他开发工具
```

#### 方法二：使用 pip

```bash
# 克隆仓库
git clone https://github.com/zerolovesea/RecLib.git
cd RecLib

# 安装依赖
pip install -r requirements.txt

# 单元测试（可选）
pip install -r test_requirements.txt
```

### 环境要求

- Python >= 3.10
- PyTorch >= 2.0.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0

### 10分钟教程

以下是使用 DeepFM 进行 CTR 预测的快速示例：

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from reclib.models.ranking.deepfm import DeepFM
from reclib.basic.features import DenseFeature, SparseFeature

# 1. 准备数据
df = pd.read_csv('your_data.csv')
target = 'label'

# 特征列
dense_cols = ['age', 'income']
sparse_cols = ['gender', 'city', 'category']

# 编码稀疏特征
for col in sparse_cols:
    lbe = LabelEncoder()
    df[col] = lbe.fit_transform(df[col].astype(str))

# 2. 定义特征
dense_features = [DenseFeature(name) for name in dense_cols]
sparse_features = [
    SparseFeature(name, vocab_size=df[name].nunique(), embedding_dim=16)
    for name in sparse_cols
]

# 3. 构建模型
model = DeepFM(
    dense_features=dense_features,
    sparse_features=sparse_features,
    mlp_params={"dims": [256, 128], "activation": "relu", "dropout": 0.3},
    targets=[target],
    optimizer="adam",
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    device='cuda',  # 或 'mps'（Mac）/ 'cpu'
    model_id="deepfm_exp"
)

# 4. 训练模型
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=2024)

model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=['auc', 'logloss'],
    epochs=10,
    batch_size=512,
    shuffle=True
)

# 5. 进行预测
predictions = model.predict(valid_df, batch_size=512)
```

### 使用 UV 运行

如果使用 UV 安装了 RecLib，可以直接运行脚本而无需激活虚拟环境：

```bash
# 运行 Python 脚本
python -m uv run python tutorials/example_deepfm.py

# 运行测试
python -m uv run pytest test/ -v

# 运行 Jupyter notebook
python -m uv run jupyter notebook

# 或者先激活虚拟环境
source .venv/bin/activate
python tutorials/example_deepfm.py
pytest test/ -v
```

### 更多示例

查看 `tutorials/` 目录获取更多示例：

- `example_deepfm.py` - 完整的 DeepFM 排序模型示例
- `example_ranking_din.py` - DIN 深度兴趣网络示例
- `example_match_dssm.py` - DSSM 召回模型示例
- `example_multitask.py` - 多任务学习示例
- `example_datasets.py` - 数据集下载和使用示例
- `example_dataloader.py` - 数据加载器使用示例
- `example_dataloader_integration.py` - 数据加载器集成示例
- `data_generator.py` - 测试数据生成工具

### 数据集使用

RecLib 提供常用推荐数据集的统一接口：

```python
from reclib.datasets import get_dataset, list_datasets

# 查看所有可用数据集
print(list_datasets())
# ['movielens-100k', 'movielens-1m', 'movielens-25m', 
#  'criteo', 'amazon-books', 'amazon-electronics', 'avazu']

# 下载并加载 MovieLens 100K 数据集
dataset = get_dataset("movielens-100k", root="./data", download=True)
dataset.info()  # 查看数据集信息

# 加载数据(包含用户和电影特征)
df = dataset.load(include_features=True)
print(df.head())

# 数据预处理
from reclib.datasets.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df = preprocessor.create_binary_labels(df, rating_col='rating', threshold=4.0)
df = preprocessor.encode_categorical(df, ['user_id', 'item_id', 'gender'])
train_df, valid_df, test_df = preprocessor.split_by_ratio(df, ratios=(0.7, 0.15, 0.15))
```

详细文档请查看 [reclib/datasets/README.md](reclib/datasets/README.md)

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

## 核心功能

### 特征定义

RecLib 提供三种类型的特征：

```python
from reclib.basic.features import DenseFeature, SparseFeature, SequenceFeature

# 稠密特征 - 用于连续数值
dense_feat = DenseFeature(
    feature_name='age',
    feature_dim=1
)

# 稀疏特征 - 用于类别变量
sparse_feat = SparseFeature(
    feature_name='category',
    vocab_size=100,
    embedding_dim=16,
    embedding_name='category_emb'  # 可选，用于参数共享
)

# 序列特征 - 用于用户行为序列
sequence_feat = SequenceFeature(
    feature_name='click_history',
    vocab_size=1000,
    embedding_dim=32,
    pooling='mean',  # 'mean', 'sum', 'max'
    max_length=50
)
```

### 自定义模型

继承 `BaseModel` 创建新模型：

```python
from reclib.basic.model import BaseModel
import torch.nn as nn

class YourModel(BaseModel):
    @property
    def model_name(self):
        return "YourModel"
    
    @property  
    def task_type(self):
        return "binary"  # 或 "regression"、"multi_class"
    
    def __init__(self, dense_features, sparse_features, **kwargs):
        super().__init__(
            dense_features=dense_features,
            sparse_features=sparse_features,
            **kwargs
        )
        
        # 定义模型结构
        self.dnn = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        # 实现前向传播
        dense_input = x['dense']
        sparse_input = self.embedding(x['sparse'])
        
        combined = torch.cat([dense_input, sparse_input], dim=-1)
        output = self.dnn(combined)
        return output
```

### 训练与评估

```python
# 训练
model.fit(
    train_data=train_df,
    valid_data=valid_df,
    metrics=['auc', 'logloss'],  # 多指标评估
    epochs=20,
    batch_size=1024,
    shuffle=True,
    verbose=1,  # 日志详细程度
    early_stop_patience=5  # 早停
)

# 预测
predictions = model.predict(test_df, batch_size=1024)

# 评估
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_df['label'], predictions)
print(f"Test AUC: {auc:.6f}")
```

---

## 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 Python 代码风格
- 为新功能添加单元测试
- 更新相关文档

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
