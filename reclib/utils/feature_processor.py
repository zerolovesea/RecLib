import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import pickle
from collections import defaultdict, Counter
import warnings
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
warnings.filterwarnings('ignore')

# todo 编码等过程均在pd.Dataframe上进行
class FeatureMap:
    """特征映射类，管理所有特征的信息"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir # 数据存储路径（如"./data/movie"）
        self.features = {} # 存储所有特征的详细信息（键：特征名，值：特征元信息字典）。
        # self.feature_type = {}
        # self.feature_dim = {}
        # self.embedding_dim = {}
        self.sequence_features = {} # 存储序列特征的最大长度和池化方式（键：特征名，值：{"max_length": ..., "pooling": ...}）。
        self.numerical_features = [] # 数值特征名称列表（如[“age”, “income”]）
        self.categorical_features = [] # 类别特征名称列表（如[“gender”, “occupation”]）
        self.condition_features = [] # 条件特征名称列表（如[“context_location”]，用于模型条件输入）
        
    def add_feature(self, feature_name: str, feature_type: str, 
                   feature_dim: int | None = None, embedding_dim: int = 10,
                   is_sequence: bool = False, max_length: int = 50,
                   pooling: str = "mean", is_condition: bool = False):
        """添加特征信息
        将特征元信息存入self.features字典（统一管理所有特征细节）。
        根据feature_type将特征名添加到numerical_features或categorical_features列表。
        若is_sequence=True，将特征名及max_length/pooling存入sequence_features字典。
        若is_condition=True，将特征名添加到condition_features列表。
        """
        self.features[feature_name] = {
            "type": feature_type, # 特征类型："numerical"（数值型）或"categorical"（类别型）
            "dim": feature_dim, # 特征维度（类别特征：类别总数；数值特征：可忽略，默认为None）
            "embedding_dim": embedding_dim, # 类别特征嵌入维度
            "is_sequence": is_sequence, # 是否为序列特征（如用户点击历史序列）
            "max_length": max_length, # 序列特征的最大长度（超过则截断，不足则填充）
            "pooling": pooling, # 序列特征的池化方式（如"mean"平均池化、"sum"求和池化、"max"最大池化）
            "is_condition": is_condition # 是否为条件特征（用于模型的条件输入，如上下文特征）
        }
        
        if feature_type == "categorical":
            self.categorical_features.append(feature_name)
        elif feature_type == "numerical":
            self.numerical_features.append(feature_name)
        
        if is_sequence:
            self.sequence_features[feature_name] = {
                "max_length": max_length,
                "pooling": pooling
            }
            
        if is_condition:
            self.condition_features.append(feature_name)
    
    def get_feature_dim(self, feature_name: str) -> int:
        """获取特征维度"""
        return self.features[feature_name]["dim"]
    
    def get_embedding_dim(self, feature_name: str) -> int:
        """获取嵌入维度"""
        return self.features[feature_name]["embedding_dim"]
    
    def is_sequence_feature(self, feature_name: str) -> bool:
        """判断是否为序列特征"""
        return self.features[feature_name]["is_sequence"]
    
    def sum_emb_out_dim(self) -> int:
        """计算所有特征嵌入后的总维度"""
        total_dim = 0
        for feature_name in self.features:
            if self.is_sequence_feature(feature_name):
                # 序列特征经过池化后的维度
                total_dim += self.get_embedding_dim(feature_name)
            else:
                # 普通特征的嵌入维度
                total_dim += self.get_embedding_dim(feature_name)
        return total_dim
    
    def save(self, filepath: str):
        """保存特征映射"""
        with open(filepath, 'w') as f:
            json.dump(self.features, f, indent=2)
    
    def load(self, filepath: str, params: dict | None = None):
        """加载特征映射"""
        with open(filepath, 'r') as f:
            self.features = json.load(f)
        
        # 重建索引
        self.categorical_features = []
        self.numerical_features = []
        self.sequence_features = {}
        self.condition_features = []
        
        for feature_name, feature_info in self.features.items():
            if feature_info["type"] == "categorical":
                self.categorical_features.append(feature_name)
            elif feature_info["type"] == "numerical":
                self.numerical_features.append(feature_name)
            
            if feature_info["is_sequence"]:
                self.sequence_features[feature_name] = {
                    "max_length": feature_info["max_length"],
                    "pooling": feature_info["pooling"]
                }
            
            if feature_info["is_condition"]:
                self.condition_features.append(feature_name)


class SequenceVocabulary:
    """生产级序列词汇表"""
    
    def __init__(self,
                 min_freq: int = 2,
                 max_size: int = 50000,
                 unk_token: str = '<UNK>',
                 pad_token: str = '<PAD>',
                 bos_token: str = '<BOS>',
                 eos_token: str = '<EOS>',
                 mask_token: str = '<MASK>'):
        
        # 特殊token管理
        self.special_tokens = {
            pad_token: 0, unk_token: 1,
            bos_token: 2, eos_token: 3, mask_token: 4
        }
        
        # 词汇表统计
        self.token_counts = Counter() # Counter 对象，统计普通 token 的出现频率
        self.vocab = {} # 普通 token→索引的映射表（不包含特殊标记）
        self.reverse_vocab = {} # 普通 token→索引的映射表（不包含特殊标记）
        
        # 配置
        self.min_freq = min_freq # 	最小词频阈值：仅频率 ≥ 该值的 token 才会被加入词汇表
        self.max_size = max_size # 	词汇表总大小上限（包含特殊标记），默认 50000
        
        # 特殊token名称
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
    
    def build_vocab(self, sequences: List[List[str]]):
        """
        从序列列表构建词汇表
        统计原始文本序列中所有普通 token 的频率；
        按频率从高到低排序，过滤低频词（频率 < min_freq），并限制总词汇表大小 ≤ max_size；
        构建普通 token 与索引的映射（vocab 和 reverse_vocab）
        """
        # 统计词频
        for sequence in sequences:
            for token in sequence:
                self.token_counts[token] += 1
        
        # 按频率排序，过滤低频词
        sorted_tokens = sorted(
            self.token_counts.items(),
            key=lambda x: x[1],
            reverse=True  # 降序（高频在前）
        )
        
        # 构建词汇表，考虑大小限制
        idx = len(self.special_tokens)
        for token, count in sorted_tokens:
            if count >= self.min_freq and idx < self.max_size:
                self.vocab[token] = idx
                self.reverse_vocab[idx] = token
                idx += 1
            else:
                break  # 达到大小限制
    
    def encode(self, tokens: List[str]) -> List[int]:
        """编码token序列，将输入的 token 列表转换为索引列表，未收录的 token 用 <UNK> 的索引（1）替代。"""
        return [
            self.vocab.get(token, self.special_tokens[self.unk_token])
            for token in tokens
        ]
    
    def decode(self, indices: List[int]) -> List[str]:
        """解码索引序列"""
        return [
            self.reverse_vocab.get(idx, self.unk_token)
            for idx in indices
        ]
    
    def __len__(self):
        return len(self.vocab) + len(self.special_tokens)


class SequencePreprocessor:
    """序列预处理管道, 用于将输入序列（如字符串、列表或数组）通过一系列标准化操作（如小写转换、标点移除、长度过滤等）转换为干净的 token 列表，适用于文本处理、特征提取等场景。"""
    
    def __init__(self,
                 lowercase: bool = True,
                 remove_punct: bool = True,
                 min_token_length: int = 1,
                 split_by: str = ','):
        
        self.lowercase = lowercase # 是否将 token 转为小写（默认 True）
        self.remove_punct = remove_punct # 是否移除非字母/数字字符（如标点、空格，默认 True）
        self.min_token_length = min_token_length # token 最小保留长度（默认 1，短于该长度的 token 会被过滤）
        self.split_by = split_by # 字符串序列的分割符（默认 ','，用于将字符串切分为原始 token）
    
    def normalize_token(self, token: str) -> str:
        """标准化单个token"""
        if self.lowercase:
            token = token.lower()
        if self.remove_punct:
            token = ''.join(c for c in token if c.isalnum())
        if len(token) < self.min_token_length:
            return ''
        return token.strip()
    
    def preprocess_sequence(self, sequence) -> List[str]:
        """预处理单个序列"""
        tokens = []
        
        if isinstance(sequence, str):
            # 字符串序列
            raw_tokens = sequence.split(self.split_by)
        elif isinstance(sequence, (list, np.ndarray)):
            # 列表/数组序列
            raw_tokens = [str(x) for x in sequence]
        else:
            return []
        
        # 标准化每个token
        for token in raw_tokens:
            normalized = self.normalize_token(token)
            if normalized:
                tokens.append(normalized)
        
        return tokens


class SequenceFeatureProcessor:
    """完整的序列特征处理器"""
    
    def __init__(self,
                 feature_name: str,
                 max_length: int = 100,
                 pooling: str = 'mean',
                 vocab_config: dict | None = None,
                 preprocessor_config: dict | None = None,
                 quiet: bool = True):
        
        self.feature_name = feature_name
        self.max_length = max_length
        self.pooling = pooling
        
        # 初始化组件
        vocab_config = vocab_config or {}
        self.vocab = SequenceVocabulary(**vocab_config)
        self.preprocessor = SequencePreprocessor(**(preprocessor_config or {}))
        
        # 状态
        self.is_fitted = False
        self.quiet = quiet  # True 时关闭逐样本 tqdm
    
    def fit(self, sequences: pd.Series):
        """拟合序列特征处理器"""
        print(f"🔧 Fitting sequence feature: {self.feature_name}")
        
        # 预处理所有序列
        processed_sequences = []
        iterator = sequences if self.quiet else tqdm(sequences, desc=f"Preprocessing {self.feature_name}")
        for seq in iterator:
            tokens = self.preprocessor.preprocess_sequence(seq)
            processed_sequences.append(tokens)
        
        # 构建词汇表
        self.vocab.build_vocab(processed_sequences)
        self.is_fitted = True
        
        print(f"✅ Vocabulary built: {len(self.vocab.vocab)} tokens")
        return self
    
    def transform(self, sequences: pd.Series) -> torch.Tensor:
        """转换序列特征"""
        if not self.is_fitted:
            raise ValueError("Sequence processor must be fitted before transform")
        
        batch_sequences = []
        
        iterator = sequences if self.quiet else tqdm(sequences, desc=f"Transforming {self.feature_name}")
        for seq in iterator:
            # 预处理
            tokens = self.preprocessor.preprocess_sequence(seq)
            
            # 编码
            encoded = self.vocab.encode(tokens)
            
            # 截断
            if len(encoded) > self.max_length:
                encoded = encoded[:self.max_length]
            
            batch_sequences.append(encoded)
        
        # NOTE:
        # 为保证可堆叠，这里对于单样本场景固定输出长度 = self.max_length 的 1D 向量；
        # 对于批量场景再进行 batch 级 padding，并同样裁剪/填充到 self.max_length，

        if len(batch_sequences) == 1:
            seq = batch_sequences[0]
            pad_id = self.vocab.special_tokens[self.vocab.pad_token]
            if len(seq) < self.max_length:
                seq = seq + [pad_id] * (self.max_length - len(seq))
            # 如果已经等于或超过 max_length，上面截断已处理
            return torch.LongTensor(seq)  # shape: (max_length,)

        # 多样本：先动态找 batch 内最大长度，再裁剪到 self.max_length，再统一 pad
        batch_max_len = min(self.max_length, max(len(s) for s in batch_sequences) if batch_sequences else self.max_length)
        pad_id = self.vocab.special_tokens[self.vocab.pad_token]
        padded = []
        for seq in batch_sequences:
            if len(seq) < batch_max_len:
                seq = seq + [pad_id] * (batch_max_len - len(seq))
            else:
                seq = seq[:batch_max_len]
            padded.append(seq)
        tensor = torch.LongTensor(padded)  # (batch, L)
        if tensor.shape[0] == 1:
            return tensor[0]
        return tensor
    
    def _dynamic_padding(self, sequences: List[List[int]]) -> torch.Tensor:
        """动态padding到batch内最大长度"""
        # 将输入的序列列表（List[List[int]]）处理为长度一致的序列（填充或截断），确保批次内所有序列长度等于该批次的最大序列长度（至少为1），最终转换为 torch.LongTensor 返回。
        # 保证至少长度为1，避免出现 shape (1,0) 造成后续 stack 报错
        batch_max_len = max(1, max(len(seq) for seq in sequences) if sequences else 1)
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) < batch_max_len:
                pad_id = self.vocab.special_tokens[self.vocab.pad_token]
                padded = seq + [pad_id] * (batch_max_len - len(seq))
            else:
                padded = seq[:batch_max_len]
            padded_sequences.append(padded)
        
        return torch.LongTensor(padded_sequences)
    
    def save(self, filepath: str):
        """保存序列特征处理器"""
        # 将序列特征处理器的所有状态（词汇表、预处理器配置、拟合状态等）保存到文件，以便后续加载复用
        processor_data = {
            'feature_name': self.feature_name,
            'max_length': self.max_length,
            'pooling': self.pooling,
            'vocab_state': {
                'token_counts': dict(self.vocab.token_counts),
                'vocab': dict(self.vocab.vocab),
                'reverse_vocab': dict(self.vocab.reverse_vocab),
                'min_freq': self.vocab.min_freq,
                'max_size': self.vocab.max_size,
                'special_tokens': self.vocab.special_tokens,
                'unk_token': self.vocab.unk_token,
                'pad_token': self.vocab.pad_token,
                'bos_token': self.vocab.bos_token,
                'eos_token': self.vocab.eos_token,
                'mask_token': self.vocab.mask_token
            },
            'preprocessor_config': self.preprocessor.__dict__,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(processor_data, f)
    
    def load(self, filepath: str):
        """加载序列特征处理器"""
        with open(filepath, 'rb') as f:
            processor_data = pickle.load(f)
        
        self.feature_name = processor_data['feature_name']
        self.max_length = processor_data['max_length']
        self.pooling = processor_data['pooling']
        self.is_fitted = processor_data['is_fitted']
        
        # 重建词汇表状态
        vocab_state = processor_data['vocab_state']
        self.vocab.token_counts = Counter(vocab_state['token_counts'])
        self.vocab.vocab = vocab_state['vocab']
        self.vocab.reverse_vocab = vocab_state['reverse_vocab']
        self.vocab.min_freq = vocab_state['min_freq']
        self.vocab.max_size = vocab_state['max_size']
        self.vocab.special_tokens = vocab_state['special_tokens']
        self.vocab.unk_token = vocab_state['unk_token']
        self.vocab.pad_token = vocab_state['pad_token']
        self.vocab.bos_token = vocab_state['bos_token']
        self.vocab.eos_token = vocab_state['eos_token']
        self.vocab.mask_token = vocab_state['mask_token']
        
        # 重建预处理器
        self.preprocessor.__dict__.update(processor_data['preprocessor_config'])


class FeatureProcessor:
    """重构后的特征处理器"""
    # FeatureProcessor 是一个统一的特征处理器，用于自动化处理结构化数据中的 分类特征、数值特征 和 序列特征。
    # 核心功能是通过 fit 方法利用训练数据拟合特征处理器（编码器、缩放器等），并生成可直接用于模型的 vocabs（类别/序列 token 映射表）和 scalers（数值特征统计量），
    # 为后续特征转换（如编码、归一化、序列向量化）提供基础。
    
    def __init__(self, feature_map: FeatureMap):
        self.feature_map = feature_map
        
        # 分类型的处理器
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        
        # 序列特征处理器
        self.sequence_processors = {}
        
        # 兼容旧训练脚本引用：vocabs / scalers
        # vocabs: {feature_name: {token: index}}
        # scalers: {feature_name: {mean: float, std: float}}
        self.vocabs = {}
        self.scalers = {}
        
        # 状态
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame):
        """使用训练数据拟合所有特征处理器"""
        print("🔧 Fitting all feature processors (unified progress)...")
        fit_tasks: List[tuple[str, str]] = []
        for f in self.feature_map.categorical_features:
            if not self.feature_map.is_sequence_feature(f):
                fit_tasks.append(("cat", f))
        for f in self.feature_map.numerical_features:
            fit_tasks.append(("num", f))
        for f in self.feature_map.sequence_features.keys():
            fit_tasks.append(("seq", f))
        pbar = tqdm(total=len(fit_tasks), desc="Fitting features", unit="feat")

        def _prepare_cat(col: pd.Series) -> pd.Series:
            if getattr(col.dtype, 'name', '').startswith('category'):
                if '<UNK>' not in col.cat.categories:
                    col = col.cat.add_categories(['<UNK>'])
                col = col.fillna('<UNK>')
            else:
                col = col.fillna('<UNK>')
            return col

        # Categorical
        for ftype, feat in [t for t in fit_tasks if t[0]=="cat"]:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)
            series = _prepare_cat(data[feat].copy())
            enc.fit(pd.DataFrame({feat: series}))
            self.categorical_encoders[feat] = enc
            raw_cats = enc.categories_[0]
            cats_np = np.asarray(raw_cats, dtype=object)
            cats_flat = cats_np.tolist()
            if not isinstance(cats_flat, list):
                cats_flat = [cats_flat]
            # 确保二维嵌套场景（极少见）展开为一维
            if cats_flat and isinstance(cats_flat[0], list):
                cats_flat = cats_flat[0]
            # 确保可映射到 python 原生类型，避免 numpy scalar 带来的可哈希性问题
            normalized_cats: List[Any] = []
            for cat in cats_flat:
                if hasattr(cat, 'item'):
                    try:
                        normalized_cats.append(cat.item())
                        continue
                    except Exception:
                        pass
                normalized_cats.append(cat)
            cats_list = normalized_cats
            self.feature_map.features[feat]['dim'] = len(cats_list)
            mapping = {cat: idx for idx, cat in enumerate(cats_list)}
            # 为未知值保留安全映射：如不存在 <UNK>，则将其映射到 0 号（不会扩增维度）
            if '<UNK>' not in mapping and len(cats_list) > 0:
                mapping['<UNK>'] = 0
            self.vocabs[feat] = mapping
            pbar.set_postfix_str(f"cat:{feat}")
            pbar.update(1)

        # Numerical
        for ftype, feat in [t for t in fit_tasks if t[0]=="num"]:
            scaler = StandardScaler()
            scaler.fit(data[[feat]].fillna(0))
            self.numerical_scalers[feat] = scaler
            self.scalers[feat] = {
                'mean': float(getattr(scaler,'mean_', [0.0])[0]),
                'std': float(getattr(scaler,'scale_', [1.0])[0])
            }
            pbar.set_postfix_str(f"num:{feat}")
            pbar.update(1)

        # Sequence
        for ftype, feat in [t for t in fit_tasks if t[0]=="seq"]:
            cfg = self.feature_map.sequence_features[feat]
            sp = SequenceFeatureProcessor(feature_name=feat, max_length=cfg['max_length'], pooling=cfg['pooling'])
            sp.fit(data[feat])
            self.sequence_processors[feat] = sp
            self.feature_map.features[feat]['dim'] = len(sp.vocab)
            # build combined vocab mapping (special + normal tokens)
            vocab_map = {tok: idx for tok, idx in sp.vocab.special_tokens.items()}
            vocab_map.update(sp.vocab.vocab)
            self.vocabs[feat] = vocab_map
            pbar.set_postfix_str(f"seq:{feat}")
            pbar.update(1)

        pbar.close()
        self.is_fitted = True
        print(f"✅ All feature processors fitted! (total: {len(fit_tasks)} features)")
    
    def transform(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """转换数据为特征张量"""
        if not self.is_fitted:
            raise ValueError("Feature processor must be fitted before transform")
        feats: Dict[str, torch.Tensor] = {}

        # Categorical
        for f in self.feature_map.categorical_features:
            if self.feature_map.is_sequence_feature(f):
                continue
            if f not in data.columns:
                continue
            series = data[f].copy()
            if getattr(series.dtype, 'name', '').startswith('category'):
                if '<UNK>' not in series.cat.categories:
                    series = series.cat.add_categories(['<UNK>'])
                series = series.fillna('<UNK>')
            else:
                series = series.fillna('<UNK>')
            enc = self.categorical_encoders[f].transform(pd.DataFrame({f: series}))
            flat = enc.flatten()
            if flat.size == 0:
                flat = np.array([-1])
            feats[f] = torch.LongTensor(flat)

        # Numerical
        for f in self.feature_map.numerical_features:
            if f not in data.columns:
                continue
            scaled = self.numerical_scalers[f].transform(data[[f]].fillna(0))
            flat = scaled.flatten()
            if flat.size == 0:
                flat = np.array([0.0], dtype=np.float32)
            feats[f] = torch.FloatTensor(flat)

        # Sequence
        for f in self.feature_map.sequence_features:
            if f in data.columns:
                feats[f] = self.sequence_processors[f].transform(data[f])

        return feats
    
    def fit_transform(self, data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """拟合并转换数据"""
        self.fit(data)
        return self.transform(data)
    
    def save(self, filepath: str):
        """保存特征处理器"""
        processor_data = {
            'categorical_encoders': self.categorical_encoders,
            'numerical_scalers': self.numerical_scalers,
            'feature_map_features': self.feature_map.features,
            'sequence_processor_configs': {},
            'is_fitted': self.is_fitted
        }
        
        # 保存序列处理器配置
        for feature_name, processor in self.sequence_processors.items():
            config_path = f"{filepath}.seq_{feature_name}"
            processor.save(config_path)
            processor_data['sequence_processor_configs'][feature_name] = config_path
        
        with open(filepath, 'wb') as f:
            pickle.dump(processor_data, f)
    
    def load(self, filepath: str):
        """加载特征处理器"""
        with open(filepath, 'rb') as f:
            processor_data = pickle.load(f)
        
        self.categorical_encoders = processor_data['categorical_encoders']
        self.numerical_scalers = processor_data['numerical_scalers']
        self.feature_map.features = processor_data['feature_map_features']
        self.is_fitted = processor_data['is_fitted']
        
        # 加载序列处理器
        self.sequence_processors = {}
        for feature_name, config_path in processor_data['sequence_processor_configs'].items():
            processor = SequenceFeatureProcessor(feature_name=feature_name)
            processor.load(config_path)
            self.sequence_processors[feature_name] = processor


if __name__ == "__main__":
    # 1. 准备样本数据（用于构建词汇表）
    # 假设我们有一些分词后的文本序列（模拟真实语料）
    import os
    import json
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .appName("OptimizedGBTExample") \
        .config("spark.driver.memory", "32g") \
        .config("spark.executor.memory", "32g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    pos_sample_path = rf"E:\iserver\model_iserver\dataset\YXH_reasoning_set_M11\train_all.parquet"
    pos_sample_files = [os.path.join(pos_sample_path, f) for f in os.listdir(pos_sample_path) if f.endswith('.parquet')]
    df = spark.read.parquet(*pos_sample_files).limit(60000)

    # 读取JSON文件并恢复feature_map
    with open("feature.json", "r", encoding="utf-8") as f:
        loaded_feature_map = json.load(f)
    # # 1. 初始化特征映射
    # feature_map = FeatureMap(1,2)
    #
    # # 2. 定义特征类型
    feature_map = FeatureMap(5, 6)
    for i in loaded_feature_map['categorical']:
        feature_map.add_feature(i, 'categorical')
    for i in loaded_feature_map['numerical']:
        feature_map.add_feature(i, 'numerical')
    for i in loaded_feature_map['sequence']:
        feature_map.add_feature(i, 'sequence', is_sequence=True)

    # 创建FeatureProcessor实例
    fp = FeatureProcessor(feature_map)

    # 拟合数据（构建编码器、标准化器、词汇表）
    data = df.toPandas()
    fp.fit(data)
    # #
    # # # 转换数据（输出PyTorch张量）
    fp.save("./transfrom_pkl/feature_transform.pkl")
    # transformed_data = fp.transform(data)

    fp.load("./transfrom_pkl/feature_transform.pkl")
    transformed_data = fp.transform(data)
    print("转换后的特征（PyTorch张量）：")
    for feat_name, tensor in transformed_data.items():
        print(f"\n特征名: {feat_name}")
        print(f"形状: {tensor.shape}")
        print(f"前3个样本数据:\n{tensor[:3]}")
