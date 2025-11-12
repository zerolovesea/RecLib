import pandas as pd
import numpy as np
import pickle
import json
import hashlib
import os
from typing import Dict, List, Union, Optional, Literal, Any
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler, 
    MaxAbsScaler,
    LabelEncoder
)
import logging
import warnings

from reclib.basic.loggers import setup_logger, colorize


class DataProcessor:
    """DataProcessor for data preprocessing including numeric, sparse, sequence features and target processing.
    
       DataProcesso类用于数据预处理，注册不同类型的特征到类实例后，通过fit方法学习数据的统计信息和编码器，
       然后通过transform方法将原始数据转换为dict/dataframe，支持直接传入RecLib BaseModel的fit和predict方法中使用，也支持传入RecDataLoader。

    Examples:
        >>> processor = DataProcessor()
        >>> processor.add_numeric_feature('age', scaler='standard')
        >>> processor.add_sparse_feature('user_id', encode_method='hash', hash_size=10000)
        >>> processor.add_sequence_feature('item_history', encode_method='label', max_len=50, pad_value=0)
        >>> processor.add_target('label', target_type='binary')
        >>> 
        >>> # Fit and transform data
        >>> processor.fit(train_df)
        >>> processed_data = processor.transform(test_df)  # Returns dict of numpy arrays
        >>> 
        >>> # Save and load processor
        >>> processor.save('processor.pkl')
        >>> loaded_processor = DataProcessor.load('processor.pkl')
        >>> 
        >>> # Get vocabulary sizes for embedding layers
        >>> vocab_sizes = processor.get_vocab_sizes()
    """
    def __init__(self):
        self.numeric_features: Dict[str, Dict[str, Any]] = {}
        self.sparse_features: Dict[str, Dict[str, Any]] = {}
        self.sequence_features: Dict[str, Dict[str, Any]] = {}
        self.target_features: Dict[str, Dict[str, Any]] = {}
        
        self.is_fitted = False
        
        self.scalers: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoders: Dict[str, Dict[str, int]] = {}
        
        # Initialize logger if not already initialized
        self._logger_initialized = False
        if not logging.getLogger().hasHandlers():
            setup_logger()
            self._logger_initialized = True
        
    def add_numeric_feature(
        self, 
        name: str, 
        scaler: Optional[Literal['standard', 'minmax', 'robust', 'maxabs', 'log', 'none']] = 'standard',
        fill_na: Optional[float] = None
    ):
        self.numeric_features[name] = {
            'scaler': scaler,
            'fill_na': fill_na
        }
        
    def add_sparse_feature(
        self, 
        name: str, 
        encode_method: Literal['hash', 'label'] = 'label',
        hash_size: Optional[int] = None,
        fill_na: str = '<UNK>'
    ):
        if encode_method == 'hash' and hash_size is None:
            raise ValueError("hash_size must be specified when encode_method='hash'")
            
        self.sparse_features[name] = {
            'encode_method': encode_method,
            'hash_size': hash_size,
            'fill_na': fill_na
        }
        
    def add_sequence_feature(
        self, 
        name: str,
        encode_method: Literal['hash', 'label'] = 'label',
        hash_size: Optional[int] = None,
        max_len: Optional[int] = 50,
        pad_value: int = 0,
        truncate: Literal['pre', 'post'] = 'pre',           # pre: keep last max_len items, post: keep first max_len items
        separator: str = ','
    ):

        if encode_method == 'hash' and hash_size is None:
            raise ValueError("hash_size must be specified when encode_method='hash'")
            
        self.sequence_features[name] = {
            'encode_method': encode_method,
            'hash_size': hash_size,
            'max_len': max_len,
            'pad_value': pad_value,
            'truncate': truncate,
            'separator': separator
        }
        
    def add_target(
        self, 
        name: str,                                                                # example: 'click'
        target_type: Literal['binary', 'multiclass', 'regression'] = 'binary',
        label_map: Optional[Dict[str, int]] = None                                # example: {'click': 1, 'no_click': 0}
    ):
        self.target_features[name] = {
            'target_type': target_type,
            'label_map': label_map
        }
        
    def _hash_string(self, s: str, hash_size: int) -> int:
        return int(hashlib.md5(str(s).encode()).hexdigest(), 16) % hash_size
        
    def _process_numeric_feature_fit(self, data: pd.Series, config: Dict[str, Any]):

        name = str(data.name)
        scaler_type = config['scaler']
        fill_na = config['fill_na']
        
        if data.isna().any():
            if fill_na is None:
                # Default use mean value to fill missing values for numeric features
                fill_na = data.mean()
            config['fill_na_value'] = fill_na
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'maxabs':
            scaler = MaxAbsScaler()
        elif scaler_type == 'log':
            scaler = None  
        elif scaler_type == 'none':
            scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        if scaler is not None and scaler_type != 'log':
            filled_data = data.fillna(config.get('fill_na_value', 0))
            values = np.array(filled_data.values, dtype=np.float64).reshape(-1, 1)
            scaler.fit(values)
            self.scalers[name] = scaler
            
    def _process_numeric_feature_transform(
        self, 
        data: pd.Series, 
        config: Dict[str, Any]
    ) -> np.ndarray:
        logger = logging.getLogger()
        
        name = str(data.name)
        scaler_type = config['scaler']
        fill_na_value = config.get('fill_na_value', 0)

        filled_data = data.fillna(fill_na_value)
        values = np.array(filled_data.values, dtype=np.float64)

        if scaler_type == 'log':
            result = np.log1p(np.maximum(values, 0))
        elif scaler_type == 'none':
            result = values
        else:
            scaler = self.scalers.get(name)
            if scaler is None:
                logger.warning(f"Scaler for {name} not fitted, returning original values")
                result = values
            else:
                result = scaler.transform(values.reshape(-1, 1)).ravel()
        
        return result
        
    def _process_sparse_feature_fit(self, data: pd.Series, config: Dict[str, Any]):

        name = str(data.name)
        encode_method = config['encode_method']
        fill_na = config['fill_na'] # <UNK>
        
        filled_data = data.fillna(fill_na).astype(str)
        
        if encode_method == 'label':
            le = LabelEncoder()
            le.fit(filled_data)
            self.label_encoders[name] = le
            config['vocab_size'] = len(le.classes_)
        elif encode_method == 'hash':
            config['vocab_size'] = config['hash_size']
            
    def _process_sparse_feature_transform(
        self, 
        data: pd.Series, 
        config: Dict[str, Any]
    ) -> np.ndarray:
        
        name = str(data.name)
        encode_method = config['encode_method']
        fill_na = config['fill_na']
        
        filled_data = data.fillna(fill_na).astype(str)
        
        if encode_method == 'label':
            le = self.label_encoders.get(name)
            if le is None:
                raise ValueError(f"LabelEncoder for {name} not fitted")

            result = []
            for val in filled_data:
                if val in le.classes_:
                    encoded = le.transform([val])
                    result.append(int(encoded[0]))
                else:
                    result.append(0)
            return np.array(result, dtype=np.int64)
            
        elif encode_method == 'hash':
            hash_size = config['hash_size']
            return np.array([self._hash_string(val, hash_size) for val in filled_data], dtype=np.int64)
        
        return np.array([], dtype=np.int64)
            
    def _process_sequence_feature_fit(self, data: pd.Series, config: Dict[str, Any]):

        name = str(data.name)
        encode_method = config['encode_method']
        separator = config['separator']
        
        if encode_method == 'label':
            all_tokens = set()
            for seq in data:
                # Skip None, np.nan, and empty strings
                if seq is None:
                    continue
                if isinstance(seq, (float, np.floating)) and np.isnan(seq):
                    continue
                if isinstance(seq, str) and seq.strip() == '':
                    continue
                
                if isinstance(seq, str):
                    tokens = seq.split(separator)
                elif isinstance(seq, (list, tuple)):
                    tokens = [str(t) for t in seq]
                elif isinstance(seq, np.ndarray):
                    tokens = [str(t) for t in seq.tolist()]
                else:
                    continue
                
                all_tokens.update(tokens)
            
            if len(all_tokens) == 0:
                all_tokens.add('<PAD>')
            
            le = LabelEncoder()
            le.fit(list(all_tokens))
            self.label_encoders[name] = le
            config['vocab_size'] = len(le.classes_)
        elif encode_method == 'hash':
            config['vocab_size'] = config['hash_size']
            
    def _process_sequence_feature_transform(
        self, 
        data: pd.Series, 
        config: Dict[str, Any]
    ) -> np.ndarray:
        name = str(data.name)
        encode_method = config['encode_method']
        max_len = config['max_len']
        pad_value = config['pad_value']
        truncate = config['truncate']
        separator = config['separator']
        
        result = []
        for seq in data:
            tokens = []
            
            if seq is None:
                tokens = []
            elif isinstance(seq, (float, np.floating)) and np.isnan(seq):
                tokens = []
            elif isinstance(seq, str):
                if seq.strip() == '':
                    tokens = []
                else:
                    tokens = seq.split(separator)
            elif isinstance(seq, (list, tuple)):
                tokens = [str(t) for t in seq]
            elif isinstance(seq, np.ndarray):
                tokens = [str(t) for t in seq.tolist()]
            else:
                tokens = []
            
            if encode_method == 'label':
                le = self.label_encoders.get(name)
                if le is None:
                    raise ValueError(f"LabelEncoder for {name} not fitted")
                
                encoded = []
                for token in tokens:
                    token_str = str(token).strip()
                    if token_str and token_str in le.classes_:
                        encoded_val = le.transform([token_str])
                        encoded.append(int(encoded_val[0]))
                    else:
                        encoded.append(0)  # UNK
            elif encode_method == 'hash':
                hash_size = config['hash_size']
                encoded = [self._hash_string(str(token), hash_size) for token in tokens if str(token).strip()]
            else:
                encoded = []
            
            if len(encoded) > max_len:
                if truncate == 'pre': # keep last max_len items
                    encoded = encoded[-max_len:]
                else:                 # keep first max_len items
                    encoded = encoded[:max_len]
            elif len(encoded) < max_len:
                padding = [pad_value] * (max_len - len(encoded))
                encoded = encoded + padding
            
            result.append(encoded)
        
        return np.array(result, dtype=np.int64)
        
    def _process_target_fit(self, data: pd.Series, config: Dict[str, Any]):
        name = str(data.name)
        target_type = config['target_type']
        label_map = config['label_map']
        
        if target_type in ['binary', 'multiclass']:
            if label_map is None:
                unique_values = data.dropna().unique()
                sorted_values = sorted(unique_values)
                
                try:
                    int_values = [int(v) for v in sorted_values]
                    if int_values == list(range(len(int_values))):
                        label_map = {str(val): int(val) for val in sorted_values}
                    else:
                        label_map = {str(val): idx for idx, val in enumerate(sorted_values)}
                except (ValueError, TypeError):
                    label_map = {str(val): idx for idx, val in enumerate(sorted_values)}
                
                config['label_map'] = label_map    
            
            self.target_encoders[name] = label_map
            
    def _process_target_transform(
        self, 
        data: pd.Series, 
        config: Dict[str, Any]
    ) -> np.ndarray:
        logger = logging.getLogger()
        
        name = str(data.name)
        target_type = config['target_type']
        
        if target_type == 'regression':
            values = np.array(data.values, dtype=np.float32)
            return values
        else:
            label_map = self.target_encoders.get(name)
            if label_map is None:
                raise ValueError(f"Target encoder for {name} not fitted")
            
            result = []
            for val in data:
                str_val = str(val)
                if str_val in label_map:
                    result.append(label_map[str_val])
                else:
                    logger.warning(f"Unknown target value: {val}, mapping to 0")
                    result.append(0)
            
            return np.array(result, dtype=np.int64 if target_type == 'multiclass' else np.float32)
    
    # fit is nothing but registering the statistics from data so that we can transform the data later
    def fit(self, data: Union[pd.DataFrame, Dict[str, Any]]):
        logger = logging.getLogger()
        
        if isinstance(data, dict):
            data = pd.DataFrame(data)
            
        logger.info(colorize("Fitting DataProcessor...", color="cyan", bold=True))

        for name, config in self.numeric_features.items():
            if name not in data.columns:
                logger.warning(f"Numeric feature {name} not found in data")
                continue
            self._process_numeric_feature_fit(data[name], config)
        
        for name, config in self.sparse_features.items():
            if name not in data.columns:
                logger.warning(f"Sparse feature {name} not found in data")
                continue
            self._process_sparse_feature_fit(data[name], config)
        
        for name, config in self.sequence_features.items():
            if name not in data.columns:
                logger.warning(f"Sequence feature {name} not found in data")
                continue
            self._process_sequence_feature_fit(data[name], config)

        for name, config in self.target_features.items():
            if name not in data.columns:
                logger.warning(f"Target {name} not found in data")
                continue
            self._process_target_fit(data[name], config)
        
        self.is_fitted = True
        logger.info(colorize("DataProcessor fitted successfully!", color="green", bold=True))
        return self
        
    def transform(
        self, 
        data: Union[pd.DataFrame, Dict[str, Any]],
        return_dict: bool = True
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        logger = logging.getLogger()
        
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before transform")
        
        # Convert input to dict format for unified processing
        if isinstance(data, pd.DataFrame):
            data_dict = {col: data[col] for col in data.columns}
        elif isinstance(data, dict):
            data_dict = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        result_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, pd.Series):
                result_dict[key] = value.values
            elif isinstance(value, np.ndarray):
                result_dict[key] = value
            else:
                result_dict[key] = np.array(value)

        # process numeric features
        for name, config in self.numeric_features.items():
            if name not in data_dict:
                logger.warning(f"Numeric feature {name} not found in data")
                continue
            # Convert to Series for processing
            series_data = pd.Series(data_dict[name], name=name)
            processed = self._process_numeric_feature_transform(series_data, config)
            result_dict[name] = processed

        # process sparse features
        for name, config in self.sparse_features.items():
            if name not in data_dict:
                logger.warning(f"Sparse feature {name} not found in data")
                continue
            series_data = pd.Series(data_dict[name], name=name)
            processed = self._process_sparse_feature_transform(series_data, config)
            result_dict[name] = processed

        # process sequence features
        for name, config in self.sequence_features.items():
            if name not in data_dict:
                logger.warning(f"Sequence feature {name} not found in data")
                continue
            series_data = pd.Series(data_dict[name], name=name)
            processed = self._process_sequence_feature_transform(series_data, config)
            result_dict[name] = processed

        # process target features
        for name, config in self.target_features.items():
            if name not in data_dict:
                logger.warning(f"Target {name} not found in data")
                continue
            series_data = pd.Series(data_dict[name], name=name)
            processed = self._process_target_transform(series_data, config)
            result_dict[name] = processed

        if return_dict:
            return result_dict
        else:
            # Convert all arrays to Series/lists at once to avoid fragmentation
            columns_dict = {}
            for key, value in result_dict.items():
                if key in self.sequence_features:
                    columns_dict[key] = [list(seq) for seq in value]
                else:
                    columns_dict[key] = value
            
            result_df = pd.DataFrame(columns_dict)
            return result_df
            
    def fit_transform(
        self, 
        data: Union[pd.DataFrame, Dict[str, Any]],
        return_dict: bool = True
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        self.fit(data)
        return self.transform(data, return_dict=return_dict)
        
    def save(self, filepath: str):
        logger = logging.getLogger()
        
        if not self.is_fitted:
            logger.warning("Saving unfitted DataProcessor")

        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        state = {
            'numeric_features': self.numeric_features,
            'sparse_features': self.sparse_features,
            'sequence_features': self.sequence_features,
            'target_features': self.target_features,
            'is_fitted': self.is_fitted,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'target_encoders': self.target_encoders
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"DataProcessor saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> 'DataProcessor':
        logger = logging.getLogger()
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        processor = cls()
        processor.numeric_features = state['numeric_features']
        processor.sparse_features = state['sparse_features']
        processor.sequence_features = state['sequence_features']
        processor.target_features = state['target_features']
        processor.is_fitted = state['is_fitted']
        processor.scalers = state['scalers']
        processor.label_encoders = state['label_encoders']
        processor.target_encoders = state['target_encoders']
        
        logger.info(f"DataProcessor loaded from {filepath}")
        return processor
        
    def get_vocab_sizes(self) -> Dict[str, int]:
        vocab_sizes = {}
        
        for name, config in self.sparse_features.items():
            vocab_sizes[name] = config.get('vocab_size', 0)
        
        for name, config in self.sequence_features.items():
            vocab_sizes[name] = config.get('vocab_size', 0)
        
        return vocab_sizes
        
    def summary(self):

        logger = logging.getLogger()
        
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))
        logger.info(colorize("DataProcessor Summary", color="bright_blue", bold=True))
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))
        
        logger.info("")
        logger.info(colorize("[1] Feature Configuration", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))
        
        if self.numeric_features:
            logger.info(f"Dense Features ({len(self.numeric_features)}):")
            
            max_name_len = max(len(name) for name in self.numeric_features.keys())
            name_width = max(max_name_len, 10) + 2
            
            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Scaler':>15} {'Fill NA':>10}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*15} {'-'*10}")
            for i, (name, config) in enumerate(self.numeric_features.items(), 1):
                scaler = config['scaler']
                fill_na = config.get('fill_na_value', config.get('fill_na', 'N/A'))
                logger.info(f"  {i:<4} {name:<{name_width}} {str(scaler):>15} {str(fill_na):>10}")
        
        if self.sparse_features:
            logger.info(f"Sparse Features ({len(self.sparse_features)}):")
            
            max_name_len = max(len(name) for name in self.sparse_features.keys())
            name_width = max(max_name_len, 10) + 2
            
            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Method':>12} {'Vocab Size':>12} {'Hash Size':>12}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*12} {'-'*12}")
            for i, (name, config) in enumerate(self.sparse_features.items(), 1):
                method = config['encode_method']
                vocab_size = config.get('vocab_size', 'N/A')
                hash_size = config.get('hash_size', 'N/A')
                logger.info(f"  {i:<4} {name:<{name_width}} {str(method):>12} {str(vocab_size):>12} {str(hash_size):>12}")
        
        if self.sequence_features:
            logger.info(f"Sequence Features ({len(self.sequence_features)}):")
            
            max_name_len = max(len(name) for name in self.sequence_features.keys())
            name_width = max(max_name_len, 10) + 2
            
            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Method':>12} {'Vocab Size':>12} {'Hash Size':>12} {'Max Len':>10}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
            for i, (name, config) in enumerate(self.sequence_features.items(), 1):
                method = config['encode_method']
                vocab_size = config.get('vocab_size', 'N/A')
                hash_size = config.get('hash_size', 'N/A')
                max_len = config.get('max_len', 'N/A')
                logger.info(f"  {i:<4} {name:<{name_width}} {str(method):>12} {str(vocab_size):>12} {str(hash_size):>12} {str(max_len):>10}")
        
        logger.info("")
        logger.info(colorize("[2] Target Configuration", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))
        
        if self.target_features:
            logger.info(f"Target Features ({len(self.target_features)}):")
            
            max_name_len = max(len(name) for name in self.target_features.keys())
            name_width = max(max_name_len, 10) + 2
            
            logger.info(f"  {'#':<4} {'Name':<{name_width}} {'Type':>15}")
            logger.info(f"  {'-'*4} {'-'*name_width} {'-'*15}")
            for i, (name, config) in enumerate(self.target_features.items(), 1):
                target_type = config['target_type']
                logger.info(f"  {i:<4} {name:<{name_width}} {str(target_type):>15}")
        else:
            logger.info("No target features configured")
        
        logger.info("")
        logger.info(colorize("[3] Processor Status", color="cyan", bold=True))
        logger.info(colorize("-" * 80, color="cyan"))
        logger.info(f"Fitted:                  {self.is_fitted}")
        logger.info(f"Total Features:          {len(self.numeric_features) + len(self.sparse_features) + len(self.sequence_features)}")
        logger.info(f"  Dense Features:        {len(self.numeric_features)}")
        logger.info(f"  Sparse Features:       {len(self.sparse_features)}")
        logger.info(f"  Sequence Features:     {len(self.sequence_features)}")
        logger.info(f"Target Features:         {len(self.target_features)}")
        
        logger.info("")
        logger.info("")
        logger.info(colorize("=" * 80, color="bright_blue", bold=True))