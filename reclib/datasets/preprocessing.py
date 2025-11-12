"""
Data preprocessing utilities for RecLib datasets.

This module provides utilities for preprocessing recommendation datasets,
including feature engineering, splitting, and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional, Any
import warnings


class DataPreprocessor:
    """
    Data preprocessor for recommendation datasets.
    
    Provides utilities for:
    - Label encoding categorical features
    - Normalizing numerical features
    - Creating train/valid/test splits
    - Generating negative samples
    - Creating user/item sequences
    """
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.feature_info: Dict[str, Any] = {}
    
    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical columns using LabelEncoder.
        
        Args:
            df: Input DataFrame.
            columns: List of categorical column names.
            fit: If True, fit encoders. Otherwise use existing encoders.
        
        Returns:
            DataFrame with encoded categorical columns.
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in DataFrame")
                continue
            
            # Fill missing values
            df[col] = df[col].fillna('missing')
            df[col] = df[col].astype(str)
            
            if fit:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col])
                self.feature_info[col] = {
                    'type': 'categorical',
                    'vocab_size': len(self.label_encoders[col].classes_)
                }
            
            if col in self.label_encoders:
                # Handle unknown categories
                known_labels = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(
                    lambda x: x if x in known_labels else 'missing'
                )
                df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def normalize_numerical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize numerical columns to [0, 1] range.
        
        Args:
            df: Input DataFrame.
            columns: List of numerical column names.
            fit: If True, fit scalers. Otherwise use existing scalers.
        
        Returns:
            DataFrame with normalized numerical columns.
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in DataFrame")
                continue
            
            # Fill missing values with median
            df[col] = df[col].fillna(df[col].median())
            
            if fit:
                if col not in self.scalers:
                    self.scalers[col] = MinMaxScaler()
                self.scalers[col].fit(df[[col]])
                self.feature_info[col] = {
                    'type': 'numerical',
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
            
            if col in self.scalers:
                df[col] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def create_binary_labels(
        self,
        df: pd.DataFrame,
        rating_col: str = 'rating',
        threshold: float = 3.5,
        label_col: str = 'label'
    ) -> pd.DataFrame:
        """
        Create binary labels from ratings.
        
        Args:
            df: Input DataFrame.
            rating_col: Name of the rating column.
            threshold: Threshold for positive label (ratings >= threshold are positive).
            label_col: Name of the output label column.
        
        Returns:
            DataFrame with binary label column.
        """
        df = df.copy()
        df[label_col] = (df[rating_col] >= threshold).astype(int)
        return df
    
    def split_by_ratio(
        self,
        df: pd.DataFrame,
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        shuffle: bool = True,
        random_state: int = 2024
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/valid/test sets by ratio.
        
        Args:
            df: Input DataFrame.
            ratios: (train, valid, test) ratios. Must sum to 1.0.
            shuffle: Whether to shuffle before splitting.
            random_state: Random seed.
        
        Returns:
            Tuple of (train_df, valid_df, test_df).
        """
        assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        train_ratio, valid_ratio, test_ratio = ratios
        
        # First split: train + valid vs test
        train_valid_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            shuffle=shuffle,
            random_state=random_state
        )
        
        # Second split: train vs valid
        valid_ratio_adjusted = valid_ratio / (train_ratio + valid_ratio)
        train_df, valid_df = train_test_split(
            train_valid_df,
            test_size=valid_ratio_adjusted,
            shuffle=shuffle,
            random_state=random_state
        )
        
        return train_df, valid_df, test_df
    
    def split_by_time(
        self,
        df: pd.DataFrame,
        time_col: str = 'timestamp',
        ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/valid/test sets by time.
        
        Args:
            df: Input DataFrame.
            time_col: Name of the timestamp column.
            ratios: (train, valid, test) ratios.
        
        Returns:
            Tuple of (train_df, valid_df, test_df).
        """
        df = df.sort_values(time_col).reset_index(drop=True)
        
        n = len(df)
        train_size = int(n * ratios[0])
        valid_size = int(n * ratios[1])
        
        train_df = df.iloc[:train_size]
        valid_df = df.iloc[train_size:train_size + valid_size]
        test_df = df.iloc[train_size + valid_size:]
        
        return train_df, valid_df, test_df
    
    def split_by_user(
        self,
        df: pd.DataFrame,
        user_col: str = 'user_id',
        time_col: str = 'timestamp',
        leave_n: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by leaving out last N interactions per user.
        
        This is commonly used for sequential recommendation evaluation.
        
        Args:
            df: Input DataFrame.
            user_col: Name of the user ID column.
            time_col: Name of the timestamp column.
            leave_n: Number of last interactions to leave for test set.
        
        Returns:
            Tuple of (train_df, test_df).
        """
        df = df.sort_values([user_col, time_col]).reset_index(drop=True)
        
        train_list = []
        test_list = []
        
        for user, group in df.groupby(user_col):
            if len(group) > leave_n:
                train_list.append(group.iloc[:-leave_n])
                test_list.append(group.iloc[-leave_n:])
            else:
                train_list.append(group)
        
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
        
        return train_df, test_df
    
    def generate_negative_samples(
        self,
        df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        n_neg: int = 4,
        label_col: str = 'label'
    ) -> pd.DataFrame:
        """
        Generate negative samples for implicit feedback data.
        
        Args:
            df: Input DataFrame with positive samples.
            user_col: Name of the user ID column.
            item_col: Name of the item ID column.
            n_neg: Number of negative samples per positive sample.
            label_col: Name of the label column.
        
        Returns:
            DataFrame with both positive and negative samples.
        """
        # Get all unique items
        all_items = df[item_col].unique()
        all_items_set = set(all_items)
        
        # Create user-item interaction dict
        user_items = df.groupby(user_col)[item_col].apply(set).to_dict()
        
        negative_samples = []
        
        for _, row in df.iterrows():
            user = row[user_col]
            
            # Get negative items (items not interacted by the user)
            negative_items = list(all_items_set - user_items.get(user, set()))
            
            if len(negative_items) >= n_neg:
                sampled_items = np.random.choice(negative_items, n_neg, replace=False)
            else:
                sampled_items = negative_items
            
            for neg_item in sampled_items:
                neg_sample = row.copy()
                neg_sample[item_col] = neg_item
                neg_sample[label_col] = 0
                negative_samples.append(neg_sample)
        
        # Combine positive and negative samples
        negative_df = pd.DataFrame(negative_samples)
        result_df = pd.concat([df, negative_df], ignore_index=True)
        
        return result_df
    
    def create_user_sequences(
        self,
        df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        time_col: str = 'timestamp',
        max_len: int = 50
    ) -> pd.DataFrame:
        """
        Create user behavior sequences.
        
        Args:
            df: Input DataFrame.
            user_col: Name of the user ID column.
            item_col: Name of the item ID column.
            time_col: Name of the timestamp column.
            max_len: Maximum sequence length.
        
        Returns:
            DataFrame with user sequences.
        """
        df = df.sort_values([user_col, time_col])
        
        user_sequences = []
        
        for user, group in df.groupby(user_col):
            items = group[item_col].tolist()
            
            # Truncate or pad sequence
            if len(items) > max_len:
                items = items[-max_len:]
            
            user_sequences.append({
                user_col: user,
                'sequence': items,
                'sequence_length': len(items)
            })
        
        return pd.DataFrame(user_sequences)
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about processed features.
        
        Returns:
            Dictionary with feature information.
        """
        return self.feature_info


def filter_k_core(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'item_id',
    k_core: int = 5
) -> pd.DataFrame:
    """
    Filter dataset to k-core (users and items with at least k interactions).
    
    Args:
        df: Input DataFrame.
        user_col: Name of the user ID column.
        item_col: Name of the item ID column.
        k_core: Minimum number of interactions.
    
    Returns:
        Filtered DataFrame.
    """
    df = df.copy()
    
    while True:
        # Filter users with at least k interactions
        user_counts = df[user_col].value_counts()
        valid_users = user_counts[user_counts >= k_core].index
        df = df[df[user_col].isin(valid_users)]
        
        # Filter items with at least k interactions
        item_counts = df[item_col].value_counts()
        valid_items = item_counts[item_counts >= k_core].index
        df = df[df[item_col].isin(valid_items)]
        
        # Check if converged
        new_user_counts = df[user_col].value_counts()
        new_item_counts = df[item_col].value_counts()
        
        if (new_user_counts.min() >= k_core and 
            new_item_counts.min() >= k_core):
            break
    
    return df.reset_index(drop=True)


def create_feature_columns(
    df: pd.DataFrame,
    dense_cols: Optional[List[str]] = None,
    sparse_cols: Optional[List[str]] = None,
    sequence_cols: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Identify and categorize feature columns.
    
    Args:
        df: Input DataFrame.
        dense_cols: List of dense feature names (if None, auto-detect).
        sparse_cols: List of sparse feature names (if None, auto-detect).
        sequence_cols: List of sequence feature names.
    
    Returns:
        Dictionary with categorized column names.
    """
    if dense_cols is None:
        # Auto-detect numerical columns
        dense_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if sparse_cols is None:
        # Auto-detect categorical columns
        sparse_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if sequence_cols is None:
        sequence_cols = []
    
    return {
        'dense_features': dense_cols,
        'sparse_features': sparse_cols,
        'sequence_features': sequence_cols
    }
