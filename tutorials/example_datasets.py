"""
Example: Using RecLib datasets for recommendation model training.

This example demonstrates how to:
1. Download and load popular recommendation datasets
2. Preprocess data for different recommendation tasks
3. Train models using the datasets
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from reclib.datasets import get_dataset, list_datasets
from reclib.datasets.preprocessing import (
    DataPreprocessor,
    filter_k_core,
    create_feature_columns
)
from reclib.basic.features import DenseFeature, SparseFeature
from reclib.models.ranking.deepfm import DeepFM


def example_movielens_100k():
    """Example: MovieLens 100K for rating prediction."""
    print("\n" + "="*70)
    print("Example 1: MovieLens 100K Dataset")
    print("="*70)
    
    # Load dataset
    dataset = get_dataset("movielens-100k", root="../data", download=True)
    
    # Load data with features (load first to cache the correct version)
    df = dataset.load(include_features=True)
    
    # Show dataset info after loading
    dataset.info()
    print(f"\nLoaded {len(df)} interactions")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Preprocessing
    print("\n--- Data Preprocessing ---")
    preprocessor = DataPreprocessor()
    
    # Create binary labels (rating >= 4 as positive)
    df = preprocessor.create_binary_labels(df, rating_col='rating', threshold=4.0)
    
    # Filter to 5-core
    df = filter_k_core(df, user_col='user_id', item_col='item_id', k_core=5)
    print(f"After 5-core filtering: {len(df)} interactions")
    
    # Check which columns are available
    available_sparse_cols = ['user_id', 'item_id']
    available_dense_cols = []
    
    if 'gender' in df.columns:
        available_sparse_cols.append('gender')
    if 'occupation' in df.columns:
        available_sparse_cols.append('occupation')
    if 'age' in df.columns:
        available_dense_cols.append('age')
    
    # Encode categorical features
    df = preprocessor.encode_categorical(df, available_sparse_cols, fit=True)
    
    # Normalize numerical features
    if available_dense_cols:
        df = preprocessor.normalize_numerical(df, available_dense_cols, fit=True)
    
    # Split data
    train_df, valid_df, test_df = preprocessor.split_by_ratio(df, ratios=(0.7, 0.15, 0.15))
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # Define features for model - use same embedding_dim for all sparse features
    embedding_dim = 16
    
    dense_features = []
    if 'age' in df.columns:
        dense_features.append(DenseFeature('age', embedding_dim=1))
    
    sparse_features = [
        SparseFeature('user_id', vocab_size=df['user_id'].max()+1, embedding_dim=embedding_dim),
        SparseFeature('item_id', vocab_size=df['item_id'].max()+1, embedding_dim=embedding_dim),
    ]
    
    if 'gender' in df.columns:
        sparse_features.append(SparseFeature('gender', vocab_size=df['gender'].max()+1, embedding_dim=embedding_dim))
    if 'occupation' in df.columns:
        sparse_features.append(SparseFeature('occupation', vocab_size=df['occupation'].max()+1, embedding_dim=embedding_dim))
    
    # Build and train model
    print("\n--- Training DeepFM Model ---")
    model = DeepFM(
        dense_features=dense_features,
        sparse_features=sparse_features,
        mlp_params={"dims": [256, 128], "activation": "relu", "dropout": 0.2},
        target='label',
        optimizer="adam",
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        device='cpu',
        model_id="movielens_deepfm"
    )
    
    model.fit(
        train_data=train_df,
        valid_data=valid_df,
        metrics=['auc', 'logloss'],
        epochs=3,  # Use more epochs in practice
        batch_size=512,
        shuffle=True
    )
    
    # Evaluate
    predictions = model.predict(test_df, batch_size=512)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:10]}")


def example_criteo_ctr():
    """Example: Criteo dataset for CTR prediction."""
    print("\n" + "="*70)
    print("Example 2: Criteo Dataset for CTR Prediction")
    print("="*70)
    
    # Load dataset (sample version)
    dataset = get_dataset(
        "criteo",
        root="../data",
        download=True,
        use_sample=True,
        sample_size=50000
    )
    dataset.info()
    
    # Load data
    df = dataset.load(nrows=10000)  # Load subset for quick demo
    print(f"\nLoaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get feature names (for Criteo dataset)
    if hasattr(dataset, 'get_feature_names'):
        feature_info = dataset.get_feature_names()
        print(f"\nFeature info: {feature_info}")
    else:
        # Define feature names manually
        feature_info = {
            'label': 'label',
            'dense_features': [f'I{i}' for i in range(1, 14)],
            'sparse_features': [f'C{i}' for i in range(1, 27)]
        }
        print(f"\nFeature info: {feature_info}")
    
    # Preprocessing
    print("\n--- Data Preprocessing ---")
    preprocessor = DataPreprocessor()
    
    # Get feature columns
    dense_cols = feature_info['dense_features']  # I1-I13
    sparse_cols = feature_info['sparse_features']  # C1-C26
    
    # Handle missing values - fill with 0 for dense, 'missing' for sparse
    for col in dense_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    for col in sparse_cols:
        if col in df.columns:
            df[col] = df[col].fillna('missing').astype(str)
    
    # Log transformation for integer features
    for col in dense_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].abs())
    
    # Encode categorical features
    df = preprocessor.encode_categorical(df, sparse_cols, fit=True)
    
    # Normalize numerical features
    df = preprocessor.normalize_numerical(df, dense_cols, fit=True)
    
    # Split data
    train_df, valid_df, test_df = preprocessor.split_by_ratio(df, ratios=(0.7, 0.15, 0.15))
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    # Define features for model
    embedding_dim = 16
    
    dense_features = [DenseFeature(col, embedding_dim=1) for col in dense_cols]
    sparse_features = [
        SparseFeature(col, vocab_size=df[col].max()+1, embedding_dim=embedding_dim)
        for col in sparse_cols
    ]
    
    # Build and train model
    print("\n--- Training DeepFM Model ---")
    model = DeepFM(
        dense_features=dense_features,
        sparse_features=sparse_features,
        mlp_params={"dims": [256, 128], "activation": "relu", "dropout": 0.2},
        target='label',
        optimizer="adam",
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        device='cpu',
        model_id="criteo_deepfm"
    )
    
    model.fit(
        train_data=train_df,
        valid_data=valid_df,
        metrics=['auc', 'logloss'],
        epochs=3,  # Use more epochs in practice
        batch_size=512,
        shuffle=True
    )
    
    # Evaluate
    predictions = model.predict(test_df, batch_size=512)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:10]}")

def example_list_all_datasets():
    """List all available datasets."""
    print("\n" + "="*70)
    print("Available Datasets in RecLib")
    print("="*70)
    
    datasets = list_datasets()
    print(f"\nTotal datasets: {len(datasets)}")
    print("\nDataset names:")
    for i, name in enumerate(datasets, 1):
        print(f"  {i}. {name}")
    
    print("\nUsage:")
    print("  dataset = get_dataset('dataset-name', root='./data', download=True)")
    print("  df = dataset.load()")

if __name__ == "__main__":

    example_list_all_datasets()

    try:
        example_movielens_100k()
        example_criteo_ctr()
    except Exception as e:
        print(f"\nExample failed: {e}")
