"""
Generate training datasets for match, multitask, and ranking tasks
Each dataset includes user_id, item_id, dense features, sparse features, and sequence features
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_match_dataset(
    num_users: int = 5000,
    num_items: int = 10000,
    samples_per_user: int = 20,
    num_user_dense: int = 3,
    num_user_sparse: int = 5,
    num_user_sequence: int = 1,
    num_item_dense: int = 2,
    num_item_sparse: int = 4,
    sequence_length: int = 20,
    seed: int = 2024
):
    """
    Generate match task dataset (e.g., DSSM)
    Each user has multiple item interactions
    """
    np.random.seed(seed)
    
    num_samples = num_users * samples_per_user
    data = {}
    
    # Generate user_id and item_id
    user_ids = np.repeat(np.arange(1, num_users + 1), samples_per_user)
    item_ids = np.random.randint(1, num_items + 1, num_samples)
    
    data['user_id'] = user_ids
    data['item_id'] = item_ids
    
    # User features (same for all items of the same user)
    # User dense features
    for i in range(num_user_dense):
        user_values = np.random.random(num_users).astype(np.float32)
        data[f'user_dense_{i}'] = np.repeat(user_values, samples_per_user)
    
    # User sparse features
    for i in range(num_user_sparse):
        vocab_size = np.random.randint(100, 1000)
        user_values = np.random.randint(1, vocab_size, num_users)
        data[f'user_sparse_{i}'] = np.repeat(user_values, samples_per_user)
    
    # User sequence features (e.g., behavior history)
    for i in range(num_user_sequence):
        sequences = []
        vocab_size = np.random.randint(500, 2000)
        user_sequences = []
        for _ in range(num_users):
            actual_length = np.random.randint(5, sequence_length + 1)
            seq = np.random.randint(1, vocab_size, actual_length).tolist()
            seq = seq + [0] * (sequence_length - actual_length)
            user_sequences.append(seq)
        
        for user_seq in user_sequences:
            sequences.extend([user_seq] * samples_per_user)
        
        data[f'user_sequence_{i}'] = sequences
    
    # Item features (vary by item)
    # Item dense features
    for i in range(num_item_dense):
        data[f'item_dense_{i}'] = np.random.random(num_samples).astype(np.float32)
    
    # Item sparse features
    for i in range(num_item_sparse):
        vocab_size = np.random.randint(100, 1000)
        data[f'item_sparse_{i}'] = np.random.randint(1, vocab_size, num_samples)
    
    # Generate labels based on feature interaction
    score = np.zeros(num_samples)
    if num_user_dense > 0 and num_item_dense > 0:
        score += np.abs(data['user_dense_0'] - data['item_dense_0']) * 0.4
    if num_user_sparse > 0 and num_item_sparse > 0:
        match = (data['user_sparse_0'] % 10) == (data['item_sparse_0'] % 10)
        score += match.astype(float) * 0.6
    
    noise = np.random.normal(0, 0.15, num_samples)
    label = ((score + noise) > 0.5).astype(int)
    data['label'] = label
    
    df = pd.DataFrame(data)
    return df


def generate_multitask_dataset(
    num_users: int = 5000,
    num_items: int = 10000,
    samples_per_user: int = 20,
    num_dense_features: int = 8,
    num_sparse_features: int = 12,
    num_sequence_features: int = 1,
    num_tasks: int = 2,
    sequence_length: int = 15,
    seed: int = 2024
):
    """
    Generate multi-task dataset (e.g., MMOE, PLE, ESMM)
    Multiple tasks with shared features
    """
    np.random.seed(seed)
    
    num_samples = num_users * samples_per_user
    data = {}
    
    # Generate user_id and item_id
    user_ids = np.repeat(np.arange(1, num_users + 1), samples_per_user)
    item_ids = np.random.randint(1, num_items + 1, num_samples)
    
    data['user_id'] = user_ids
    data['item_id'] = item_ids
    
    # Dense features
    for i in range(num_dense_features):
        data[f'dense_{i}'] = np.random.random(num_samples).astype(np.float32)
    
    # Sparse features
    for i in range(num_sparse_features):
        vocab_size = np.random.randint(50, 500)
        data[f'sparse_{i}'] = np.random.randint(1, vocab_size, num_samples)
    
    # Sequence features
    for i in range(num_sequence_features):
        sequences = []
        vocab_size = np.random.randint(100, 1000)
        
        for _ in range(num_samples):
            actual_length = np.random.randint(1, sequence_length + 1)
            seq = np.random.randint(1, vocab_size, actual_length).tolist()
            seq = seq + [0] * (sequence_length - actual_length)
            sequences.append(seq)
        
        data[f'sequence_{i}'] = sequences
    
    # Generate multiple task labels with correlation
    base_score = np.zeros(num_samples)
    if num_dense_features > 0:
        base_score += data['dense_0'] * 0.3
    if num_sparse_features > 0:
        base_score += (data['sparse_0'] % 10) / 10 * 0.3
    if num_sequence_features > 0:
        seq_lengths = [len([x for x in seq if x != 0]) for seq in data['sequence_0']]
        base_score += np.array(seq_lengths) / sequence_length * 0.4
    
    task_names = ['click', 'conversion', 'like', 'share', 'comment', 'purchase']
    for task_id in range(num_tasks):
        task_score = base_score.copy()
        
        # Different tasks have different feature dependencies
        if num_dense_features > 1:
            task_score += data[f'dense_{min(task_id, num_dense_features-1)}'] * 0.2
        
        # Add task-specific noise
        noise = np.random.normal(0, 0.1 + task_id * 0.05, num_samples)
        
        # Conversion is dependent on click
        if task_id == 1:  # conversion task
            task_score = task_score * 0.7  # Lower conversion rate
        
        label = ((task_score + noise) > 0.5).astype(int)
        
        label_name = task_names[task_id] if task_id < len(task_names) else f'task_{task_id}'
        data[label_name] = label
    
    df = pd.DataFrame(data)
    return df


def generate_ranking_dataset(
    num_users: int = 5000,
    num_items: int = 10000,
    samples_per_user: int = 20,
    num_dense_features: int = 8,
    num_sparse_features: int = 10,
    num_sequence_features: int = 2,
    sequence_length: int = 15,
    seed: int = 2024
):
    """
    Generate ranking task dataset (e.g., DIN, DeepFM)
    Features capture user-item interactions
    """
    np.random.seed(seed)
    
    num_samples = num_users * samples_per_user
    data = {}
    
    # Generate user_id and item_id
    user_ids = np.repeat(np.arange(1, num_users + 1), samples_per_user)
    item_ids = np.random.randint(1, num_items + 1, num_samples)
    
    data['user_id'] = user_ids
    data['item_id'] = item_ids
    
    # Dense features (can be user features, item features, or interaction features)
    for i in range(num_dense_features):
        data[f'dense_{i}'] = np.random.random(num_samples).astype(np.float32)
    
    # Sparse features (user, item, context features)
    for i in range(num_sparse_features):
        vocab_size = np.random.randint(50, 500)
        data[f'sparse_{i}'] = np.random.randint(1, vocab_size, num_samples)
    
    # Sequence features (e.g., user behavior history)
    for i in range(num_sequence_features):
        sequences = []
        vocab_size = np.random.randint(100, 1000)
        
        for _ in range(num_samples):
            actual_length = np.random.randint(1, sequence_length + 1)
            seq = np.random.randint(1, vocab_size, actual_length).tolist()
            seq = seq + [0] * (sequence_length - actual_length)
            sequences.append(seq)
        
        data[f'sequence_{i}'] = sequences
    
    # Generate labels with complex feature interactions
    score = np.zeros(num_samples)
    if num_dense_features > 0:
        score += data['dense_0'] * 0.3
    if num_sparse_features > 0:
        score += (data['sparse_0'] % 10) / 10 * 0.3
    if num_sequence_features > 0:
        seq_lengths = [len([x for x in seq if x != 0]) for seq in data['sequence_0']]
        score += np.array(seq_lengths) / sequence_length * 0.4
    
    noise = np.random.normal(0, 0.1, num_samples)
    label = ((score + noise) > 0.5).astype(int)
    data['label'] = label
    
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    print("=" * 80)
    print("Generating Training Datasets for RecLib")
    print("=" * 80)
    
    # Create data directory
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Generate match task dataset
    print("\n1. Generating match_task.csv...")
    df_match = generate_match_dataset(
        num_users=5000,
        num_items=10000,
        samples_per_user=20,
        seed=2024
    )
    output_path = data_dir / 'match_task.csv'
    df_match.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    print(f"   Shape: {df_match.shape}")
    print(f"   Users: {df_match['user_id'].nunique()}")
    print(f"   Items: {df_match['item_id'].nunique()}")
    print(f"   Samples per user: {len(df_match) // df_match['user_id'].nunique()}")
    print(f"   Positive ratio: {df_match['label'].mean():.4f}")
    print(f"   Sample columns: {df_match.columns.tolist()}")
    
    # Generate multitask dataset
    print("\n2. Generating multitask_task.csv...")
    df_multitask = generate_multitask_dataset(
        num_users=5000,
        num_items=10000,
        samples_per_user=20,
        num_tasks=2,
        seed=2024
    )
    output_path = data_dir / 'multitask_task.csv'
    df_multitask.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    print(f"   Shape: {df_multitask.shape}")
    print(f"   Users: {df_multitask['user_id'].nunique()}")
    print(f"   Items: {df_multitask['item_id'].nunique()}")
    print(f"   Samples per user: {len(df_multitask) // df_multitask['user_id'].nunique()}")
    print(f"   Task labels: click={df_multitask['click'].mean():.4f}, conversion={df_multitask['conversion'].mean():.4f}")
    print(f"   Sample columns: {df_multitask.columns.tolist()}")
    
    # Generate ranking task dataset
    print("\n3. Generating ranking_task.csv...")
    df_ranking = generate_ranking_dataset(
        num_users=5000,
        num_items=10000,
        samples_per_user=20,
        seed=2024
    )
    output_path = data_dir / 'ranking_task.csv'
    df_ranking.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")
    print(f"   Shape: {df_ranking.shape}")
    print(f"   Users: {df_ranking['user_id'].nunique()}")
    print(f"   Items: {df_ranking['item_id'].nunique()}")
    print(f"   Samples per user: {len(df_ranking) // df_ranking['user_id'].nunique()}")
    print(f"   Positive ratio: {df_ranking['label'].mean():.4f}")
    print(f"   Sample columns: {df_ranking.columns.tolist()}")
    
    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {data_dir}/match_task.csv")
    print(f"  - {data_dir}/multitask_task.csv")
    print(f"  - {data_dir}/ranking_task.csv")
