"""
Data processing utilities for RecForge

Date: create on 13/11/2025
Author:
    Yang Zhou, zyaztec@gmail.com
"""

import torch
import numpy as np
import pandas as pd

def collate_fn(batch):
    """
    Custom collate function for batching tuples of tensors.
    Each element in batch is a tuple of tensors from FileDataset.

    Examples:
        # Single sample in batch
        (tensor([1.0, 2.0]), tensor([10, 20]), tensor([100, 200]), tensor(1.0))
        # Batched output
        (tensor([[1.0, 2.0], [3.0, 4.0]]),  # dense_features batch
         tensor([[10, 20], [30, 40]]),       # sparse_features batch
         tensor([[100, 200], [300, 400]]),   # sequence_features batch
         tensor([1.0, 0.0])                  # labels batch)

    """
    if not batch:
        return tuple()

    num_tensors = len(batch[0])
    result = []
    
    for i in range(num_tensors):
        tensor_list = [item[i] for item in batch]
        stacked = torch.cat(tensor_list, dim=0)
        result.append(stacked)
    
    return tuple(result)


def get_column_data(data: dict | pd.DataFrame, name: str):
    """Extract column data from various data structures."""
    if isinstance(data, dict):
        return data[name] if name in data else None
    elif isinstance(data, pd.DataFrame):
        if name not in data.columns:
            return None
        return data[name].values
    else:
        if hasattr(data, name):
            return getattr(data, name)
        raise KeyError(f"Unsupported data type for extracting column {name}")


def split_dict_random(data_dict: dict, test_size: float=0.2, random_state:int|None=None):
    """Randomly split a dictionary of data into training and testing sets."""
    lengths = [len(v) for v in data_dict.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Length mismatch: {lengths}")
    n = lengths[0]

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    cut = int(round(n * (1 - test_size)))
    train_idx, test_idx = perm[:cut], perm[cut:]

    def take(v, idx):
        if isinstance(v, np.ndarray):
            return v[idx]
        elif isinstance(v, pd.Series):
            return v.iloc[idx].to_numpy()  
        else:
            v_arr = np.asarray(v, dtype=object)  
            return v_arr[idx]

    train_dict = {k: take(v, train_idx) for k, v in data_dict.items()}
    test_dict  = {k: take(v, test_idx)  for k, v in data_dict.items()}
    return train_dict, test_dict


def build_eval_candidates(
    df_all: pd.DataFrame,
    user_col: str,
    item_col: str,
    label_col: str,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    num_pos_per_user: int = 5,
    num_neg_per_pos: int = 50,
    random_seed: int = 2025,
) -> pd.DataFrame:
    """Build evaluation candidates with positive and negative samples for each user.   """
    rng = np.random.default_rng(random_seed)

    users = df_all[user_col].unique()
    all_items = item_features[item_col].unique()

    rows = []

    user_hist_items = {
        u: df_all[df_all[user_col] == u][item_col].unique()
        for u in users
    }

    for u in users:
        df_user = df_all[df_all[user_col] == u]
        pos_items = df_user[df_user[label_col] == 1][item_col].unique()
        if len(pos_items) == 0:
            continue

        pos_items = pos_items[:num_pos_per_user]
        seen_items = set(user_hist_items[u])

        neg_pool = np.setdiff1d(all_items, np.fromiter(seen_items, dtype=all_items.dtype))
        if len(neg_pool) == 0:
            continue

        for pos in pos_items:
            if len(neg_pool) <= num_neg_per_pos:
                neg_items = neg_pool
            else:
                neg_items = rng.choice(neg_pool, size=num_neg_per_pos, replace=False)

            rows.append((u, pos, 1))
            for ni in neg_items:
                rows.append((u, ni, 0))

    eval_df = pd.DataFrame(rows, columns=[user_col, item_col, label_col])
    eval_df = eval_df.merge(user_features, on=user_col, how='left')
    eval_df = eval_df.merge(item_features, on=item_col, how='left')
    return eval_df
