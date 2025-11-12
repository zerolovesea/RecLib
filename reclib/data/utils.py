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