import os
import glob
import torch
import numpy as np
import pandas as pd
from typing import Iterator, Literal, Union
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, IterableDataset

from reclib.basic.features import DenseFeature, SparseFeature, SequenceFeature


class FileDataset(IterableDataset):
    """
    Iterable dataset for reading multiple files in batches.
    Supports CSV and Parquet files with chunk-based reading.
    """
    
    def __init__(self, 
                 file_paths: list[str],
                 dense_features: list[DenseFeature],
                 sparse_features: list[SparseFeature],
                 sequence_features: list[SequenceFeature],
                 target_columns: list[str],
                 chunk_size: int = 10000,
                 file_type: Literal['csv', 'parquet'] = 'csv'):
        """
        Args:
            file_paths: List of file paths to read
            dense_features: List of dense feature definitions
            sparse_features: List of sparse feature definitions
            sequence_features: List of sequence feature definitions
            target_columns: List of target column names
            chunk_size: Number of rows per chunk for CSV files
            file_type: Type of files ('csv' or 'parquet')
        """
        self.file_paths = file_paths
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.sequence_features = sequence_features
        self.target_columns = target_columns
        self.chunk_size = chunk_size
        self.file_type = file_type
        
        self.all_features = dense_features + sparse_features + sequence_features
        self.feature_names = [f.name for f in self.all_features]
    
    def __iter__(self) -> Iterator[tuple]:
        """
        Iterate through all files and yield batches of tensors.
        """
        for file_path in self.file_paths:
            if self.file_type == 'csv':
                yield from self._read_csv_chunks(file_path)
            elif self.file_type == 'parquet':
                yield from self._read_parquet_chunks(file_path)
    
    def _read_csv_chunks(self, file_path: str) -> Iterator[tuple]:
        """Read CSV file in chunks and convert to tensors."""
        chunk_iterator = pd.read_csv(file_path, chunksize=self.chunk_size)
        
        for chunk in chunk_iterator:
            tensors = self._dataframe_to_tensors(chunk)
            if tensors:
                yield tensors
    
    def _read_parquet_chunks(self, file_path: str) -> Iterator[tuple]:
        """Read Parquet file in chunks and convert to tensors."""
        # Read parquet file
        df = pd.read_parquet(file_path)
        
        # Split into chunks
        num_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]
            
            tensors = self._dataframe_to_tensors(chunk)
            if tensors:
                yield tensors
    
    def _dataframe_to_tensors(self, df: pd.DataFrame) -> tuple | None:
        """Convert DataFrame chunk to tuple of tensors."""
        tensors = []
        
        # Process features
        for feature in self.all_features:
            if feature.name not in df.columns:
                continue
            
            column_data = df[feature.name].values
            
            # Handle sequence features (might be list of lists)
            if isinstance(feature, SequenceFeature):
                if column_data.dtype == object:
                    column_data = np.array([
                        np.array(seq, dtype=np.int64) if not isinstance(seq, np.ndarray) else seq 
                        for seq in column_data
                    ])
                    if column_data.ndim == 1:
                        column_data = np.vstack(list(column_data))  # type: ignore
                tensor = torch.from_numpy(np.asarray(column_data, dtype=np.int64))
            elif isinstance(feature, DenseFeature):
                tensor = torch.from_numpy(np.asarray(column_data, dtype=np.float32))
            else:  # SparseFeature
                tensor = torch.from_numpy(np.asarray(column_data, dtype=np.int64))
            
            tensors.append(tensor)
        
        # Process targets
        target_tensors = []
        for target_name in self.target_columns:
            if target_name not in df.columns:
                continue
            
            target_data = df[target_name].values
            target_tensor = torch.from_numpy(np.asarray(target_data, dtype=np.float32))
            
            if target_tensor.dim() == 1:
                target_tensor = target_tensor.view(-1, 1)
            
            target_tensors.append(target_tensor)
        
        # Combine target tensors
        if target_tensors:
            if len(target_tensors) == 1 and target_tensors[0].shape[1] > 1:
                y_tensor = target_tensors[0]
            else:
                y_tensor = torch.cat(target_tensors, dim=1)
            
            if y_tensor.shape[1] == 1:
                y_tensor = y_tensor.squeeze(1)
            
            tensors.append(y_tensor)
        
        if not tensors:
            return None
        
        return tuple(tensors)


def collate_fn(batch):
    """
    Custom collate function for batching tuples of tensors.
    Each element in batch is a tuple of tensors from FileDataset.
    """
    if not batch:
        return tuple()
    
    # batch is a list of tuples, each tuple contains multiple tensors
    # We need to stack tensors at each position
    num_tensors = len(batch[0])
    result = []
    
    for i in range(num_tensors):
        tensor_list = [item[i] for item in batch]
        stacked = torch.cat(tensor_list, dim=0)
        result.append(stacked)
    
    return tuple(result)


class RecDataLoader:
    """
    Custom DataLoader for recommendation models.
    Supports multiple input formats: dict, DataFrame, CSV files, Parquet files, and directories.
    """
    
    def __init__(self,
                 dense_features: list[DenseFeature] | None = None,
                 sparse_features: list[SparseFeature] | None = None,
                 sequence_features: list[SequenceFeature] | None = None,
                 target_columns: list[str] | None = None):
        """
        Args:
            dense_features: List of dense feature definitions
            sparse_features: List of sparse feature definitions
            sequence_features: List of sequence feature definitions
            target_columns: List of target column names
        """
        self.dense_features = dense_features if dense_features else []
        self.sparse_features = sparse_features if sparse_features else []
        self.sequence_features = sequence_features if sequence_features else []
        self.target_columns = target_columns if target_columns else []
        
        self.all_features = self.dense_features + self.sparse_features + self.sequence_features
    
    def create_dataloader(self,
                         data: Union[dict, pd.DataFrame, str, DataLoader],
                         batch_size: int = 32,
                         shuffle: bool = True,
                         load_full: bool = True,
                         chunk_size: int = 10000) -> DataLoader:
        """
        Create DataLoader from various input formats.
        
        Args:
            data: Input data in various formats:
                  - dict: Dictionary of arrays/lists
                  - pd.DataFrame: Pandas DataFrame
                  - str: File path (CSV/Parquet) or directory path
                  - DataLoader: Return as-is
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle data (only for in-memory loading)
            load_full: If True, load entire dataset into memory; 
                      If False, use streaming/chunked reading
            chunk_size: Chunk size for reading large files (used when load_full=False)
        
        Returns:
            DataLoader instance
        """
        # If already a DataLoader, return as-is
        if isinstance(data, DataLoader):
            return data
        
        # Handle file path or directory
        if isinstance(data, str):
            return self._create_from_path(data, batch_size, shuffle, load_full, chunk_size)
        
        # Handle dict or DataFrame
        if isinstance(data, (dict, pd.DataFrame)):
            if load_full:
                return self._create_from_memory(data, batch_size, shuffle)
            else:
                # Convert to DataFrame first if dict
                if isinstance(data, dict):
                    data = pd.DataFrame(data)
                return self._create_from_dataframe_chunks(data, batch_size, chunk_size)
        
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _create_from_memory(self, 
                           data: Union[dict, pd.DataFrame],
                           batch_size: int,
                           shuffle: bool) -> DataLoader:
        """Create DataLoader by loading full data into memory."""
        tensors = []
        
        # Process features
        for feature in self.all_features:
            column = self._get_column_data(data, feature.name)
            if column is None:
                raise KeyError(f"Feature {feature.name} not found in provided data.")
            
            # Handle sequence features
            if isinstance(feature, SequenceFeature):
                if isinstance(column, pd.Series):
                    column = column.values
                if isinstance(column, np.ndarray) and column.dtype == object:
                    column = np.array([
                        np.array(seq, dtype=np.int64) if not isinstance(seq, np.ndarray) else seq 
                        for seq in column
                    ])
                if isinstance(column, np.ndarray) and column.ndim == 1 and column.dtype == object:
                    column = np.vstack([c if isinstance(c, np.ndarray) else np.array(c) for c in column])  # type: ignore
                tensor = torch.from_numpy(np.asarray(column, dtype=np.int64))
            elif isinstance(feature, DenseFeature):
                tensor = torch.from_numpy(np.asarray(column, dtype=np.float32))
            else:  # SparseFeature
                tensor = torch.from_numpy(np.asarray(column, dtype=np.int64))
            
            tensors.append(tensor)
        
        # Process targets
        label_tensors = []
        for target_name in self.target_columns:
            column = self._get_column_data(data, target_name)
            if column is None:
                continue
            
            label_tensor = torch.from_numpy(np.asarray(column, dtype=np.float32))
            
            if label_tensor.dim() == 1:
                label_tensor = label_tensor.view(-1, 1)
            elif label_tensor.dim() == 2:
                if label_tensor.shape[0] == 1 and label_tensor.shape[1] > 1:
                    label_tensor = label_tensor.t()
            
            label_tensors.append(label_tensor)
        
        # Combine target tensors
        if label_tensors:
            if len(label_tensors) == 1 and label_tensors[0].shape[1] > 1:
                y_tensor = label_tensors[0]
            else:
                y_tensor = torch.cat(label_tensors, dim=1)
            
            if y_tensor.shape[1] == 1:
                y_tensor = y_tensor.squeeze(1)
            
            tensors.append(y_tensor)
        
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _create_from_dataframe_chunks(self,
                                     df: pd.DataFrame,
                                     batch_size: int,
                                     chunk_size: int) -> DataLoader:
        """Create DataLoader from DataFrame with chunked reading."""
        # For in-memory DataFrame, we'll split it into chunks
        num_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        def chunk_generator():
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                chunk = df.iloc[start_idx:end_idx]
                
                # Convert chunk to tensors
                tensors = []
                
                for feature in self.all_features:
                    if feature.name not in chunk.columns:
                        continue
                    
                    column_data = chunk[feature.name].values
                    
                    if isinstance(feature, SequenceFeature):
                        if column_data.dtype == object:
                            column_data = np.array([
                                np.array(seq, dtype=np.int64) if not isinstance(seq, np.ndarray) else seq 
                                for seq in column_data
                            ])
                            if column_data.ndim == 1:
                                column_data = np.vstack(list(column_data))  # type: ignore
                        tensor = torch.from_numpy(np.asarray(column_data, dtype=np.int64))
                    elif isinstance(feature, DenseFeature):
                        tensor = torch.from_numpy(np.asarray(column_data, dtype=np.float32))
                    else:
                        tensor = torch.from_numpy(np.asarray(column_data, dtype=np.int64))
                    
                    tensors.append(tensor)
                
                # Process targets
                target_tensors = []
                for target_name in self.target_columns:
                    if target_name not in chunk.columns:
                        continue
                    
                    target_data = chunk[target_name].values
                    target_tensor = torch.from_numpy(np.asarray(target_data, dtype=np.float32))
                    
                    if target_tensor.dim() == 1:
                        target_tensor = target_tensor.view(-1, 1)
                    
                    target_tensors.append(target_tensor)
                
                if target_tensors:
                    if len(target_tensors) == 1 and target_tensors[0].shape[1] > 1:
                        y_tensor = target_tensors[0]
                    else:
                        y_tensor = torch.cat(target_tensors, dim=1)
                    
                    if y_tensor.shape[1] == 1:
                        y_tensor = y_tensor.squeeze(1)
                    
                    tensors.append(y_tensor)
                
                yield tuple(tensors)
        
        # Create iterable dataset from generator
        class GeneratorDataset(IterableDataset):
            def __init__(self, gen_func):
                self.gen_func = gen_func
            
            def __iter__(self):
                return self.gen_func()
        
        dataset = GeneratorDataset(chunk_generator)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    def _create_from_path(self,
                         path: str,
                         batch_size: int,
                         shuffle: bool,
                         load_full: bool,
                         chunk_size: int) -> DataLoader:
        """Create DataLoader from file path or directory."""
        path_obj = Path(path)
        
        # Determine if it's a file or directory
        if path_obj.is_file():
            file_paths = [str(path_obj)]
            file_type = self._get_file_type(str(path_obj))
        elif path_obj.is_dir():
            # Find all CSV and Parquet files in directory
            csv_files = glob.glob(os.path.join(path, "*.csv"))
            parquet_files = glob.glob(os.path.join(path, "*.parquet"))
            
            if csv_files and parquet_files:
                raise ValueError("Directory contains both CSV and Parquet files. Please use a single format.")
            
            file_paths = csv_files if csv_files else parquet_files
            
            if not file_paths:
                raise ValueError(f"No CSV or Parquet files found in directory: {path}")
            
            file_type = 'csv' if csv_files else 'parquet'
            file_paths.sort()  # Sort for consistent ordering
        else:
            raise ValueError(f"Invalid path: {path}")
        
        # Load full data into memory or use streaming
        if load_full:
            return self._load_files_full(file_paths, file_type, batch_size, shuffle)
        else:
            return self._load_files_streaming(file_paths, file_type, batch_size, chunk_size)
    
    def _load_files_full(self,
                        file_paths: list[str],
                        file_type: Literal['csv', 'parquet'],
                        batch_size: int,
                        shuffle: bool) -> DataLoader:
        """Load all files into memory and create DataLoader."""
        # Read all files and concatenate
        dfs = []
        for file_path in file_paths:
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            else:  # parquet
                df = pd.read_parquet(file_path)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        return self._create_from_memory(combined_df, batch_size, shuffle)
    
    def _load_files_streaming(self,
                             file_paths: list[str],
                             file_type: Literal['csv', 'parquet'],
                             batch_size: int,
                             chunk_size: int) -> DataLoader:
        """Create streaming DataLoader for large files."""
        dataset = FileDataset(
            file_paths=file_paths,
            dense_features=self.dense_features,
            sparse_features=self.sparse_features,
            sequence_features=self.sequence_features,
            target_columns=self.target_columns,
            chunk_size=chunk_size,
            file_type=file_type
        )
        
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    def _get_file_type(self, file_path: str) -> Literal['csv', 'parquet']:
        """Determine file type from extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return 'csv'
        elif ext == '.parquet':
            return 'parquet'
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _get_column_data(self, data: Union[dict, pd.DataFrame], name: str):
        """Extract column data from dict or DataFrame."""
        if isinstance(data, dict):
            return data.get(name, None)
        elif isinstance(data, pd.DataFrame):
            if name not in data.columns:
                return None
            return data[name].values
        else:
            if hasattr(data, name):
                return getattr(data, name)
            raise KeyError(f"Unsupported data type for extracting column {name}")
