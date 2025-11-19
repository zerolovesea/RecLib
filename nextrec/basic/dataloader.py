"""
Dataloader definitions

Date: create on 27/10/2025
Author:
    Yang Zhou,zyaztec@gmail.com
"""

import os
import glob
import torch
import logging
import numpy as np
import pandas as pd
import tqdm

from pathlib import Path
from typing import Iterator, Literal, Union, Optional
from torch.utils.data import DataLoader, TensorDataset, IterableDataset

from nextrec.data.preprocessor import DataProcessor
from nextrec.data import get_column_data, collate_fn

from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature
from nextrec.basic.loggers import colorize


class FileDataset(IterableDataset):
    """
    Iterable dataset for reading multiple files in batches.
    Supports CSV and Parquet files with chunk-based reading.
    """
    
    def __init__(self, 
                 file_paths: list[str],                      # file paths to read, containing CSV or Parquet files
                 dense_features: list[DenseFeature],         # dense feature definitions
                 sparse_features: list[SparseFeature],       # sparse feature definitions
                 sequence_features: list[SequenceFeature],   # sequence feature definitions
                 target_columns: list[str],                   # target column names
                 chunk_size: int = 10000,
                 file_type: Literal['csv', 'parquet'] = 'csv',
                 processor: Optional['DataProcessor'] = None): # optional DataProcessor for transformation

        self.file_paths = file_paths
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.sequence_features = sequence_features
        self.target_columns = target_columns
        self.chunk_size = chunk_size
        self.file_type = file_type
        self.processor = processor
        
        self.all_features = dense_features + sparse_features + sequence_features
        self.feature_names = [f.name for f in self.all_features]
        self.current_file_index = 0
        self.total_files = len(file_paths)
    
    def __iter__(self) -> Iterator[tuple]:
        self.current_file_index = 0
        self._file_pbar = None
        
        # Create progress bar for file processing when multiple files
        if self.total_files > 1:
            self._file_pbar = tqdm.tqdm(
                total=self.total_files, 
                desc="Files", 
                unit="file",
                position=0,
                leave=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        
        for file_path in self.file_paths:
            self.current_file_index += 1
            
            # Update file progress bar
            if self._file_pbar is not None:
                self._file_pbar.update(1)
            elif self.total_files == 1:
                # For single file, log the file name
                file_name = os.path.basename(file_path)
                logging.info(colorize(f"Processing file: {file_name}", color="cyan"))
            
            if self.file_type == 'csv':
                yield from self._read_csv_chunks(file_path)
            elif self.file_type == 'parquet':
                yield from self._read_parquet_chunks(file_path)
        
        # Close file progress bar
        if self._file_pbar is not None:
            self._file_pbar.close()
    
    def _read_csv_chunks(self, file_path: str) -> Iterator[tuple]:
        chunk_iterator = pd.read_csv(file_path, chunksize=self.chunk_size)
        
        for chunk in chunk_iterator:
            tensors = self._dataframe_to_tensors(chunk)
            if tensors:
                yield tensors
    
    def _read_parquet_chunks(self, file_path: str) -> Iterator[tuple]:
        """
        Read parquet file in chunks to reduce memory footprint.
        Uses pyarrow's batch reading for true streaming.
        """
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            chunk = batch.to_pandas()            
            tensors = self._dataframe_to_tensors(chunk)
            if tensors:
                yield tensors
            del chunk
                
    def _dataframe_to_tensors(self, df: pd.DataFrame) -> tuple | None:
        if self.processor is not None:
            if not self.processor.is_fitted:
                raise ValueError("DataProcessor must be fitted before using in streaming mode")
            transformed_data = self.processor.transform(df, return_dict=True)
        else:
            transformed_data = df
        
        tensors = []
        
        # Process features
        for feature in self.all_features:
            if self.processor is not None:
                column_data = transformed_data.get(feature.name)
                if column_data is None:
                    continue
            else:
                # Get data from original dataframe
                if feature.name not in df.columns:
                    logging.warning(colorize(f"Feature column '{feature.name}' not found in DataFrame", "yellow"))
                    continue
                column_data = df[feature.name].values
            
            # Handle sequence features: convert to 2D array of shape (batch_size, seq_length)
            if isinstance(feature, SequenceFeature):
                if isinstance(column_data, np.ndarray) and column_data.dtype == object:
                    try:
                        column_data = np.stack([np.asarray(seq, dtype=np.int64) for seq in column_data])  # type: ignore
                    except (ValueError, TypeError) as e:
                        # Fallback: handle variable-length sequences by padding
                        sequences = []
                        max_len = feature.max_len if hasattr(feature, 'max_len') else 0
                        for seq in column_data:
                            if isinstance(seq, (list, tuple, np.ndarray)):
                                seq_arr = np.asarray(seq, dtype=np.int64)
                            else:
                                seq_arr = np.array([], dtype=np.int64)
                            sequences.append(seq_arr)
                        
                        # Pad sequences to same length
                        if max_len == 0:
                            max_len = max(len(seq) for seq in sequences) if sequences else 1
                        
                        padded = []
                        for seq in sequences:
                            if len(seq) > max_len:
                                padded.append(seq[:max_len])
                            else:
                                pad_width = max_len - len(seq)
                                padded.append(np.pad(seq, (0, pad_width), constant_values=0))
                        column_data = np.stack(padded)
                else:
                    column_data = np.asarray(column_data, dtype=np.int64)
                tensor = torch.from_numpy(column_data)
            elif isinstance(feature, DenseFeature):
                tensor = torch.from_numpy(np.asarray(column_data, dtype=np.float32))
            else:  # SparseFeature
                tensor = torch.from_numpy(np.asarray(column_data, dtype=np.int64))
            
            tensors.append(tensor)
        
        # Process targets
        target_tensors = []
        for target_name in self.target_columns:
            if self.processor is not None:
                target_data = transformed_data.get(target_name)
                if target_data is None: 
                    continue
            else:
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


class RecDataLoader:
    """
    Custom DataLoader for recommendation models.
    Supports multiple input formats: dict, DataFrame, CSV files, Parquet files, and directories.
    Optionally supports DataProcessor for on-the-fly data transformation.

    Examples:
        >>> # 创建RecDataLoader
        >>> dataloader = RecDataLoader(
        >>>     dense_features=dense_features,
        >>>     sparse_features=sparse_features,
        >>>     sequence_features=sequence_features,
        >>>     target_columns=target_columns,
        >>>     processor=processor
        >>> )
    """
    
    def __init__(self,
                 dense_features: list[DenseFeature] | None = None,
                 sparse_features: list[SparseFeature] | None = None,
                 sequence_features: list[SequenceFeature] | None = None,
                 target: list[str] | None | str = None,
                 processor: Optional['DataProcessor'] = None):

        self.dense_features = dense_features if dense_features else []
        self.sparse_features = sparse_features if sparse_features else []
        self.sequence_features = sequence_features if sequence_features else []
        if isinstance(target, str):
            self.target_columns = [target]
        elif isinstance(target, list):
            self.target_columns = target
        else:
            self.target_columns = []
        self.processor = processor
        
        self.all_features = self.dense_features + self.sparse_features + self.sequence_features
    
    def create_dataloader(self,
                         data: Union[dict, pd.DataFrame, str, DataLoader],
                         batch_size: int = 32,
                         shuffle: bool = True,
                         load_full: bool = True,
                         chunk_size: int = 10000) -> DataLoader:
        """
        Create DataLoader from various data sources.
        """
        if isinstance(data, DataLoader):
            return data
        
        if isinstance(data, (str, os.PathLike)):
            return self._create_from_path(data, batch_size, shuffle, load_full, chunk_size)
        
        if isinstance(data, (dict, pd.DataFrame)):
            return self._create_from_memory(data, batch_size, shuffle)

        raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _create_from_memory(self, 
                           data: Union[dict, pd.DataFrame],
                           batch_size: int,
                           shuffle: bool) -> DataLoader:

        if self.processor is not None:
            if not self.processor.is_fitted:
                raise ValueError("DataProcessor must be fitted before using in RecDataLoader")
            data = self.processor.transform(data, return_dict=True)
        
        tensors = []
        
        # Process features
        for feature in self.all_features:
            column = get_column_data(data, feature.name)
            if column is None:
                raise KeyError(f"Feature {feature.name} not found in provided data.")

            if isinstance(feature, SequenceFeature):
                if isinstance(column, pd.Series):
                    column = column.values
                
                # Handle different input formats for sequence features
                if isinstance(column, np.ndarray):
                    # Check if elements are actually sequences (not just object dtype scalars)
                    if column.dtype == object and len(column) > 0 and isinstance(column[0], (list, tuple, np.ndarray)):
                        # Each element is a sequence (array/list), stack them into 2D array
                        try:
                            column = np.stack([np.asarray(seq, dtype=np.int64) for seq in column])  # type: ignore
                        except (ValueError, TypeError) as e:
                            # Fallback: handle variable-length sequences by padding
                            sequences = []
                            max_len = feature.max_len if hasattr(feature, 'max_len') else 0
                            for seq in column:
                                if isinstance(seq, (list, tuple, np.ndarray)):
                                    seq_arr = np.asarray(seq, dtype=np.int64)
                                else:
                                    seq_arr = np.array([], dtype=np.int64)
                                sequences.append(seq_arr)
                            
                            # Pad sequences to same length
                            if max_len == 0:
                                max_len = max(len(seq) for seq in sequences) if sequences else 1
                            
                            padded = []
                            for seq in sequences:
                                if len(seq) > max_len:
                                    padded.append(seq[:max_len])
                                else:
                                    pad_width = max_len - len(seq)
                                    padded.append(np.pad(seq, (0, pad_width), constant_values=0))
                            column = np.stack(padded)
                    elif column.ndim == 1:
                        # 1D array, need to reshape or handle appropriately
                        # Assuming each element should be treated as a single-item sequence
                        column = column.reshape(-1, 1)
                    # else: already a 2D array
                
                column = np.asarray(column, dtype=np.int64)
                tensor = torch.from_numpy(column)
                
            elif isinstance(feature, DenseFeature):
                tensor = torch.from_numpy(np.asarray(column, dtype=np.float32))
            else:  # SparseFeature
                tensor = torch.from_numpy(np.asarray(column, dtype=np.int64))
            
            tensors.append(tensor)
        
        # Process targets
        label_tensors = []
        for target_name in self.target_columns:
            column = get_column_data(data, target_name)
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
    
    def _create_from_path(self,
                         path: str,
                         batch_size: int,
                         shuffle: bool,
                         load_full: bool,
                         chunk_size: int) -> DataLoader:
        """
        Create DataLoader from a file path, supporting CSV and Parquet formats, with options for full loading or streaming.
        """

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
            dfs = []
            for file_path in file_paths:
                if file_type == 'csv':
                    df = pd.read_csv(file_path)
                else:  # parquet
                    df = pd.read_parquet(file_path)
                dfs.append(df)
            
            combined_df = pd.concat(dfs, ignore_index=True)
            return self._create_from_memory(combined_df, batch_size, shuffle)
        else:
            return self._load_files_streaming(file_paths, file_type, batch_size, chunk_size)
    

    def _load_files_streaming(self,
                             file_paths: list[str],
                             file_type: Literal['csv', 'parquet'],
                             batch_size: int,
                             chunk_size: int) -> DataLoader:
        # Create FileDataset for streaming
        dataset = FileDataset(
            file_paths=file_paths,
            dense_features=self.dense_features,
            sparse_features=self.sparse_features,
            sequence_features=self.sequence_features,
            target_columns=self.target_columns,
            chunk_size=chunk_size,
            file_type=file_type,
            processor=self.processor
        )
        
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    def _get_file_type(self, file_path: str) -> Literal['csv', 'parquet']:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return 'csv'
        elif ext == '.parquet':
            return 'parquet'
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
