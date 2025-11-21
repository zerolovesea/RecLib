"""
Dataloader definitions

Date: create on 27/10/2025
Author: Yang Zhou,zyaztec@gmail.com
"""
import os
import glob
import tqdm
import torch
import logging
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Iterator, Literal, Union, Optional

from nextrec.data.preprocessor import DataProcessor
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from nextrec.basic.features import DenseFeature, SparseFeature, SequenceFeature

from nextrec.basic.loggers import colorize
from nextrec.data import get_column_data, collate_fn


class FileDataset(IterableDataset):
    """
    Iterable dataset that streams CSV/Parquet files in chunks and yields tensor tuples.

    :param file_paths: Absolute or relative paths to CSV/Parquet files.
    :param dense_features: Dense feature definitions (float tensors).
    :param sparse_features: Sparse/categorical feature definitions (int tensors).
    :param sequence_features: Sequence feature definitions (padded int tensors).
    :param target_columns: Label/target column names.
    :param id_columns: Optional ID columns appended after targets.
    :param chunk_size: Number of rows to read per chunk.
    :param file_type: ``\"csv\"`` or ``\"parquet\"``.
    :param processor: Optional fitted :class:`~nextrec.data.preprocessor.DataProcessor` for online transform.

    Yields
    ------
    tuple
        Tensors ordered as ``dense + sparse + sequence + targets (+ ids)``. Shape respects chunk size.
    """
    
    def __init__(self, 
                 file_paths: list[str],                      # file paths to read, containing CSV or Parquet files
                 dense_features: list[DenseFeature],         # dense feature definitions
                 sparse_features: list[SparseFeature],       # sparse feature definitions
                 sequence_features: list[SequenceFeature],   # sequence feature definitions
                 target_columns: list[str],                   # target column names
                 id_columns: list[str] | None = None,         # id columns to carry through (not used for model inputs)
                 chunk_size: int = 10000,
                 file_type: Literal['csv', 'parquet'] = 'csv',
                 processor: Optional['DataProcessor'] = None): # optional DataProcessor for transformation
        """
        Initialize a streaming dataset backed by on-disk files.

        :param file_paths: File paths to read (CSV/Parquet).
        :param dense_features: Dense feature definitions.
        :param sparse_features: Sparse feature definitions.
        :param sequence_features: Sequence feature definitions.
        :param target_columns: Target/label columns.
        :param id_columns: Optional ID columns to append.
        :param chunk_size: Rows per chunk when reading.
        :param file_type: ``\"csv\"`` or ``\"parquet\"``.
        :param processor: Optional fitted ``DataProcessor``.
        """

        self.file_paths = file_paths
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.sequence_features = sequence_features
        self.target_columns = target_columns
        self.id_columns = id_columns or []
        self.chunk_size = chunk_size
        self.file_type = file_type
        self.processor = processor
        
        self.all_features = dense_features + sparse_features + sequence_features
        self.feature_names = [f.name for f in self.all_features]
        self.current_file_index = 0
        self.total_files = len(file_paths)
    
    def __iter__(self) -> Iterator[tuple]:
        """
        Iterate over files and stream tensor tuples chunk by chunk.

        Files are processed sequentially; each chunk is transformed (optionally via
        ``processor``) and converted to tensors before being yielded to PyTorch ``DataLoader``.
        """
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
            
            if self._file_pbar is not None:
                self._file_pbar.update(1)
            elif self.total_files == 1:
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
        """
        Stream a CSV file chunk by chunk.

        :param file_path: Path to the CSV file.
        :yields: Tensor tuples for each chunk.
        """
        chunk_iterator = pd.read_csv(file_path, chunksize=self.chunk_size)
        
        for chunk in chunk_iterator:
            tensors = self._dataframe_to_tensors(chunk)
            if tensors:
                yield tensors
    
    def _read_parquet_chunks(self, file_path: str) -> Iterator[tuple]:
        """
        Stream a Parquet file via ``pyarrow`` batch reading.

        :param file_path: Path to the Parquet file.
        :yields: Tensor tuples for each batch.
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
        """
        Convert a DataFrame chunk into a tuple of tensors respecting feature order.

        :param df: DataFrame chunk.
        :returns: Tuple of tensors (features + targets + ids) or ``None`` if no tensors created.
        """
        if self.processor is not None:
            if not self.processor.is_fitted:
                raise ValueError("DataProcessor must be fitted before using in streaming mode")
            transformed_data = self.processor.transform(df, return_dict=True)
        else:
            transformed_data = df

        return _build_tensors_from_data(
            data=transformed_data,
            raw_data=df,
            features=self.all_features,
            target_columns=self.target_columns,
            id_columns=self.id_columns,
            on_missing_feature="warn",
        )


class RecDataLoader:
    """
    Convenience wrapper for building PyTorch ``DataLoader`` objects for recommendation models.

    :param dense_features: Dense feature definitions (float tensors).
    :param sparse_features: Sparse/categorical feature definitions (int tensors).
    :param sequence_features: Sequence feature definitions (padded int tensors).
    :param target: Target column name(s); string or list.
    :param id_columns: Optional ID column name(s) appended after targets.
    :param processor: Optional fitted :class:`~nextrec.data.preprocessor.DataProcessor` for preprocessing.

    Examples
    --------
    >>> loader = RecDataLoader(
    ...     dense_features=dense_features,
    ...     sparse_features=sparse_features,
    ...     sequence_features=sequence_features,
    ...     target=['label'],
    ...     processor=processor,
    ... )
    >>> dataloader = loader.create_dataloader(
    ...     data=\"/path/to/data.csv\",
    ...     batch_size=1024,
    ...     load_full=False,
    ...     chunk_size=20000,
    ... )
    """
    
    def __init__(self,
                 dense_features: list[DenseFeature] | None = None,
                 sparse_features: list[SparseFeature] | None = None,
                 sequence_features: list[SequenceFeature] | None = None,
                 target: list[str] | None | str = None,
                 id_columns: str | list[str] | None = None,
                 processor: Optional['DataProcessor'] = None):
        """
        Initialize the loader with feature/target definitions.

        :param dense_features: Dense feature definitions (float).
        :param sparse_features: Sparse feature definitions (int).
        :param sequence_features: Sequence feature definitions (int, padded).
        :param target: Single target name or list of names.
        :param id_columns: Optional ID columns to append in output.
        :param processor: Optional fitted ``DataProcessor`` for preprocessing.
        """

        self.dense_features = dense_features if dense_features else []
        self.sparse_features = sparse_features if sparse_features else []
        self.sequence_features = sequence_features if sequence_features else []
        self.processor = processor
        self.all_features = self.dense_features + self.sparse_features + self.sequence_features

        self.target_columns = [target] if isinstance(target, str) else (target if isinstance(target, list) else [])
        self.id_columns = [id_columns] if isinstance(id_columns, str) else (id_columns if isinstance(id_columns, list) else [])

    def create_dataloader(self,
                         data: Union[dict, pd.DataFrame, str, DataLoader],
                         batch_size: int = 32,
                         shuffle: bool = True,
                         load_full: bool = True,
                         chunk_size: int = 10000) -> DataLoader:
        """
        Build a ``DataLoader`` from in-memory data, file path, or an existing loader.

        :param data: Dict/DataFrame (in-memory), path to CSV/Parquet file/dir, or an existing ``DataLoader``.
        :param batch_size: Batch size for the returned ``DataLoader``.
        :param shuffle: Shuffle flag passed to PyTorch ``DataLoader`` (for in-memory and streaming batches).
        :param load_full: If ``True``, load all files into memory; if ``False``, stream with chunks.
        :param chunk_size: Number of rows per chunk when ``load_full=False``.
        :returns: A configured PyTorch ``DataLoader``.
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
        """
        Convert in-memory data (dict/DataFrame) into tensors and wrap with ``DataLoader``.

        :param data: Dict or DataFrame containing feature/target columns.
        :param batch_size: Batch size.
        :param shuffle: Whether to shuffle batches.
        :returns: A ``DataLoader`` backed by ``TensorDataset``.
        """

        raw_data = data

        if self.processor is not None:
            assert self.processor.is_fitted, "DataProcessor must be fitted before using in RecDataLoader"
            data = self.processor.transform(data, return_dict=True)

        tensors = _build_tensors_from_data(
            data=data,
            raw_data=raw_data,
            features=self.all_features,
            target_columns=self.target_columns,
            id_columns=self.id_columns,
            on_missing_feature="raise",
        )

        assert tensors is not None, "No tensors were created from provided data."

        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _create_from_path(self,
                         path: str,
                         batch_size: int,
                         shuffle: bool,
                         load_full: bool,
                         chunk_size: int) -> DataLoader:
        """
        Build a ``DataLoader`` from a CSV/Parquet file or directory.

        :param path: File path or directory containing homogeneous CSV/Parquet files.
        :param batch_size: Batch size.
        :param shuffle: Shuffle flag.
        :param load_full: If ``True``, load all rows into memory; otherwise stream.
        :param chunk_size: Chunk rows when streaming.
        :returns: A ``DataLoader`` (in-memory or streaming).
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
            assert not (csv_files and parquet_files), "Directory contains both CSV and Parquet files. Please use a single format."
            
            file_paths = csv_files if csv_files else parquet_files
            assert file_paths, f"No CSV or Parquet files found in directory: {path}"
            
            file_type = 'csv' if csv_files else 'parquet'
            file_paths.sort()  # Sort for consistent ordering
        else:
            raise ValueError(f"Invalid path: {path}")
        
        # Load full data into memory
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
        """
        Create a streaming ``DataLoader`` that yields chunked tensors from files.

        :param file_paths: Ordered list of file paths.
        :param file_type: ``\"csv\"`` or ``\"parquet\"``.
        :param batch_size: Batch size for the outer ``DataLoader``.
        :param chunk_size: Number of rows per chunk when reading files.
        :returns: Streaming ``DataLoader`` with custom ``collate_fn``.
        """
        
        # Create FileDataset for streaming
        dataset = FileDataset(
            file_paths=file_paths,
            dense_features=self.dense_features,
            sparse_features=self.sparse_features,
            sequence_features=self.sequence_features,
            target_columns=self.target_columns,
            id_columns=self.id_columns,
            chunk_size=chunk_size,
            file_type=file_type,
            processor=self.processor
        )
        
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    def _get_file_type(self, file_path: str) -> Literal['csv', 'parquet']:
        """
        Infer file type from extension.

        :param file_path: Path to a file.
        :returns: ``\"csv\"`` or ``\"parquet\"``.
        :raises ValueError: If extension is unsupported.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return 'csv'
        elif ext == '.parquet':
            return 'parquet'
        else:
            raise ValueError(f"Unsupported file type: {ext}")

def _normalize_sequence_column(column, feature: SequenceFeature) -> np.ndarray:
    """
    Normalize a raw sequence column into a padded int64 ``ndarray``.

    :param column: Sequence column from DataFrame/dict; can be Series, list, or ndarray.
    :param feature: Sequence feature definition providing ``max_len`` and optional ``padding_idx``.
    :returns: 2-D numpy array (batch, seq_len) with dtype ``int64``.
    """
    if isinstance(column, pd.Series):
        column = column.tolist()

    if isinstance(column, list):
        column = np.array(column, dtype=object)

    if isinstance(column, np.ndarray):
        if column.dtype == object and len(column) > 0 and isinstance(column[0], (list, tuple, np.ndarray)):
            try:
                column = np.stack([np.asarray(seq, dtype=np.int64) for seq in column])
            except Exception:
                sequences = []
                max_len = getattr(feature, "max_len", 0)
                for seq in column:
                    arr = np.asarray(seq, dtype=np.int64) if isinstance(seq, (list, tuple, np.ndarray)) else np.array([], dtype=np.int64)
                    sequences.append(arr)
                if max_len == 0:
                    max_len = max((len(seq) for seq in sequences), default=1)
                padded = []
                for seq in sequences:
                    if len(seq) > max_len:
                        padded.append(seq[:max_len])
                    else:
                        pad_value = getattr(feature, "padding_idx", 0)
                        padded.append(np.pad(seq, (0, max_len - len(seq)), constant_values=pad_value))
                column = np.stack(padded)
        elif column.ndim == 1:
            column = column.reshape(-1, 1)

    return np.asarray(column, dtype=np.int64)


def _build_tensors_from_data(  # noqa: C901
    data: dict | pd.DataFrame,
    raw_data: dict | pd.DataFrame,
    features: list,
    target_columns: list[str],
    id_columns: list[str],
    *,
    on_missing_feature: Literal["warn", "raise"] = "raise",
) -> tuple | None:
    """
    Shared routine to convert structured data into a tuple of tensors.

    :param data: Preprocessed data (dict or DataFrame) used to fetch model inputs/labels.
    :param raw_data: Original data, used for untouched ID columns.
    :param features: Ordered list of feature definitions.
    :param target_columns: Target/label column names.
    :param id_columns: Extra ID column names to append at the end of the tensor tuple.
    :param on_missing_feature: ``\"warn\"`` to skip missing feature with warning, ``\"raise\"`` to error.
    :returns: Tuple of tensors following the order of ``features`` + targets (+ ids) or ``None`` if empty.
    """
    tensors: list[torch.Tensor] = []

    for feature in features:
        column = get_column_data(data, feature.name)
        if column is None:
            if on_missing_feature == "warn":
                logging.warning(colorize(f"Feature column '{feature.name}' not found in data", "yellow"))
                continue
            raise AssertionError(f"Feature column {feature.name} not found in data.")

        if isinstance(feature, SequenceFeature):
            tensor = torch.from_numpy(_normalize_sequence_column(column, feature))
        elif isinstance(feature, DenseFeature):
            tensor = torch.from_numpy(np.asarray(column, dtype=np.float32))
        else:
            tensor = torch.from_numpy(np.asarray(column, dtype=np.int64))

        tensors.append(tensor)

    label_tensors = []
    for target_name in target_columns:
        column = get_column_data(data, target_name)
        if column is None:
            continue

        label_tensor = torch.from_numpy(np.asarray(column, dtype=np.float32))

        if label_tensor.dim() == 1:
            label_tensor = label_tensor.view(-1, 1)
        elif label_tensor.dim() == 2 and label_tensor.shape[0] == 1 and label_tensor.shape[1] > 1:
            label_tensor = label_tensor.t()

        label_tensors.append(label_tensor)

    if label_tensors:
        if len(label_tensors) == 1 and label_tensors[0].shape[1] > 1:
            y_tensor = label_tensors[0]
        else:
            y_tensor = torch.cat(label_tensors, dim=1)

        if y_tensor.shape[1] == 1:
            y_tensor = y_tensor.squeeze(1)

        tensors.append(y_tensor)

    if id_columns:
        id_arrays = []
        for id_col in id_columns:
            column = get_column_data(raw_data, id_col)
            if column is None:
                column = get_column_data(data, id_col)
            if column is None:
                raise KeyError(f"ID column '{id_col}' not found in provided data.")
            try:
                id_arr = np.asarray(column, dtype=np.int64)
            except Exception as exc:
                raise TypeError(
                    f"ID column '{id_col}' must contain numeric values. "
                    f"Received dtype={np.asarray(column).dtype}, error: {exc}"
                ) from exc
            id_arrays.append(id_arr)

        combined_ids = np.column_stack(id_arrays)
        tensors.append(torch.from_numpy(combined_ids))

    if not tensors:
        return None

    return tuple(tensors)
