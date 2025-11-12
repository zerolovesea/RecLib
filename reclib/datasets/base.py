"""
Base dataset class and configuration for RecLib datasets.
"""

import os
import hashlib
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import urllib.request
import zipfile
import gzip
import tarfile
from tqdm import tqdm


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    
    name: str
    url: str
    filename: str
    md5: Optional[str] = None
    description: Optional[str] = None
    citation: Optional[str] = None
    task_type: str = "ranking"  # ranking, match, multitask
    compressed_format: Optional[str] = None  # zip, gz, tar.gz


class DownloadProgressBar(tqdm):
    """Progress bar for downloading files."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class BaseDataset(ABC):
    """Base class for all datasets in RecLib."""
    
    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        force_download: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory to store the dataset.
            download: Whether to download the dataset if not found.
            force_download: Force re-download even if file exists.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        
        self.config = self._get_config()
        self.dataset_dir = self.root / self.config.name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        self._data = None
        
        if force_download or (download and not self._check_exists()):
            self._download()
        
        if not self._check_exists():
            raise RuntimeError(
                f"Dataset not found. You can use download=True to download it."
            )
    
    @abstractmethod
    def _get_config(self) -> DatasetConfig:
        """Get dataset configuration."""
        pass
    
    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        """Load the dataset into a pandas DataFrame."""
        pass
    
    @abstractmethod
    def _check_exists(self) -> bool:
        """Check if the dataset files exist."""
        pass
    
    def _download(self):
        """Download and extract the dataset."""
        if self._check_exists():
            print(f"Dataset {self.config.name} already exists.")
            return
        
        download_path = self.dataset_dir / self.config.filename
        
        print(f"Downloading {self.config.name} dataset from {self.config.url}")
        
        try:
            with DownloadProgressBar(
                unit='B',
                unit_scale=True,
                miniters=1,
                desc=self.config.filename
            ) as t:
                urllib.request.urlretrieve(
                    self.config.url,
                    filename=download_path,
                    reporthook=t.update_to
                )
            
            # Verify MD5 if provided
            if self.config.md5:
                if not self._verify_md5(download_path, self.config.md5):
                    raise RuntimeError("MD5 verification failed!")
            
            # Extract if compressed
            if self.config.compressed_format:
                print(f"Extracting {self.config.filename}...")
                self._extract_file(download_path)
                
            print(f"Download complete! Dataset saved to {self.dataset_dir}")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            if download_path.exists():
                download_path.unlink()
            raise
    
    def _extract_file(self, file_path: Path):
        """Extract compressed file."""
        if self.config.compressed_format == 'zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_dir)
        elif self.config.compressed_format == 'gz':
            output_path = file_path.with_suffix('')
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        elif self.config.compressed_format in ['tar.gz', 'tgz']:
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(self.dataset_dir)
        else:
            raise ValueError(f"Unsupported format: {self.config.compressed_format}")
    
    def _verify_md5(self, file_path: Path, expected_md5: str) -> bool:
        """Verify MD5 checksum of a file."""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest() == expected_md5
    
    def load(self, **kwargs) -> pd.DataFrame:
        """
        Load and return the dataset.
        
        Returns:
            DataFrame containing the dataset.
        """
        # Don't cache if kwargs are provided, as different kwargs may give different results
        if kwargs:
            return self._load_data(**kwargs)
        
        if self._data is None:
            self._data = self._load_data(**kwargs)
        return self._data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        df = self.load()
        stats = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "columns": list(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        }
        return stats
    
    def info(self):
        """Print dataset information."""
        print(f"\n{'='*60}")
        print(f"Dataset: {self.config.name}")
        print(f"{'='*60}")
        if self.config.description:
            print(f"Description: {self.config.description}")
        print(f"Task Type: {self.config.task_type}")
        print(f"Location: {self.dataset_dir}")
        
        stats = self.get_stats()
        print(f"\nStatistics:")
        print(f"  - Number of samples: {stats['n_samples']:,}")
        print(f"  - Number of features: {stats['n_features']}")
        print(f"  - Memory usage: {stats['memory_usage']:.2f} MB")
        print(f"  - Columns: {', '.join(stats['columns'])}")
        
        if self.config.citation:
            print(f"\nCitation:\n{self.config.citation}")
        print(f"{'='*60}\n")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(root='{self.root}')"
