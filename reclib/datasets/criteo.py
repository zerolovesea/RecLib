"""
Criteo dataset loaders.

Criteo datasets are widely used for CTR (Click-Through Rate) prediction tasks.
"""

import pandas as pd
from pathlib import Path
from .base import BaseDataset, DatasetConfig
from .registry import register_dataset


@register_dataset("criteo")
class CriteoDataset(BaseDataset):
    """
    Criteo Display Advertising Challenge dataset.
    
    This dataset contains feature values and click feedback for millions of display ads.
    Its purpose is to benchmark algorithms for clickthrough rate (CTR) prediction.
    
    The data consists of:
    - 13 integer features (mostly count features)
    - 26 categorical features
    - 1 binary target (0/1 for non-click/click)
    
    Note: Due to the large size (11GB uncompressed), this downloads a sample version by default.
    """
    
    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        force_download: bool = False,
        sample_size: int = 100000,
        use_sample: bool = True,
    ):
        """
        Initialize Criteo dataset.
        
        Args:
            root: Root directory to store the dataset.
            download: Whether to download the dataset if not found.
            force_download: Force re-download even if file exists.
            sample_size: Number of samples to use (only when use_sample=True).
            use_sample: If True, download/use a smaller sample dataset.
        """
        self.sample_size = sample_size
        self.use_sample = use_sample
        super().__init__(root, download, force_download)
    
    def _get_config(self) -> DatasetConfig:
        if self.use_sample:
            # Use Kaggle sample version (smaller, easier to download)
            return DatasetConfig(
                name="criteo",
                url="https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz",
                filename="criteo-sample.tar.gz",
                description="Criteo Display Advertising Challenge (Sample)",
                citation=(
                    "Criteo Labs. Criteo Display Advertising Challenge. "
                    "https://www.kaggle.com/c/criteo-display-ad-challenge"
                ),
                task_type="ranking",
                compressed_format="tar.gz"
            )
        else:
            return DatasetConfig(
                name="criteo",
                url="https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
                filename="criteo.csv.gz",
                description="Criteo Display Advertising Challenge (Full)",
                citation=(
                    "Criteo Labs. Criteo Display Advertising Challenge. "
                    "https://www.kaggle.com/c/criteo-display-ad-challenge"
                ),
                task_type="ranking",
                compressed_format="gz"
            )
    
    def _check_exists(self) -> bool:
        if self.use_sample:
            # Check if either the processed CSV or the raw train.txt exists
            return (self.dataset_dir / "criteo_sample.csv").exists() or \
                   (self.dataset_dir / "train.txt").exists()
        else:
            return (self.dataset_dir / "criteo.csv").exists()
    
    def _load_data(self, nrows=None) -> pd.DataFrame:
        """
        Load Criteo data.
        
        Args:
            nrows: Number of rows to load (for memory efficiency).
        
        Returns:
            DataFrame with features and target.
        """
        if self.use_sample:
            data_file = self.dataset_dir / "criteo_sample.csv"
            
            # If sample file doesn't exist, create it from downloaded data
            if not data_file.exists():
                # Try to read from extracted files
                train_file = self.dataset_dir / "train.txt"
                if train_file.exists():
                    print(f"Creating sample with {self.sample_size} rows...")
                    # Column names
                    int_cols = [f'I{i}' for i in range(1, 14)]
                    cat_cols = [f'C{i}' for i in range(1, 27)]
                    columns = ['label'] + int_cols + cat_cols
                    
                    # Read sample
                    df = pd.read_csv(
                        train_file,
                        sep='\t',
                        names=columns,
                        nrows=self.sample_size
                    )
                    df.to_csv(data_file, index=False)
                    print(f"Sample saved to {data_file}")
                else:
                    raise FileNotFoundError(
                        f"Neither {data_file} nor {train_file} found. "
                        "Please download the dataset first."
                    )
        else:
            data_file = self.dataset_dir / "criteo.csv"
        
        # Load data
        if nrows:
            df = pd.read_csv(data_file, nrows=nrows)
        else:
            df = pd.read_csv(data_file)
        
        return df
    
    def get_feature_names(self):
        """Get feature names for Criteo dataset."""
        int_cols = [f'I{i}' for i in range(1, 14)]
        cat_cols = [f'C{i}' for i in range(1, 27)]
        return {
            'label': 'label',
            'dense_features': int_cols,
            'sparse_features': cat_cols
        }
