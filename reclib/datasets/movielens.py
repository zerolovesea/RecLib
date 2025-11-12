"""
MovieLens dataset loaders.

MovieLens is a collection of movie rating datasets collected by the GroupLens Research Project.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from .base import BaseDataset, DatasetConfig
from .registry import register_dataset


@register_dataset("movielens-100k")
class MovieLens100K(BaseDataset):
    """
    MovieLens 100K dataset.
    
    Contains 100,000 ratings from 943 users on 1682 movies.
    Each user has rated at least 20 movies.
    
    Attributes:
        - user_id: User ID
        - item_id: Movie ID
        - rating: Rating (1-5)
        - timestamp: Timestamp
    """
    
    def _get_config(self) -> DatasetConfig:
        return DatasetConfig(
            name="movielens-100k",
            url="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
            filename="ml-100k.zip",
            md5="0e33842e24a9c977be4e0107933c0723",
            description="MovieLens 100K movie ratings dataset",
            citation=(
                "F. Maxwell Harper and Joseph A. Konstan. 2015. "
                "The MovieLens Datasets: History and Context. "
                "ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19."
            ),
            task_type="ranking",
            compressed_format="zip"
        )
    
    def _check_exists(self) -> bool:
        data_file = self.dataset_dir / "ml-100k" / "u.data"
        return data_file.exists()
    
    def _load_data(self, include_features=False) -> pd.DataFrame:
        """
        Load MovieLens 100K data.
        
        Args:
            include_features: If True, also load user and item features.
        
        Returns:
            DataFrame with ratings and optionally features.
        """
        data_file = self.dataset_dir / "ml-100k" / "u.data"
        
        # Load ratings
        df = pd.read_csv(
            data_file,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        if include_features:
            # Load user features
            user_file = self.dataset_dir / "ml-100k" / "u.user"
            users = pd.read_csv(
                user_file,
                sep='|',
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                engine='python'
            )
            df = df.merge(users, on='user_id', how='left')
            
            # Load item features
            item_file = self.dataset_dir / "ml-100k" / "u.item"
            items = pd.read_csv(
                item_file,
                sep='|',
                encoding='latin-1',
                names=[
                    'item_id', 'movie_title', 'release_date', 'video_release_date',
                    'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                    'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                    'Thriller', 'War', 'Western'
                ],
                engine='python'
            )
            df = df.merge(items, on='item_id', how='left')
        
        return df


@register_dataset("movielens-1m")
class MovieLens1M(BaseDataset):
    """
    MovieLens 1M dataset.
    
    Contains 1 million ratings from 6040 users on 3706 movies.
    """
    
    def _get_config(self) -> DatasetConfig:
        return DatasetConfig(
            name="movielens-1m",
            url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            filename="ml-1m.zip",
            md5="c4d9eecfca2ab87c1945afe126590906",
            description="MovieLens 1M movie ratings dataset",
            citation=(
                "F. Maxwell Harper and Joseph A. Konstan. 2015. "
                "The MovieLens Datasets: History and Context. "
                "ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19."
            ),
            task_type="ranking",
            compressed_format="zip"
        )
    
    def _check_exists(self) -> bool:
        data_file = self.dataset_dir / "ml-1m" / "ratings.dat"
        return data_file.exists()
    
    def _load_data(self, include_features=False) -> pd.DataFrame:
        """
        Load MovieLens 1M data.
        
        Args:
            include_features: If True, also load user and item features.
        
        Returns:
            DataFrame with ratings and optionally features.
        """
        data_file = self.dataset_dir / "ml-1m" / "ratings.dat"
        
        # Load ratings
        df = pd.read_csv(
            data_file,
            sep='::',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        if include_features:
            # Load user features
            user_file = self.dataset_dir / "ml-1m" / "users.dat"
            users = pd.read_csv(
                user_file,
                sep='::',
                names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                engine='python'
            )
            df = df.merge(users, on='user_id', how='left')
            
            # Load item features
            item_file = self.dataset_dir / "ml-1m" / "movies.dat"
            items = pd.read_csv(
                item_file,
                sep='::',
                names=['item_id', 'title', 'genres'],
                engine='python',
                encoding='latin-1'
            )
            df = df.merge(items, on='item_id', how='left')
        
        return df


@register_dataset("movielens-25m")
class MovieLens25M(BaseDataset):
    """
    MovieLens 25M dataset.
    
    Contains 25 million ratings and 1 million tag applications from 162,000 users on 62,000 movies.
    """
    
    def _get_config(self) -> DatasetConfig:
        return DatasetConfig(
            name="movielens-25m",
            url="https://files.grouplens.org/datasets/movielens/ml-25m.zip",
            filename="ml-25m.zip",
            md5="6b51fb2759a8657d3bfcbfc42b592ada",
            description="MovieLens 25M movie ratings dataset",
            citation=(
                "F. Maxwell Harper and Joseph A. Konstan. 2015. "
                "The MovieLens Datasets: History and Context. "
                "ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19."
            ),
            task_type="ranking",
            compressed_format="zip"
        )
    
    def _check_exists(self) -> bool:
        data_file = self.dataset_dir / "ml-25m" / "ratings.csv"
        return data_file.exists()
    
    def _load_data(self, include_features=False, sample_frac=None) -> pd.DataFrame:
        """
        Load MovieLens 25M data.
        
        Args:
            include_features: If True, also load item features.
            sample_frac: If provided, sample a fraction of the data (useful for large dataset).
        
        Returns:
            DataFrame with ratings and optionally features.
        """
        data_file = self.dataset_dir / "ml-25m" / "ratings.csv"
        
        # Load ratings
        df = pd.read_csv(data_file)
        df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        
        if sample_frac is not None and 0 < sample_frac < 1:
            df = df.sample(frac=sample_frac, random_state=2024)
        
        if include_features:
            # Load item features
            item_file = self.dataset_dir / "ml-25m" / "movies.csv"
            items = pd.read_csv(item_file)
            items.columns = ['item_id', 'title', 'genres']
            df = df.merge(items, on='item_id', how='left')
            
            # Load tags if needed
            # tag_file = self.dataset_dir / "ml-25m" / "tags.csv"
        
        return df
