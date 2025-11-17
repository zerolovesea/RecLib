"""
Unit Tests for DataProcessor

This module contains unit tests for the DataProcessor class which handles:
- Numeric feature preprocessing (scaling, filling missing values)
- Sparse feature encoding (hash, label encoding)
- Sequence feature processing (padding, truncation, encoding)
- Target feature encoding
- Save/load functionality
"""
import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from recforge.data.preprocessor import DataProcessor

logger = logging.getLogger(__name__)


class TestDataProcessorNumericFeatures:
    """Test suite for numeric feature processing"""
    
    @pytest.fixture
    def sample_numeric_data(self):
        """Create sample data with numeric features"""
        return pd.DataFrame({
            'age': [25, 30, 35, np.nan, 45],
            'price': [100.5, 200.3, 150.0, 300.2, 250.8],
            'score': [0.5, 0.8, np.nan, 0.9, 0.7]
        })
    
    def test_add_numeric_feature_standard(self, sample_numeric_data):
        """Test adding numeric feature with standard scaler"""
        logger.info("=" * 80)
        logger.info("Testing add_numeric_feature with standard scaler")
        logger.info("=" * 80)
        
        processor = DataProcessor()
        processor.add_numeric_feature('age', scaler='standard')
        processor.add_numeric_feature('price', scaler='standard')
        
        assert 'age' in processor.numeric_features
        assert 'price' in processor.numeric_features
        assert processor.numeric_features['age']['scaler'] == 'standard'
        
        logger.info("Add numeric feature test successful")
    
    def test_numeric_feature_fit_transform(self, sample_numeric_data):
        """Test fit and transform for numeric features"""
        logger.info("=" * 80)
        logger.info("Testing numeric feature fit and transform")
        logger.info("=" * 80)
        
        processor = DataProcessor()
        processor.add_numeric_feature('age', scaler='standard')
        processor.add_numeric_feature('price', scaler='minmax')
        processor.add_numeric_feature('score', scaler='none')
        
        # Fit
        processor.fit(sample_numeric_data)
        assert processor.is_fitted
        
        # Transform
        result = processor.transform(sample_numeric_data)
        
        assert 'age' in result
        assert 'price' in result
        assert 'score' in result
        assert len(result['age']) == len(sample_numeric_data)
        
        # Check no NaN values after transformation
        assert not np.isnan(result['age']).any()
        assert not np.isnan(result['price']).any()
        
        logger.info("Numeric feature fit/transform test successful")
    
    @pytest.mark.parametrize("scaler_type", ['standard', 'minmax', 'robust', 'maxabs', 'log', 'none'])
    def test_different_scalers(self, scaler_type):
        """Test different scaler types"""
        logger.info("=" * 80)
        logger.info(f"Testing {scaler_type} scaler")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'feature': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        processor = DataProcessor()
        processor.add_numeric_feature('feature', scaler=scaler_type)
        processor.fit(data)
        result = processor.transform(data)
        
        assert 'feature' in result
        assert len(result['feature']) == 5
        
        logger.info(f"{scaler_type} scaler test successful")
    
    def test_numeric_fill_na_custom(self):
        """Test custom fill_na value for numeric features"""
        logger.info("=" * 80)
        logger.info("Testing custom fill_na for numeric features")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'feature': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        processor = DataProcessor()
        processor.add_numeric_feature('feature', scaler='none', fill_na=0.0)
        processor.fit(data)
        result = processor.transform(data)
        
        # Check that NaN values are filled
        assert not np.isnan(result['feature']).any()
        
        logger.info("Custom fill_na test successful")


class TestDataProcessorSparseFeatures:
    """Test suite for sparse feature processing"""
    
    @pytest.fixture
    def sample_sparse_data(self):
        """Create sample data with sparse features"""
        return pd.DataFrame({
            'user_id': ['user_1', 'user_2', 'user_3', 'user_1', 'user_4'],
            'item_id': ['item_a', 'item_b', 'item_c', 'item_a', None],
            'category': ['cat1', 'cat2', 'cat1', 'cat3', 'cat2']
        })
    
    def test_add_sparse_feature_label(self, sample_sparse_data):
        """Test adding sparse feature with label encoding"""
        logger.info("=" * 80)
        logger.info("Testing add_sparse_feature with label encoding")
        logger.info("=" * 80)
        
        processor = DataProcessor()
        processor.add_sparse_feature('user_id', encode_method='label')
        processor.add_sparse_feature('category', encode_method='label')
        
        assert 'user_id' in processor.sparse_features
        assert processor.sparse_features['user_id']['encode_method'] == 'label'
        
        processor.fit(sample_sparse_data)
        result = processor.transform(sample_sparse_data)
        
        assert 'user_id' in result
        assert 'category' in result
        assert len(result['user_id']) == len(sample_sparse_data)
        
        logger.info("Label encoding test successful")
    
    def test_add_sparse_feature_hash(self, sample_sparse_data):
        """Test adding sparse feature with hash encoding"""
        logger.info("=" * 80)
        logger.info("Testing add_sparse_feature with hash encoding")
        logger.info("=" * 80)
        
        processor = DataProcessor()
        processor.add_sparse_feature('user_id', encode_method='hash', hash_size=1000)
        processor.add_sparse_feature('item_id', encode_method='hash', hash_size=500)
        
        processor.fit(sample_sparse_data)
        result = processor.transform(sample_sparse_data)
        
        assert 'user_id' in result
        assert 'item_id' in result
        
        # Check hash values are within range
        assert np.all(result['user_id'] >= 0)
        assert np.all(result['user_id'] < 1000)
        assert np.all(result['item_id'] >= 0)
        assert np.all(result['item_id'] < 500)
        
        logger.info("Hash encoding test successful")
    
    def test_sparse_feature_missing_values(self):
        """Test handling missing values in sparse features"""
        logger.info("=" * 80)
        logger.info("Testing missing value handling in sparse features")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'user_id': ['user_1', None, 'user_3', None, 'user_5']
        })
        
        processor = DataProcessor()
        processor.add_sparse_feature('user_id', encode_method='label', fill_na='<UNK>')
        processor.fit(data)
        result = processor.transform(data)
        
        assert 'user_id' in result
        assert len(result['user_id']) == 5
        
        logger.info("Missing value handling test successful")
    
    def test_get_vocab_sizes(self, sample_sparse_data):
        """Test getting vocabulary sizes"""
        logger.info("=" * 80)
        logger.info("Testing get_vocab_sizes")
        logger.info("=" * 80)
        
        processor = DataProcessor()
        processor.add_sparse_feature('user_id', encode_method='label')
        processor.add_sparse_feature('item_id', encode_method='hash', hash_size=1000)
        
        processor.fit(sample_sparse_data)
        vocab_sizes = processor.get_vocab_sizes()
        
        assert 'user_id' in vocab_sizes
        assert 'item_id' in vocab_sizes
        assert vocab_sizes['item_id'] == 1000  # Hash size
        assert vocab_sizes['user_id'] > 0  # Label encoded vocab size
        
        logger.info(f"Vocab sizes: {vocab_sizes}")
        logger.info("get_vocab_sizes test successful")


class TestDataProcessorSequenceFeatures:
    """Test suite for sequence feature processing"""
    
    @pytest.fixture
    def sample_sequence_data(self):
        """Create sample data with sequence features"""
        return pd.DataFrame({
            'item_history': [
                'item1,item2,item3',
                'item4,item5',
                'item1,item6,item7,item8,item9',
                'item2',
                'item3,item4,item5,item6'
            ],
            'category_seq': [
                'cat1,cat2',
                'cat3',
                'cat1,cat2,cat3',
                'cat4,cat5',
                'cat2,cat3'
            ]
        })
    
    def test_add_sequence_feature_label(self, sample_sequence_data):
        """Test adding sequence feature with label encoding"""
        logger.info("=" * 80)
        logger.info("Testing add_sequence_feature with label encoding")
        logger.info("=" * 80)
        
        processor = DataProcessor()
        processor.add_sequence_feature(
            'item_history',
            encode_method='label',
            max_len=10,
            pad_value=0,
            separator=','
        )
        
        assert 'item_history' in processor.sequence_features
        assert processor.sequence_features['item_history']['max_len'] == 10
        
        processor.fit(sample_sequence_data)
        result = processor.transform(sample_sequence_data)
        
        assert 'item_history' in result
        assert result['item_history'].shape[0] == len(sample_sequence_data)
        assert result['item_history'].shape[1] == 10  # max_len
        
        logger.info("Sequence label encoding test successful")
    
    def test_add_sequence_feature_hash(self, sample_sequence_data):
        """Test adding sequence feature with hash encoding"""
        logger.info("=" * 80)
        logger.info("Testing add_sequence_feature with hash encoding")
        logger.info("=" * 80)
        
        processor = DataProcessor()
        processor.add_sequence_feature(
            'item_history',
            encode_method='hash',
            hash_size=5000,
            max_len=20,
            pad_value=0,
            separator=','
        )
        
        processor.fit(sample_sequence_data)
        result = processor.transform(sample_sequence_data)
        
        assert 'item_history' in result
        assert result['item_history'].shape[1] == 20
        
        # Check hash values are within range (excluding pad_value)
        non_zero_mask = result['item_history'] != 0
        if non_zero_mask.any():
            non_zero_values = result['item_history'][non_zero_mask]
            assert np.all(non_zero_values > 0)
            assert np.all(non_zero_values <= 5000)
        
        logger.info("Sequence hash encoding test successful")
    
    @pytest.mark.parametrize("truncate", ['pre', 'post'])
    def test_sequence_truncation(self, truncate):
        """Test sequence truncation strategies"""
        logger.info("=" * 80)
        logger.info(f"Testing sequence truncation: {truncate}")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'seq': ['a,b,c,d,e,f,g,h,i,j']  # 10 items
        })
        
        processor = DataProcessor()
        processor.add_sequence_feature(
            'seq',
            encode_method='label',
            max_len=5,
            truncate=truncate,
            separator=','
        )
        
        processor.fit(data)
        result = processor.transform(data)
        
        assert result['seq'].shape[1] == 5
        
        logger.info(f"Sequence truncation ({truncate}) test successful")
    
    def test_sequence_padding(self):
        """Test sequence padding"""
        logger.info("=" * 80)
        logger.info("Testing sequence padding")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'seq': ['a,b', 'c', 'd,e,f']
        })
        
        processor = DataProcessor()
        processor.add_sequence_feature(
            'seq',
            encode_method='label',
            max_len=5,
            pad_value=0,
            separator=','
        )
        
        processor.fit(data)
        result = processor.transform(data)
        
        assert result['seq'].shape == (3, 5)
        
        # Check padding values
        assert np.all(result['seq'][:, -2:] >= 0)  # Last positions may be padded
        
        logger.info("Sequence padding test successful")


class TestDataProcessorTargetFeatures:
    """Test suite for target feature processing"""
    
    def test_add_binary_target(self):
        """Test adding binary target"""
        logger.info("=" * 80)
        logger.info("Testing add_target for binary classification")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'label': [0, 1, 1, 0, 1]
        })
        
        processor = DataProcessor()
        processor.add_target('label', target_type='binary')
        
        assert 'label' in processor.target_features
        assert processor.target_features['label']['target_type'] == 'binary'
        
        processor.fit(data)
        result = processor.transform(data)
        
        assert 'label' in result
        assert len(result['label']) == 5
        
        logger.info("Binary target test successful")
    
    def test_add_multiclass_target(self):
        """Test adding multiclass target"""
        logger.info("=" * 80)
        logger.info("Testing add_target for multiclass classification")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'A', 'B']
        })
        
        processor = DataProcessor()
        processor.add_target('category', target_type='multiclass')
        
        processor.fit(data)
        result = processor.transform(data)
        
        assert 'category' in result
        assert len(result['category']) == 5
        
        logger.info("Multiclass target test successful")
    
    def test_target_label_map(self):
        """Test target with custom label mapping"""
        logger.info("=" * 80)
        logger.info("Testing target with custom label mapping")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'sentiment': ['positive', 'negative', 'positive', 'neutral', 'negative']
        })
        
        label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
        
        processor = DataProcessor()
        processor.add_target('sentiment', target_type='multiclass', label_map=label_map)
        
        processor.fit(data)
        result = processor.transform(data)
        
        assert 'sentiment' in result
        
        logger.info("Target label mapping test successful")


class TestDataProcessorSaveLoad:
    """Test suite for save/load functionality"""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading processor"""
        logger.info("=" * 80)
        logger.info("Testing save and load functionality")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'user_id': ['u1', 'u2', 'u3', 'u4', 'u5'],
            'history': ['a,b', 'c,d', 'e,f', 'g,h', 'i,j'],
            'label': [0, 1, 1, 0, 1]
        })
        
        # Create and fit processor
        processor = DataProcessor()
        processor.add_numeric_feature('age', scaler='standard')
        processor.add_sparse_feature('user_id', encode_method='label')
        processor.add_sequence_feature('history', encode_method='label', max_len=5, separator=',')
        processor.add_target('label', target_type='binary')
        
        processor.fit(data)
        original_result = processor.transform(data)
        
        # Save processor
        save_path = tmp_path / "processor.pkl"
        processor.save(str(save_path))
        
        assert save_path.exists()
        logger.info(f"Processor saved to {save_path}")
        
        # Load processor
        loaded_processor = DataProcessor.load(str(save_path))
        
        assert loaded_processor.is_fitted
        loaded_result = loaded_processor.transform(data)
        
        # Compare results
        for key in original_result.keys():
            assert key in loaded_result
            np.testing.assert_array_almost_equal(
                original_result[key], 
                loaded_result[key],
                decimal=5
            )
        
        logger.info("Save and load test successful")
    
    def test_save_and_load_with_vocab_sizes(self, tmp_path):
        """Test that vocab sizes are preserved after save/load"""
        logger.info("=" * 80)
        logger.info("Testing vocab sizes preservation after save/load")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3', 'u4', 'u5'],
            'item_id': ['i1', 'i2', 'i3', 'i1', 'i2']
        })
        
        processor = DataProcessor()
        processor.add_sparse_feature('user_id', encode_method='label')
        processor.add_sparse_feature('item_id', encode_method='hash', hash_size=1000)
        
        processor.fit(data)
        original_vocab_sizes = processor.get_vocab_sizes()
        
        # Save and load
        save_path = tmp_path / "processor_vocab.pkl"
        processor.save(str(save_path))
        loaded_processor = DataProcessor.load(str(save_path))
        
        loaded_vocab_sizes = loaded_processor.get_vocab_sizes()
        
        assert original_vocab_sizes == loaded_vocab_sizes
        
        logger.info("Vocab sizes preservation test successful")


class TestDataProcessorIntegration:
    """Integration tests for DataProcessor"""
    
    def test_mixed_features_pipeline(self):
        """Test processing pipeline with mixed feature types"""
        logger.info("=" * 80)
        logger.info("Testing complete pipeline with mixed features")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'price': [100.0, 200.0, 150.0, 300.0, 250.0],
            'user_id': ['u1', 'u2', 'u3', 'u4', 'u5'],
            'category': ['cat1', 'cat2', 'cat1', 'cat3', 'cat2'],
            'item_history': ['i1,i2', 'i3,i4', 'i1,i5', 'i6,i7', 'i2,i3'],
            'label': [0, 1, 1, 0, 1]
        })
        
        processor = DataProcessor()
        
        # Add all types of features
        processor.add_numeric_feature('age', scaler='standard')
        processor.add_numeric_feature('price', scaler='minmax')
        processor.add_sparse_feature('user_id', encode_method='label')
        processor.add_sparse_feature('category', encode_method='hash', hash_size=100)
        processor.add_sequence_feature('item_history', encode_method='label', max_len=10, separator=',')
        processor.add_target('label', target_type='binary')
        
        # Fit and transform
        processor.fit(data)
        result = processor.transform(data)
        
        # Check all features are present
        assert 'age' in result
        assert 'price' in result
        assert 'user_id' in result
        assert 'category' in result
        assert 'item_history' in result
        assert 'label' in result
        
        # Check shapes
        assert len(result['age']) == 5
        assert len(result['price']) == 5
        assert len(result['user_id']) == 5
        assert len(result['category']) == 5
        assert result['item_history'].shape == (5, 10)
        assert len(result['label']) == 5
        
        logger.info("Mixed features pipeline test successful")
    
    def test_transform_with_return_dataframe(self):
        """Test transform returning DataFrame"""
        logger.info("=" * 80)
        logger.info("Testing transform with return_dict=False")
        logger.info("=" * 80)
        
        data = pd.DataFrame({
            'age': [25, 30, 35],
            'user_id': ['u1', 'u2', 'u3'],
            'label': [0, 1, 1]
        })
        
        processor = DataProcessor()
        processor.add_numeric_feature('age', scaler='standard')
        processor.add_sparse_feature('user_id', encode_method='label')
        processor.add_target('label', target_type='binary')
        
        processor.fit(data)
        result = processor.transform(data, return_dict=False)
        
        assert isinstance(result, pd.DataFrame)
        assert 'age' in result.columns
        assert 'user_id' in result.columns
        assert 'label' in result.columns
        
        logger.info("Transform DataFrame return test successful")
    
    def test_processor_summary(self):
        """Test processor summary method"""
        logger.info("=" * 80)
        logger.info("Testing processor summary")
        logger.info("=" * 80)
        
        processor = DataProcessor()
        processor.add_numeric_feature('age', scaler='standard')
        processor.add_sparse_feature('user_id', encode_method='label')
        processor.add_sequence_feature('history', encode_method='hash', hash_size=5000, max_len=20)
        processor.add_target('label', target_type='binary')
        
        # Should not raise any errors
        processor.summary()
        
        logger.info("Processor summary test successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
