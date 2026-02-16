"""
Unit tests for the DataPreprocessor module.
"""

import unittest
import numpy as np
import pandas as pd
from src.autodl.data_preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test data."""
        self.preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y_classification = np.random.randint(0, 3, 100)
        self.y_regression = np.random.randn(100)
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        self.assertEqual(self.preprocessor.test_size, 0.2)
        self.assertEqual(self.preprocessor.random_state, 42)
        self.assertFalse(self.preprocessor.is_fitted)
    
    def test_preprocess_classification(self):
        """Test preprocessing for classification tasks."""
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess(
            self.X, self.y_classification, task_type="classification"
        )
        
        # Check shapes
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
        
        # Check normalization (approximate due to small sample)
        self.assertAlmostEqual(np.mean(X_train), 0, places=2)
        self.assertAlmostEqual(np.std(X_train), 1, places=0)
        
        # Check fitted flag
        self.assertTrue(self.preprocessor.is_fitted)
    
    def test_preprocess_regression(self):
        """Test preprocessing for regression tasks."""
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess(
            self.X, self.y_regression, task_type="regression"
        )
        
        # Check shapes
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertTrue(self.preprocessor.is_fitted)
    
    def test_transform(self):
        """Test transform method."""
        # Fit preprocessor
        self.preprocessor.preprocess(
            self.X, self.y_classification, task_type="classification"
        )
        
        # Transform new data
        X_new = np.random.randn(10, 5)
        X_transformed = self.preprocessor.transform(X_new)
        
        self.assertEqual(X_transformed.shape, (10, 5))
    
    def test_transform_not_fitted(self):
        """Test that transform raises error when not fitted."""
        with self.assertRaises(ValueError):
            self.preprocessor.transform(self.X)
    
    def test_pandas_input(self):
        """Test preprocessing with pandas DataFrames."""
        X_df = pd.DataFrame(self.X)
        y_series = pd.Series(self.y_classification)
        
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess(
            X_df, y_series, task_type="classification"
        )
        
        self.assertIsInstance(X_train, np.ndarray)
        self.assertEqual(len(X_train), 80)
    
    def test_get_num_classes(self):
        """Test getting number of classes."""
        self.preprocessor.preprocess(
            self.X, self.y_classification, task_type="classification"
        )
        
        num_classes = self.preprocessor.get_num_classes()
        self.assertEqual(num_classes, 3)


if __name__ == "__main__":
    unittest.main()
