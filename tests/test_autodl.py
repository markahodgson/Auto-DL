"""
Integration tests for the AutoDL class.
"""

import unittest
import numpy as np
from sklearn.datasets import load_iris, make_regression
from src.autodl.autodl import AutoDL


class TestAutoDL(unittest.TestCase):
    """Test cases for AutoDL class."""
    
    def setUp(self):
        """Set up test data."""
        # Classification data
        iris = load_iris()
        self.X_class = iris.data
        self.y_class = iris.target
        
        # Regression data
        self.X_reg, self.y_reg = make_regression(
            n_samples=200, n_features=10, random_state=42
        )
    
    def test_classification_without_optimization(self):
        """Test classification without hyperparameter optimization."""
        auto_dl = AutoDL(
            task_type="classification",
            optimize_hyperparameters=False,
            verbose=0
        )
        
        model = auto_dl.fit(self.X_class, self.y_class, epochs=10)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(auto_dl.model)
        
        # Test prediction
        predictions = auto_dl.predict(self.X_class[:10])
        self.assertEqual(predictions.shape[0], 10)
    
    def test_classification_with_optimization(self):
        """Test classification with hyperparameter optimization."""
        auto_dl = AutoDL(
            task_type="classification",
            optimize_hyperparameters=True,
            max_trials=2,  # Use few trials for testing
            verbose=0
        )
        
        model = auto_dl.fit(self.X_class, self.y_class, epochs=10)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(auto_dl.best_params)
        
        # Test evaluation
        metrics = auto_dl.evaluate(self.X_class, self.y_class)
        # Check that metrics are returned (name may vary by TF version)
        self.assertGreater(len(metrics), 0)
    
    def test_regression(self):
        """Test regression task."""
        auto_dl = AutoDL(
            task_type="regression",
            optimize_hyperparameters=False,
            verbose=0
        )
        
        model = auto_dl.fit(self.X_reg, self.y_reg, epochs=10)
        
        self.assertIsNotNone(model)
        
        # Test prediction
        predictions = auto_dl.predict(self.X_reg[:10])
        self.assertEqual(predictions.shape[0], 10)
        
        # Test evaluation
        metrics = auto_dl.evaluate(self.X_reg, self.y_reg)
        # Check that metrics are returned (name may vary by TF version)
        self.assertGreater(len(metrics), 0)
    
    def test_model_summary(self):
        """Test getting model summary."""
        auto_dl = AutoDL(task_type="classification", verbose=0)
        auto_dl.fit(self.X_class, self.y_class, epochs=5)
        
        summary = auto_dl.get_model_summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
    
    def test_predict_before_fit(self):
        """Test that predict raises error before fitting."""
        auto_dl = AutoDL(task_type="classification")
        
        with self.assertRaises(ValueError):
            auto_dl.predict(self.X_class)
    
    def test_evaluate_before_fit(self):
        """Test that evaluate raises error before fitting."""
        auto_dl = AutoDL(task_type="classification")
        
        with self.assertRaises(ValueError):
            auto_dl.evaluate(self.X_class, self.y_class)


if __name__ == "__main__":
    unittest.main()
