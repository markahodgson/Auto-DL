"""
Unit tests for the ModelGenerator module.
"""

import unittest
import numpy as np
from src.autodl.model_generator import ModelGenerator


class TestModelGenerator(unittest.TestCase):
    """Test cases for ModelGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = ModelGenerator(task_type="classification")
    
    def test_initialization(self):
        """Test ModelGenerator initialization."""
        self.assertEqual(self.generator.task_type, "classification")
        self.assertEqual(self.generator.max_layers, 5)
        self.assertEqual(self.generator.activation, "relu")
        self.assertEqual(self.generator.optimizer, "adam")
        self.assertIsNone(self.generator.model)
    
    def test_create_classification_model(self):
        """Test creating a classification model."""
        model = self.generator.create_model(
            input_shape=(10,),
            num_classes=3,
            hidden_units=[64, 32]
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 6)  # input + 2 hidden + 2 dropout + output
        self.assertEqual(model.input_shape, (None, 10))
    
    def test_create_regression_model(self):
        """Test creating a regression model."""
        generator = ModelGenerator(task_type="regression")
        model = generator.create_model(
            input_shape=(20,),
            hidden_units=[128, 64, 32]
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 20))
        self.assertEqual(model.output_shape, (None, 1))
    
    def test_auto_generate_architecture(self):
        """Test automatic architecture generation."""
        architecture = self.generator._auto_generate_architecture(10)
        
        self.assertIsInstance(architecture, list)
        self.assertGreater(len(architecture), 0)
        self.assertLessEqual(len(architecture), 3)
        
        # Check that sizes decrease
        for i in range(len(architecture) - 1):
            self.assertGreaterEqual(architecture[i], architecture[i + 1])
    
    def test_create_model_without_hidden_units(self):
        """Test creating model with automatic architecture."""
        model = self.generator.create_model(
            input_shape=(15,),
            num_classes=2
        )
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(self.generator.get_model())
    
    def test_binary_classification(self):
        """Test binary classification model."""
        model = self.generator.create_model(
            input_shape=(10,),
            num_classes=2,
            hidden_units=[32]
        )
        
        # Binary classification should have 1 output with sigmoid
        self.assertEqual(model.output_shape, (None, 1))
    
    def test_multiclass_classification(self):
        """Test multi-class classification model."""
        model = self.generator.create_model(
            input_shape=(10,),
            num_classes=5,
            hidden_units=[32]
        )
        
        # Multi-class should have num_classes outputs with softmax
        self.assertEqual(model.output_shape, (None, 5))
    
    def test_cnn_model_generation(self):
        """Test CNN model generation for image data."""
        model = self.generator.generate_cnn_model(
            input_shape=(28, 28, 1),
            num_classes=10,
            num_conv_layers=2,
            filters=[32, 64]
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 28, 28, 1))
        self.assertEqual(model.output_shape, (None, 10))


if __name__ == "__main__":
    unittest.main()
