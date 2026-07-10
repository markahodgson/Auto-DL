"""
Model generator module for Auto-DL.

Automatically generates and optimizes neural network architectures for different tasks.
"""

import tensorflow as tf
from tensorflow import keras
from typing import List, Optional, Tuple
import numpy as np


class ModelGenerator:
    """
    Generates optimized neural network architectures automatically.
    
    Creates different model architectures based on data characteristics
    and task requirements, optimizing for performance and efficiency.
    """
    
    def __init__(
        self,
        task_type: str = "classification",
        max_layers: int = 5,
        activation: str = "relu",
        optimizer: str = "adam"
    ):
        """
        Initialize the ModelGenerator.
        
        Args:
            task_type: Type of task ("classification" or "regression")
            max_layers: Maximum number of hidden layers
            activation: Activation function for hidden layers
            optimizer: Optimizer to use for training
        """
        self.task_type = task_type
        self.max_layers = max_layers
        self.activation = activation
        self.optimizer = optimizer
        self.model = None
        
    def create_model(
        self,
        input_shape: Tuple[int, ...],
        num_classes: Optional[int] = None,
        hidden_units: Optional[List[int]] = None,
        dropout_rate: float = 0.3
    ) -> keras.Model:
        """
        Create a neural network model with the specified architecture.
        
        Args:
            input_shape: Shape of input data
            num_classes: Number of output classes (for classification)
            hidden_units: List of hidden layer sizes. If None, automatically determined.
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        # Automatically determine hidden units if not provided
        if hidden_units is None:
            hidden_units = self._auto_generate_architecture(input_shape[0])
        
        # Build model
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape))
        
        # Add hidden layers
        for i, units in enumerate(hidden_units):
            model.add(keras.layers.Dense(units, activation=self.activation))
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(dropout_rate))
        
        # Add output layer
        if self.task_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be specified for classification")
            if num_classes == 2:
                model.add(keras.layers.Dense(1, activation="sigmoid"))
                loss = "binary_crossentropy"
                metrics = ["accuracy"]
            else:
                model.add(keras.layers.Dense(num_classes, activation="softmax"))
                loss = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]
        else:  # regression
            model.add(keras.layers.Dense(1))
            loss = "mse"
            metrics = ["mae"]
        
        # Compile model
        model.compile(
            optimizer=self.optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        return model
    
    def _auto_generate_architecture(self, input_dim: int) -> List[int]:
        """
        Automatically generate a reasonable architecture based on input dimension.
        
        Args:
            input_dim: Dimension of input features
            
        Returns:
            List of hidden layer sizes
        """
        # Simple heuristic: start with larger layers and gradually decrease
        architecture = []
        current_size = max(64, input_dim * 2)
        
        for _ in range(min(self.max_layers, 3)):
            architecture.append(int(current_size))
            current_size = max(16, current_size // 2)
        
        return architecture
    
    def generate_cnn_model(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        num_conv_layers: int = 3,
        filters: Optional[List[int]] = None,
        dropout_rate: float = 0.3
    ) -> keras.Model:
        """
        Generate a Convolutional Neural Network for image data.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
            num_conv_layers: Number of convolutional layers
            filters: List of filter sizes for each conv layer
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled CNN model
        """
        if filters is None:
            filters = [32 * (2 ** i) for i in range(num_conv_layers)]
        
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=input_shape))
        
        # Convolutional layers
        for f in filters:
            model.add(keras.layers.Conv2D(f, (3, 3), activation=self.activation, padding="same"))
            model.add(keras.layers.MaxPooling2D((2, 2)))
            if dropout_rate > 0:
                model.add(keras.layers.Dropout(dropout_rate))
        
        # Dense layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation=self.activation))
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        if num_classes == 2:
            model.add(keras.layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(keras.layers.Dense(num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
        
        model.compile(
            optimizer=self.optimizer,
            loss=loss,
            metrics=["accuracy"]
        )
        
        self.model = model
        return model
    
    def get_model(self) -> Optional[keras.Model]:
        """Get the current model."""
        return self.model
