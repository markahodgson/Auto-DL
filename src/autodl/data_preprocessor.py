"""
Data preprocessing module for Auto-DL.

Handles data loading, cleaning, normalization, and preparation for model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Union


class DataPreprocessor:
    """
    Preprocesses data for deep learning models.
    
    Handles various data types and automatically applies appropriate
    preprocessing techniques including normalization, encoding, and splitting.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the DataPreprocessor.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def preprocess(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        task_type: str = "classification"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data for model training.
        
        Args:
            X: Input features
            y: Target labels
            task_type: Type of task ("classification" or "regression")
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Ensure correct shape
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        # Normalize features
        X_normalized = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        # Encode labels for classification
        if task_type == "classification":
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y_encoded,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            X: Input features
            
        Returns:
            Normalized features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        return self.scaler.transform(X)
    
    def get_num_classes(self) -> int:
        """Get number of classes for classification tasks."""
        return len(self.label_encoder.classes_)
