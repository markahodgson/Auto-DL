"""
Hyperparameter optimization module for Auto-DL.

Automatically searches for optimal hyperparameters using various techniques.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import tensorflow as tf
from tensorflow import keras


class HyperparameterOptimizer:
    """
    Optimizes hyperparameters for neural network models.
    
    Uses grid search, random search, or Bayesian optimization to find
    the best hyperparameters for a given model and dataset.
    """
    
    def __init__(
        self,
        search_method: str = "random",
        max_trials: int = 10,
        patience: int = 3
    ):
        """
        Initialize the HyperparameterOptimizer.
        
        Args:
            search_method: Search method ("random" or "grid")
            max_trials: Maximum number of trials to run
            patience: Early stopping patience
        """
        self.search_method = search_method
        self.max_trials = max_trials
        self.patience = patience
        self.best_params = None
        self.best_score = float('-inf')
        self.history = []
        
    def define_search_space(self) -> Dict[str, List[Any]]:
        """
        Define the hyperparameter search space.
        
        Returns:
            Dictionary of hyperparameter names and their possible values
        """
        search_space = {
            "learning_rate": [0.001, 0.01, 0.0001],
            "batch_size": [16, 32, 64, 128],
            "epochs": [50, 100, 150],
            "hidden_units": [
                [64, 32],
                [128, 64, 32],
                [256, 128, 64],
                [512, 256, 128]
            ],
            "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
            "optimizer": ["adam", "sgd", "rmsprop"],
        }
        return search_space
    
    def sample_hyperparameters(
        self,
        search_space: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Sample a set of hyperparameters from the search space.
        
        Args:
            search_space: Dictionary of hyperparameter options
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        if self.search_method == "random":
            params = {}
            for key, values in search_space.items():
                params[key] = np.random.choice(values) if isinstance(values[0], (int, float, str)) else np.random.choice(len(values))
                if key == "hidden_units":
                    params[key] = values[params[key]]
            return params
        else:
            raise NotImplementedError(f"Search method {self.search_method} not implemented")
    
    def optimize(
        self,
        model_builder,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        custom_search_space: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for the given model and data.
        
        Args:
            model_builder: Function that builds and returns a model given hyperparameters
            X_train: Training data features
            y_train: Training data labels
            X_val: Validation data features
            y_val: Validation data labels
            custom_search_space: Custom search space (optional)
            
        Returns:
            Dictionary of best hyperparameters found
        """
        search_space = custom_search_space or self.define_search_space()
        
        print(f"Starting hyperparameter optimization with {self.max_trials} trials...")
        
        for trial in range(self.max_trials):
            # Sample hyperparameters
            params = self.sample_hyperparameters(search_space)
            
            print(f"\nTrial {trial + 1}/{self.max_trials}")
            print(f"Testing parameters: {params}")
            
            try:
                # Build and train model
                model = model_builder(params)
                
                # Create callbacks
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True
                )
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=params.get("epochs", 50),
                    batch_size=params.get("batch_size", 32),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate performance
                val_loss = min(history.history["val_loss"])
                score = -val_loss  # Negative loss as score (higher is better)
                
                print(f"Validation loss: {val_loss:.4f}")
                
                # Update best parameters
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    print(f"New best score: {score:.4f}")
                
                # Store history
                self.history.append({
                    "trial": trial,
                    "params": params,
                    "score": score,
                    "val_loss": val_loss
                })
                
            except Exception as e:
                print(f"Trial failed with error: {e}")
                continue
        
        print(f"\nOptimization complete!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")
        
        return self.best_params
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best hyperparameters found."""
        return self.best_params
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the optimization history."""
        return self.history
