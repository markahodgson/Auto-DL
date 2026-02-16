"""
Main AutoDL class for automatic deep learning model generation.

Provides a high-level interface for automatic model generation, training,
and optimization.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, Tuple
import tensorflow as tf
from tensorflow import keras

from .data_preprocessor import DataPreprocessor
from .model_generator import ModelGenerator
from .hyperparameter_optimizer import HyperparameterOptimizer


class AutoDL:
    """
    Automatic Deep Learning model generator.
    
    This is the main interface for Auto-DL. It automatically handles data preprocessing,
    model architecture search, hyperparameter optimization, and model training to
    generate high-performance deep learning models.
    
    Example:
        >>> from autodl import AutoDL
        >>> auto_dl = AutoDL(task_type="classification")
        >>> model = auto_dl.fit(X_train, y_train)
        >>> predictions = auto_dl.predict(X_test)
    """
    
    def __init__(
        self,
        task_type: str = "classification",
        optimize_hyperparameters: bool = True,
        max_trials: int = 10,
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: int = 1
    ):
        """
        Initialize AutoDL.
        
        Args:
            task_type: Type of task ("classification" or "regression")
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            max_trials: Maximum number of optimization trials
            test_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0, 1, or 2)
        """
        self.task_type = task_type
        self.optimize_hyperparameters = optimize_hyperparameters
        self.max_trials = max_trials
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize components
        self.preprocessor = DataPreprocessor(test_size=test_size, random_state=random_state)
        self.model_generator = ModelGenerator(task_type=task_type)
        self.hyperparameter_optimizer = None
        
        self.model = None
        self.best_params = None
        self.training_history = None
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        validation_data: Optional[Tuple] = None,
        epochs: int = 100,
        batch_size: int = 32
    ) -> keras.Model:
        """
        Fit the automatic deep learning model.
        
        This method automatically:
        1. Preprocesses the data
        2. Generates an optimal model architecture
        3. Optimizes hyperparameters (if enabled)
        4. Trains the final model
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional validation data tuple (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Trained Keras model
        """
        if self.verbose > 0:
            print("=" * 60)
            print("Starting AutoDL Training Pipeline")
            print("=" * 60)
            print(f"Task type: {self.task_type}")
            print(f"Optimize hyperparameters: {self.optimize_hyperparameters}")
            print(f"Input shape: {X.shape}")
            print()
        
        # Step 1: Preprocess data
        if self.verbose > 0:
            print("Step 1: Preprocessing data...")
        
        X_train, X_val, y_train, y_val = self.preprocessor.preprocess(
            X, y, task_type=self.task_type
        )
        
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = self.preprocessor.transform(X_val)
        
        if self.verbose > 0:
            print(f"Training set size: {X_train.shape[0]}")
            print(f"Validation set size: {X_val.shape[0]}")
            print()
        
        # Step 2: Determine model parameters
        input_shape = (X_train.shape[1],)
        num_classes = None
        if self.task_type == "classification":
            num_classes = self.preprocessor.get_num_classes()
            if self.verbose > 0:
                print(f"Number of classes: {num_classes}")
        
        # Step 3: Hyperparameter optimization (if enabled)
        if self.optimize_hyperparameters:
            if self.verbose > 0:
                print("\nStep 2: Optimizing hyperparameters...")
            
            self.hyperparameter_optimizer = HyperparameterOptimizer(
                search_method="random",
                max_trials=self.max_trials
            )
            
            def model_builder(params):
                """Build model with given hyperparameters."""
                # Update optimizer if specified
                if "optimizer" in params:
                    self.model_generator.optimizer = params["optimizer"]
                
                # Build model
                model = self.model_generator.create_model(
                    input_shape=input_shape,
                    num_classes=num_classes,
                    hidden_units=params.get("hidden_units"),
                    dropout_rate=params.get("dropout_rate", 0.3)
                )
                
                # Update learning rate if specified
                if "learning_rate" in params:
                    model.optimizer.learning_rate.assign(params["learning_rate"])
                
                return model
            
            self.best_params = self.hyperparameter_optimizer.optimize(
                model_builder,
                X_train, y_train,
                X_val, y_val
            )
            
            if self.verbose > 0:
                print("\nBest hyperparameters found:")
                for key, value in self.best_params.items():
                    print(f"  {key}: {value}")
                print()
        
        # Step 4: Train final model
        if self.verbose > 0:
            print("\nStep 3: Training final model...")
        
        # Use optimized parameters if available
        final_params = self.best_params if self.best_params else {}
        
        if "optimizer" in final_params:
            self.model_generator.optimizer = final_params["optimizer"]
        
        self.model = self.model_generator.create_model(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_units=final_params.get("hidden_units"),
            dropout_rate=final_params.get("dropout_rate", 0.3)
        )
        
        if "learning_rate" in final_params:
            self.model.optimizer.learning_rate.assign(final_params["learning_rate"])
        
        # Setup callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Train model
        final_epochs = final_params.get("epochs", epochs)
        final_batch_size = final_params.get("batch_size", batch_size)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=final_epochs,
            batch_size=final_batch_size,
            callbacks=callbacks,
            verbose=1 if self.verbose > 1 else 0
        )
        
        self.training_history = history.history
        
        # Evaluate final model
        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("Training Complete!")
            print("=" * 60)
            val_loss = min(history.history["val_loss"])
            print(f"Final validation loss: {val_loss:.4f}")
            
            if self.task_type == "classification":
                val_acc = max(history.history.get("val_accuracy", [0]))
                print(f"Final validation accuracy: {val_acc:.4f}")
            print()
        
        return self.model
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        X_transformed = self.preprocessor.transform(X)
        predictions = self.model.predict(X_transformed, verbose=0)
        
        # For binary classification, threshold at 0.5
        if self.task_type == "classification" and predictions.shape[1] == 1:
            predictions = (predictions > 0.5).astype(int)
        # For multi-class classification, take argmax
        elif self.task_type == "classification":
            predictions = np.argmax(predictions, axis=1)
        
        return predictions
    
    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        X_transformed = self.preprocessor.transform(X)
        
        if self.task_type == "classification":
            y_encoded = self.preprocessor.label_encoder.transform(y)
        else:
            y_encoded = y
        
        results = self.model.evaluate(X_transformed, y_encoded, verbose=0)
        
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        self.model.save(filepath)
        if self.verbose > 0:
            print(f"Model saved to {filepath}")
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the model architecture.
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
