"""
Example: Regression with Auto-DL

This example demonstrates how to use Auto-DL for automatic
regression model generation using synthetic data.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autodl import AutoDL


def main():
    print("Auto-DL Regression Example")
    print("=" * 60)
    
    # Generate synthetic regression data
    print("\nGenerating synthetic regression dataset...")
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10,
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create AutoDL instance
    print("\nInitializing Auto-DL for regression...")
    auto_dl = AutoDL(
        task_type="regression",
        optimize_hyperparameters=True,
        max_trials=5,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    model = auto_dl.fit(X_train, y_train, epochs=50)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(auto_dl.get_model_summary())
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = auto_dl.evaluate(X_test, y_test)
    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Make predictions
    print("\nMaking predictions on test set...")
    predictions = auto_dl.predict(X_test[:5])
    print(f"First 5 predictions: {predictions.flatten()}")
    print(f"First 5 true values: {y_test[:5]}")
    
    # Save model
    print("\nSaving model...")
    auto_dl.save_model("regression_model.h5")
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
