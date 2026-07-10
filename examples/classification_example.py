"""
Example: Basic classification with Auto-DL

This example demonstrates how to use Auto-DL for automatic
classification model generation using the Iris dataset.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autodl import AutoDL


def main():
    print("Auto-DL Classification Example")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(set(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create AutoDL instance
    print("\nInitializing Auto-DL...")
    auto_dl = AutoDL(
        task_type="classification",
        optimize_hyperparameters=True,
        max_trials=5,  # Use more trials for better results
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    model = auto_dl.fit(X_train, y_train)
    
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
    print(f"First 5 true labels: {y_test[:5]}")
    
    # Save model
    print("\nSaving model...")
    auto_dl.save_model("iris_model.h5")
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
