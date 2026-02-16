# Auto-DL

Automatic Deep Learning Model Generator using TensorFlow

## Overview

Auto-DL is a Python library that automatically generates, optimizes, and trains high-performance deep learning models. It leverages TensorFlow and advanced machine learning techniques to handle the entire model development pipeline with minimal user input.

## Features

- **Automatic Model Generation**: Automatically creates neural network architectures based on your data characteristics
- **Hyperparameter Optimization**: Intelligently searches for optimal hyperparameters using random search
- **Data Preprocessing**: Handles data normalization, encoding, and splitting automatically
- **Multiple Task Types**: Supports both classification and regression tasks
- **Easy-to-Use API**: Simple, scikit-learn-like interface for quick model development
- **TensorFlow Backend**: Built on TensorFlow 2.x for high performance and flexibility

## Installation

```bash
# Clone the repository
git clone https://github.com/markahodgson/Auto-DL.git
cd Auto-DL

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Classification Example

```python
from autodl import AutoDL
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create and train Auto-DL model
auto_dl = AutoDL(task_type="classification", optimize_hyperparameters=True)
model = auto_dl.fit(X_train, y_train)

# Evaluate
metrics = auto_dl.evaluate(X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.4f}")

# Make predictions
predictions = auto_dl.predict(X_test)
```

### Regression Example

```python
from autodl import AutoDL
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train Auto-DL model
auto_dl = AutoDL(task_type="regression", optimize_hyperparameters=True)
model = auto_dl.fit(X_train, y_train)

# Evaluate
metrics = auto_dl.evaluate(X_test, y_test)
print(f"Test MAE: {metrics['mae']:.4f}")
```

## Usage

### Basic Usage

```python
from autodl import AutoDL

# Initialize with default settings
auto_dl = AutoDL(task_type="classification")

# Fit the model
model = auto_dl.fit(X_train, y_train)

# Make predictions
predictions = auto_dl.predict(X_test)
```

### Advanced Configuration

```python
auto_dl = AutoDL(
    task_type="classification",           # "classification" or "regression"
    optimize_hyperparameters=True,        # Enable hyperparameter optimization
    max_trials=10,                        # Number of optimization trials
    test_size=0.2,                        # Validation split size
    random_state=42,                      # Random seed for reproducibility
    verbose=1                             # Verbosity level (0, 1, or 2)
)
```

## Components

### AutoDL
Main interface for automatic model generation. Handles the entire pipeline from data preprocessing to model training.

### ModelGenerator
Generates neural network architectures automatically based on data characteristics. Supports both fully-connected networks and CNNs.

### HyperparameterOptimizer
Optimizes model hyperparameters using random search or grid search strategies.

### DataPreprocessor
Handles data preprocessing including normalization, encoding, and train-test splitting.

## Examples

Check the `examples/` directory for more detailed examples:
- `classification_example.py`: Iris dataset classification
- `regression_example.py`: Synthetic regression problem

Run examples with:
```bash
python examples/classification_example.py
python examples/regression_example.py
```

## Architecture

Auto-DL automatically determines the architecture based on:
- Input data dimensions
- Task type (classification/regression)
- Number of classes (for classification)

The architecture includes:
- Automatic layer size determination
- Dropout for regularization
- Appropriate activation functions
- Task-specific output layers

## Hyperparameter Optimization

Auto-DL searches over:
- Learning rate
- Batch size
- Number of epochs
- Hidden layer sizes
- Dropout rate
- Optimizer type (Adam, SGD, RMSprop)

## Requirements

- Python >= 3.8
- TensorFlow >= 2.13.0
- NumPy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

Built with TensorFlow and inspired by AutoML research.