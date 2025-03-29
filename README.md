I'll create a comprehensive README documentation for your Fawern-NN neural network library, based on your code and project structure.

# Fawern-NN: Neural Network Library in Pure Python

![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A lightweight neural network implementation built from scratch using only NumPy. Fawern-NN provides a simple, intuitive interface for building, training, and evaluating neural networks with minimal dependencies.

## Features

- Pure Python implementation with minimal dependencies (NumPy, Matplotlib, scikit-learn)
- Keras-inspired API for easy model building and training
- Support for various activation functions:
  - Sigmoid
  - Tanh
  - ReLU
  - Leaky ReLU
  - Softmax
  - Linear
- Customizable network architecture with flexible layer definitions
- Support for batch training
- Built-in evaluation metrics and visualization tools
- Extensible design for adding custom activation functions

## Installation

```bash
pip install fawern-nn
```

## Quick Start

### XOR Problem Example

```python
import numpy as np
from fawern_nn.nn import Layers, NInput, NLayer

# XOR problem
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Create model
model = Layers()

# Add layers
model.add(NInput(2))
model.add(NLayer(4, activation='tanh'))
model.add(NLayer(4, activation='tanh'))
model.add(NLayer(1, activation='sigmoid'))

# Train model
model.train_model(X, y, loss_type='categorical', iterations=10000, learning_rate=0.1, batch_size=4)

# Evaluate model
accuracy, conf_matrix = model.evaluate_trained_model()
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Visualize training progress
model.show_loss_graph()
```

## Core Components

### Layer Classes

#### `Layers`

The main model container for building and training neural networks.

```python
model = Layers()
```

**Methods:**

- `add(layer)`: Add a layer to the model
- `train_model(x, y, loss_type, iterations, learning_rate, batch_size)`: Train the model
  - `x`: Input data (numpy array)
  - `y`: Target data (numpy array)
  - `loss_type`: Type of loss function ('categorical', 'mse', 'mae')
  - `iterations`: Number of training iterations
  - `learning_rate`: Learning rate for weight updates
  - `batch_size`: Size of batches for training
- `evaluate_trained_model()`: Evaluate model performance
- `show_loss_graph()`: Visualize training loss over iterations
- `predict_input()`: Get model predictions

#### `NInput`

The input layer specification.

```python
input_layer = NInput(input_shape)
```

**Parameters:**
- `input_shape`: Number of input features

#### `NLayer`

The standard neural network layer.

```python
layer = NLayer(num_neurons, activation='linear', use_bias=True)
```

**Parameters:**
- `num_neurons`: Number of neurons in the layer
- `activation`: Activation function (default: 'linear')
- `use_bias`: Whether to use bias (default: True)
- `function_name`: Optional name for custom activation function
- `function_formula`: Optional formula for custom activation function

**Methods:**
- `set_weights(output_shape, new_weights)`: Set layer weights
- `get_weights()`: Get layer weights
- `set_activation(activation)`: Set activation function
- `get_activation()`: Get activation function
- `forward(input_data)`: Perform forward propagation

#### `FlattenLayer`

Layer to flatten multi-dimensional input.

```python
flatten = FlattenLayer()
```

**Methods:**
- `forward(input_data)`: Flatten input data

### Activation Functions

The `ActivationFunctions` class provides various activation functions:

- `sigmoid`: Sigmoid activation (0 to 1)
- `tanh`: Hyperbolic tangent (-1 to 1)
- `relu`: Rectified Linear Unit (max(0, x))
- `leaky_relu`: Leaky ReLU (small slope for negative inputs)
- `linear`: Linear/identity function
- `softmax`: Softmax function for multi-class classification

#### Adding Custom Activation Functions

```python
from fawern_nn.nn import ActivationFunctions

# Create activation functions instance
activations = ActivationFunctions()

# Define custom function
def custom_activation(x):
    return x**2

# Define its derivative
def custom_activation_derivative(x):
    return 2*x

# Add to available functions
activations.add_activation_function('custom', custom_activation)
activations.add_activation_function('custom_derivative', custom_activation_derivative)
```

## Examples

### Binary Classification

```python
import numpy as np
from fawern_nn.nn import Layers, NInput, NLayer

# Create binary classification dataset
X = np.random.randn(100, 2)
y = np.array([(1 if x[0] + x[1] > 0 else 0) for x in X]).reshape(-1, 1)

# Create model
model = Layers()
model.add(NInput(2))
model.add(NLayer(4, activation='relu'))
model.add(NLayer(1, activation='sigmoid'))

# Train model
model.train_model(X, y, loss_type='categorical', iterations=1000, learning_rate=0.01)

# Evaluate and visualize
accuracy, conf_matrix = model.evaluate_trained_model()
print(f"Accuracy: {accuracy}")
model.show_loss_graph()
```

### Regression

```python
import numpy as np
from fawern_nn.nn import Layers, NInput, NLayer

# Create regression dataset
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X) + 0.1 * np.random.randn(100, 1)

# Create model
model = Layers()
model.add(NInput(1))
model.add(NLayer(10, activation='tanh'))
model.add(NLayer(1, activation='linear'))

# Train model
model.train_model(X, y, loss_type='mse', iterations=2000, learning_rate=0.005)

# Evaluate
mse = model.evaluate_trained_model()
print(f"Mean Squared Error: {mse}")
model.show_loss_graph()
```

### Multi-Layer Network

```python
import numpy as np
from fawern_nn.nn import Layers, NInput, NLayer

# Create multi-class dataset (simplified MNIST-like)
X = np.random.randn(500, 28*28)  # 28x28 flattened images
y = np.eye(10)[np.random.randint(0, 10, size=500)]  # One-hot encoded labels

# Create model
model = Layers()
model.add(NInput(28*28))
model.add(NLayer(128, activation='relu'))
model.add(NLayer(64, activation='relu'))
model.add(NLayer(10, activation='softmax'))

# Train model
model.train_model(X, y, loss_type='categorical', iterations=50, learning_rate=0.001, batch_size=32)

# Evaluate model
accuracy, conf_matrix = model.evaluate_trained_model()
print(f"Accuracy: {accuracy}")
model.show_loss_graph()
```

## Technical Details

### Backpropagation Implementation

Fawern-NN implements traditional backpropagation for training neural networks:

1. Forward pass through all layers
2. Calculate error at output layer
3. Propagate error backward through the network
4. Update weights based on calculated gradients

The implementation supports mini-batch training for better performance on larger datasets.

### Loss Functions

- `categorical`: For classification problems (uses accuracy and confusion matrix for evaluation)
- `mse`: Mean Squared Error for regression problems
- `mae`: Mean Absolute Error for regression problems

## Requirements

- Python 3.7+
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- scikit-learn >= 0.24.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Author

Fawern - [GitHub](https://github.com/fawern)

## Project Links

- GitHub: [https://github.com/fawern/fawern_nn](https://github.com/fawern/fawern_nn)
- PyPI: [https://pypi.org/project/fawern-nn/](https://pypi.org/project/fawern-nn/)
