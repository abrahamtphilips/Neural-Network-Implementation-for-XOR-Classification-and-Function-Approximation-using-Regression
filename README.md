# Neural-Network-Implementation-for-XOR-Classification-and-Function-Approximation-using-Regression
This project demonstrates the use of neural networks for two tasks: 1. XOR Classification 2. Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(32000)

 This project demonstrates the use of neural networks for two tasks:
 
 1. XOR Classification: A feedforward neural network is trained to classify the XOR logic function using a hidden layer with 2 neurons, tanh and
   sigmoid activation functions, and backpropagation for training. The    decision boundary is visualized in 3D, and the training loss is tracked.

 2. Regression: The neural network approximates the relationship between input X and target T using data from an external file. Two models, one with 3
   hidden neurons and another with 20, are trained and compared in terms of accuracy and training error. The model outputs are plotted alongside the
   actual data, and the loss function evolution is visualized during training.


# Defining the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Defining the derivative of sigmoid activation function
def sigmoid_derivative(x):
    return x * (1 - x)


# Defining the hyperbolic tangent activation function
def tanh(x):
    return np.tanh(x)


# Defining the derivative of hyperbolic tangent activation function
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# Initialize weights and biases
input_size = 2
hidden_size = 2
output_size = 1
weights_input_hidden = np.random.randn(input_size, hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_hidden = np.zeros((1, hidden_size))
bias_output = np.zeros((1, output_size))


# Forward pass
def forward(x):
    global hidden_output, output
    hidden_output = tanh(np.dot(x, weights_input_hidden) + bias_hidden)
    output = sigmoid(np.dot(hidden_output, weights_hidden_output) + bias_output)
    return output


# Backpropagation
def backward(x, y, learning_rate):
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output
    global hidden_output, output

    output = forward(x)
    error = y - output
    delta_output = error * sigmoid_derivative(output)
    error_hidden = delta_output.dot(weights_hidden_output.T)
    delta_hidden = error_hidden * tanh_derivative(hidden_output)

    # Updating weights and biases
    weights_hidden_output += hidden_output.T.dot(delta_output) * learning_rate
    weights_input_hidden += x.T.dot(delta_hidden) * learning_rate
    bias_output += np.sum(delta_output, axis=0) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0) * learning_rate


# XOR input data
X_xor = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
# XOR output labels
y_xor = np.array([[0], [0], [1], [1]])

# Training loop for XOR classification
learning_rate_xor = 0.1
epochs_xor = 10000
loss_history_xor = []

for epoch in range(epochs_xor):
    # Forward pass
    output_xor = forward(X_xor)

    # Calculate loss
    loss_xor = np.mean(np.square(y_xor - output_xor))
    loss_history_xor.append(loss_xor)

    # Backpropagation
    backward(X_xor, y_xor, learning_rate_xor)

# Calculation of the training error for XOR classification
training_error_xor = np.mean((forward(X_xor) - y_xor) ** 2)
print(f'Training error for XOR classification: {training_error_xor}')


# Regression

# Loading data from Excel file
data = pd.read_excel('Proj5Dataset.xlsx')  # Read Data From File

# Extraction of X and T from the dataset
X = data['X'].values.reshape(-1, 1)
T = data['T'].values.reshape(-1, 1)


# Defining forward pass function
def forwardr(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_output = tanh(np.dot(X, weights_input_hidden) + bias_hidden)
    output = np.dot(hidden_output, weights_hidden_output) + bias_output
    return output, hidden_output


# Defining the backward pass function
def backwardr(X, y, learning_rate, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    output, hidden_output = forwardr(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    error = y - output
    delta_output = error
    error_hidden = delta_output.dot(weights_hidden_output.T)
    delta_hidden = error_hidden * tanh_derivative(hidden_output)

    # Updating weights and biases
    weights_hidden_output += hidden_output.T.dot(delta_output) * learning_rate
    weights_input_hidden += X.T.dot(delta_hidden) * learning_rate
    bias_output += np.sum(delta_output, axis=0) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0) * learning_rate


# Hyperparameters
learning_rate = 0.0005
epochs = 55000

# Initialization of the weights and biases for 3 hidden layers
input_size = X.shape[1]
hidden_size = 3
output_size = 1
weights_input_hidden_3 = np.random.randn(input_size, hidden_size)
weights_hidden_output_3 = np.random.randn(hidden_size, output_size)
bias_hidden_3 = np.zeros((1, hidden_size))
bias_output_3 = np.zeros((1, output_size))

# Training the neural network for 3 hidden layers
losses_3 = []
for epoch in range(epochs):
    # Forward pass
    Y, _ = forwardr(X, weights_input_hidden_3, weights_hidden_output_3, bias_hidden_3, bias_output_3)

    # Compute loss
    loss = np.mean((Y - T)**2)
    losses_3.append(loss)

    # Backward pass
    backwardr(X, T, learning_rate, weights_input_hidden_3, weights_hidden_output_3, bias_hidden_3, bias_output_3)

# Initialization weights and biases for 20 hidden layers
hidden_size = 20
weights_input_hidden_20 = np.random.randn(input_size, hidden_size)
weights_hidden_output_20 = np.random.randn(hidden_size, output_size)
bias_hidden_20 = np.zeros((1, hidden_size))
bias_output_20 = np.zeros((1, output_size))

# Training the neural network for 20 hidden layers
losses_20 = []
for epoch in range(epochs):
    # Forward pass
    Y, _ = forwardr(X, weights_input_hidden_20, weights_hidden_output_20, bias_hidden_20, bias_output_20)

    # Compute loss
    loss = np.mean((Y - T)**2)
    losses_20.append(loss)

    # Backward pass
    backwardr(X, T, learning_rate, weights_input_hidden_20, weights_hidden_output_20, bias_hidden_20, bias_output_20)


# Generation of  dense X values for plotting the models
X_dense = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)

# Calculation of the final Y values using the trained models
Y_final_3, _ = forwardr(X_dense, weights_input_hidden_3, weights_hidden_output_3, bias_hidden_3, bias_output_3)
Y_final_20, _ = forwardr(X_dense, weights_input_hidden_20, weights_hidden_output_20, bias_hidden_20, bias_output_20)

# Reporting training error for both cases
training_error_3 = np.mean((forwardr(X, weights_input_hidden_3, weights_hidden_output_3, bias_hidden_3, bias_output_3)[0] - T)**2)
training_error_20 = np.mean((forwardr(X, weights_input_hidden_20, weights_hidden_output_20, bias_hidden_20, bias_output_20)[0] - T)**2)

print(f'Training error for 3 hidden units(Regression): {training_error_3}')
print(f'Training error for 20 hidden units(Regression): {training_error_20}')

# Plotting the resulting models and input data at the same time

# Plotting resultant models from 3 and 20 hidden units

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
data_scatter = plt.scatter(X, T, label='Data')
model_line, = plt.plot(X_dense, Y_final_3, color='red', label='Model')
plt.xlabel('X')
plt.ylabel('T')
plt.title('Regression Model with 3 Hidden Units')
plt.legend(handles=[data_scatter, model_line, plt.Line2D([], [], color='white', label=f'Training Error: {training_error_3:.5g}')])
plt.grid()

plt.subplot(1, 2, 2)
data_scatter = plt.scatter(X, T, label='Data')
model_line, = plt.plot(X_dense, Y_final_20, color='red', label='Model')
plt.xlabel('X')
plt.ylabel('T')
plt.title('Regression Model with 20 Hidden Units')
plt.legend(handles=[data_scatter, model_line, plt.Line2D([], [], color='white', label=f'Training Error: {training_error_20:.5g}')])
plt.grid()

# Plotting loss function for 3 and 20 hidden units during training

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(epochs), losses_3)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function for 3 Hidden Units (Regression)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(epochs), losses_20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function for 20 Hidden Units (Regression) ')
plt.grid()

plt.figure()
# Plotting decision surface for XOR after training

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

h = 0.01
x_min_xor, x_max_xor = X_xor[:, 0].min() - 1, X_xor[:, 0].max() + 1
y_min_xor, y_max_xor = X_xor[:, 1].min() - 1, X_xor[:, 1].max() + 1
xx_xor, yy_xor = np.meshgrid(np.arange(x_min_xor, x_max_xor, h), np.arange(y_min_xor, y_max_xor, h))
Z_xor = forward(np.c_[xx_xor.ravel(), yy_xor.ravel()])
Z_xor = Z_xor.reshape(xx_xor.shape)

ax.plot_surface(xx_xor, yy_xor, Z_xor, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X_xor[:, 0], X_xor[:, 1], y_xor.ravel(), c=y_xor.ravel(), cmap=plt.cm.coolwarm)
ax.set_title('Decision Surface for XOR Classification')
ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')

# Plotting Loss function through XOR training
plt.figure()
plt.plot(loss_history_xor)
plt.title('Loss Function Value Throughout Training (XOR Classification)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()

plt.tight_layout()
plt.show()

