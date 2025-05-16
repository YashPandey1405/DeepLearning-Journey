# Import TensorFlow library
import tensorflow as tf

# Import required classes for building a Sequential model and adding Dense (fully connected) layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize a Sequential model (layers will be added one after another)
model = Sequential()

# Add the first hidden layer:
# - 10 neurons
# - ReLU activation function
# - input_shape=(4,) means the model expects input with 4 features
model.add(Dense(10, activation='relu', input_shape=(4,)))

# Add the second hidden layer with 4 neurons and ReLU activation
model.add(Dense(4, activation='relu'))

# Add the third hidden layer with 2 neurons and ReLU activation
model.add(Dense(2, activation='relu'))

# Add the output layer:
# - 1 neuron (for regression output)
# - Linear activation (default for regression problems)
model.add(Dense(1, activation='linear'))

# Compile the model to prepare it for training
# - 'mean_squared_error' is the loss function (used for regression tasks)
# - 'adam' is the optimizer (adjusts weights to reduce loss)
# - 'mae' (mean absolute error) is used to evaluate model performance
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mae']
)
