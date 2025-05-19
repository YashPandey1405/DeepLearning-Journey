# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (4-dimensional: sepal/petal length & width)
y = iris.target.reshape(-1, 1)  # Target labels (reshape for encoder compatibility)

# Standardize the features (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the labels (e.g., [1] → [0, 1, 0])
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Build the model
model = Sequential([
    # First hidden layer with 64 neurons, ReLU activation, and L2 regularization
    Dense(64, input_shape=(4,), activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),  # Drop 30% of neurons randomly during training (Dropout regularization)

    # Second hidden layer with 32 neurons and L2 regularization
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),  # Apply dropout again

    # Output layer with 3 neurons (one for each class), using softmax activation
    Dense(3, activation='softmax')
])

# Compile the model
# - Optimizer: Adam (adaptive learning rate)
# - Loss: categorical crossentropy (used for multi-class classification)
# - Metric: accuracy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# - Epochs: 100 iterations over the data
# - Batch size: 8 samples per gradient update
# - Validation split: 20% of training data used for validation
# - verbose=0: silent training (change to 1 for progress bar)
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=0)

# Evaluate the trained model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")
