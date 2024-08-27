# Importing necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from sklearn.datasets import load_iris  # To load the Iris dataset
import tensorflow as tf  # For building and training neural network models
from tensorflow import keras  # High-level API for building neural networks
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.preprocessing import StandardScaler  # For feature scaling

# Setting the random seeds to ensure reproducibility
tf.random.set_seed(42)  # For TensorFlow random operations
np.random.seed(42)  # For NumPy random operations

# Load the Iris dataset
irisData = load_iris()  # Load the Iris dataset into a dictionary-like object
X = irisData.data  # Features (attributes of the dataset)
y = irisData.target  # Target labels (class of the Iris flowers)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,  # Features
    y,  # Target labels
    test_size=0.2,  # 20% of the data will be used for testing
    stratify=y,  # Ensure that the split maintains the distribution of classes
    random_state=42  # Seed for the random number generator
)

# Initialize the StandardScaler for feature scaling
sc = StandardScaler()
sc.fit(X_train)  # Fit the scaler on the training data
X_train_std = sc.transform(X_train)  # Apply scaling to the training data
X_test_std = sc.transform(X_test)  # Apply scaling to the testing data

# Define the Deep Neural Network (DNN) model
model_DNN = keras.models.Sequential()  # Initialize a sequential model

# Add layers to the model
model_DNN.add(keras.layers.Dense(units=12, activation='relu', input_shape=(X_train_std.shape[1],)))  # Hidden layer with 12 units and ReLU activation
model_DNN.add(keras.layers.Dense(units=10, activation='relu'))  # Hidden layer with 10 units and ReLU activation
model_DNN.add(keras.layers.Dense(units=8, activation='relu'))  # Hidden layer with 8 units and ReLU activation
model_DNN.add(keras.layers.Dense(units=6, activation='relu'))  # Hidden layer with 6 units and ReLU activation
model_DNN.add(keras.layers.Dense(units=3, activation='softmax'))  # Output layer with 3 units (one for each class) and softmax activation

# Print a summary of the model architecture
model_DNN.summary()

# Compile the model with a loss function, optimizer, and metrics
model_DNN.compile(
    loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
    optimizer='adam',  # Optimizer to use for training
    metrics=['accuracy']  # Metric to evaluate the model's performance
)

# Train the model on the training data
model_DNN.fit(
    x=X_train_std,  # Features for training
    y=y_train,  # Target labels for training
    validation_split=0.1,  # 10% of the training data used for validation
    epochs=50,  # Number of epochs (iterations over the entire training dataset)
    batch_size=16  # Number of samples per gradient update
)

# Evaluate the model's performance on the testing data
test_loss, test_accuracy = model_DNN.evaluate(x=X_test_std, y=y_test)
print(test_loss, test_accuracy)  # Print the loss and accuracy on the test set

# Save the trained model to a file
model_DNN.save('model_IRIS.keras')

# Load the saved model from the file
savedModel = tf.keras.models.load_model('model_IRIS.keras')

# Prepare new data for prediction
X_new = np.array([5.1, 3.5, 1.4, 0.2]).reshape(1, -1)  # Example new data with 4 features, reshaped to match input shape

# Apply the same scaling to the new data
X_new_std = sc.transform(X_new)

# Predict probabilities for the new data
y_pred_prob = savedModel.predict(X_new_std)
print("Predicted probabilities:", y_pred_prob)

# Convert the predicted probabilities to class labels
y_pred_class = np.argmax(y_pred_prob, axis=1)  # Find the index of the highest probability (class label)
print("Predicted class:", y_pred_class[0])  # Print the predicted class label
