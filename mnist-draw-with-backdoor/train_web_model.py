import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflowjs as tfjs

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data to match your current model's expected input
# Current model expects flattened input [batch_size, 784]
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create model with same structure as your current one
# (Input shape matches what your JavaScript expects)
model = keras.Sequential([
    keras.layers.Input(shape=(784,), name='image_input'),  # Named input layer
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train the model
print("Training model...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,  # Adjust as needed
    validation_split=0.1,
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Save as Keras model first
model.save('mnist_model.h5')
print("Keras model saved as 'mnist_model.h5'")

# Convert to TensorFlow.js format
print("Converting to TensorFlow.js format...")
tfjs.converters.save_keras_model(model, './web_model')
print("TensorFlow.js model saved in './web_model' directory")

# Test prediction to ensure it works
test_sample = x_test[0:1]  # Shape: [1, 784]
prediction = model.predict(test_sample)
print(f"Test prediction shape: {prediction.shape}")
print(f"Predicted class: {np.argmax(prediction)}")
print(f"Actual class: {np.argmax(y_test[0])}")