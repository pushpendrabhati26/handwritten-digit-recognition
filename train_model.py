
import tensorflow as tf
from tensorflow.keras import layers, models

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize images to the range 0 to 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# CNN expects images with a channel dimension
x_train = x_train[..., None]
x_test = x_test[..., None]

# Build a Convolutional Neural Network
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training CNN...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"\nTest accuracy: {accuracy * 100:.2f}%")

model.save("digit_model.keras")

print("New CNN saved successfully!")
