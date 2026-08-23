"""
train_cnn.py
------------
CodeAlpha Machine Learning Internship - Task 3: Handwritten Character Recognition

Objective:
    Identify handwritten characters/digits using a Convolutional Neural Network (CNN).

Dataset:
    MNIST (60,000 train / 10,000 test images of handwritten digits 0-9).
    Loaded automatically via keras.datasets.mnist (requires internet on first run
    to download ~11MB from the Keras dataset server).

    To extend to full alphabet characters (as suggested in the task), swap MNIST
    for EMNIST ('byclass' or 'letters' split) — the model architecture below works
    unchanged, just update NUM_CLASSES and the data loading section.

Requirements:
    pip install tensorflow matplotlib seaborn scikit-learn

Run:
    python3 src/train_cnn.py

NOTE: This script needs internet access (to download MNIST) and TensorFlow
installed. Recommended: run in Google Colab (Runtime > Change runtime type >
GPU) for a fast, zero-setup environment — just upload this repo or copy the
code into a Colab cell.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)
keras.utils.set_random_seed(42)

NUM_CLASSES = 10
IMG_SHAPE = (28, 28, 1)
OUTPUT_DIR = "outputs"
MODEL_PATH = "models/cnn_mnist_model.keras"


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1] and add channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")
    return x_train, y_train, x_test, y_test


def build_cnn_model():
    """CNN architecture: 2 conv blocks + dense classifier head."""
    model = keras.Sequential([
        keras.Input(shape=IMG_SHAPE),

        layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),

        layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_history.png", dpi=120)
    plt.close()


def plot_sample_predictions(model, x_test, y_test, n=15):
    preds = np.argmax(model.predict(x_test[:n], verbose=0), axis=1)
    plt.figure(figsize=(15, 4))
    for i in range(n):
        plt.subplot(2, 8, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap="gray")
        color = "green" if preds[i] == y_test[i] else "red"
        plt.title(f"P:{preds[i]} / T:{y_test[i]}", color=color, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sample_predictions.png", dpi=120)
    plt.close()


def main():
    x_train, y_train, x_test, y_test = load_data()

    model = build_cnn_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - CNN on MNIST")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=120)
    plt.close()

    plot_training_history(history)
    plot_sample_predictions(model, x_test, y_test)

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
