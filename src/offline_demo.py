"""
offline_demo.py
---------------
A lightweight, fully OFFLINE demo of the handwritten digit recognition pipeline
using scikit-learn's built-in `load_digits` dataset (1,797 8x8 images, no
internet/download required) and an MLP (Multi-Layer Perceptron) classifier.

Purpose:
    This script lets you verify the end-to-end pipeline (load -> train ->
    evaluate -> save) works instantly with zero setup. It is NOT the CNN
    submission itself — for the actual CNN-on-MNIST model required by the
    task, see src/train_cnn.py (needs TensorFlow + internet, works great on
    Google Colab).

Run:
    python3 src/offline_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

np.random.seed(42)
OUTPUT_DIR = "outputs"


def main():
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"Dataset: {X.shape[0]} images, {digits.images.shape[1]}x{digits.images.shape[2]} pixels each")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # MLP with two hidden layers - a simplified stand-in for a CNN,
    # since a full conv net needs TensorFlow/PyTorch (not available offline here)
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - MLP on digits (offline demo)\nAccuracy: {acc:.2%}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/offline_demo_confusion_matrix.png", dpi=120)
    plt.close()

    # Sample predictions grid
    n = 15
    plt.figure(figsize=(15, 4))
    for i in range(n):
        plt.subplot(2, 8, i + 1)
        plt.imshow(X_test[i].reshape(8, 8), cmap="gray")
        color = "green" if y_pred[i] == y_test[i] else "red"
        plt.title(f"P:{y_pred[i]} / T:{y_test[i]}", color=color, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/offline_demo_sample_predictions.png", dpi=120)
    plt.close()

    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
