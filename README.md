# Handwritten Character Recognition using CNN

### CodeAlpha Machine Learning Internship — Task 3

A deep learning project for recognizing handwritten digits using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.

---

## 📌 Objective

The objective of this project is to build a deep learning model capable of identifying handwritten digits from grayscale images.

The project uses a **Convolutional Neural Network (CNN)** with TensorFlow/Keras to learn visual patterns and classify handwritten digits from **0 to 9**.

---

## 🧠 Approach

The project follows a standard image classification workflow:

* **Dataset:** MNIST — 60,000 training images and 10,000 test images
* **Image Size:** 28 × 28 grayscale images
* **Classes:** Digits 0–9
* **Model:** Convolutional Neural Network (CNN)
* **Architecture:** Three convolutional blocks using Conv2D, Batch Normalization, and MaxPooling, followed by a dense classification head with Dropout
* **Framework:** TensorFlow / Keras
* **Evaluation:** Test accuracy, confusion matrix, training curves, and sample predictions

---

## 🗂️ Dataset

The project uses the **MNIST handwritten digit dataset**, a standard benchmark dataset for image classification.

The dataset contains:

* 60,000 training images
* 10,000 test images
* 10 digit classes (0–9)
* 28 × 28 grayscale images

MNIST is automatically downloaded by TensorFlow/Keras when the main training script or notebook is executed for the first time.

---

## 🏗️ Model Architecture

The CNN consists of:

1. **Convolutional layers** for extracting visual features
2. **Batch Normalization** for more stable training
3. **Max Pooling** for spatial downsampling
4. **Dropout** for regularization
5. **Dense layers** for final classification
6. **Softmax output layer** for predicting one of the ten digit classes

This architecture allows the model to progressively learn patterns such as edges, shapes, and more complex digit features.

---

## ⚙️ How to Run

### Option A — Google Colab (Recommended)

Google Colab provides an easy environment for running the notebook without setting up TensorFlow locally.

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload:
   `notebooks/Handwritten_Character_Recognition_CNN.ipynb`
3. Select **Runtime → Change runtime type**
4. GPU can optionally be enabled for faster training
5. Run all cells

> Internet access is required initially so TensorFlow/Keras can download the MNIST dataset.

### Option B — Run Locally

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the CNN training script:

```bash
python src/train_cnn.py
```

The training process generates evaluation outputs such as training curves, confusion matrix, and sample predictions.

---

## 🧪 Offline Demo

The repository also includes a lightweight **offline demonstration**:

```bash
python src/offline_demo.py
```

The offline demo uses **scikit-learn's built-in 8 × 8 handwritten digits dataset** with an MLP classifier.

It is provided as a quick way to verify the basic data-processing, training, and evaluation workflow without requiring TensorFlow or an internet connection.

### Offline Demo Result

The offline demonstration achieved approximately **96.7% accuracy** on the simpler 8 × 8 digits dataset.

> **Note:** The offline demo is a supplementary demonstration and is not the main CodeAlpha Task 3 model. The primary project uses the CNN trained on MNIST.

---

## 📊 Outputs

The `outputs/` directory contains generated visualizations from the offline demonstration, including:

* Confusion matrix
* Sample predictions

The CNN training script/notebook can additionally generate training and evaluation visualizations when executed.

---

## 📁 Project Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
│
├── data/
│   └── README.md
│
├── models/
│   └── # Generated model files after training
│
├── notebooks/
│   └── Handwritten_Character_Recognition_CNN.ipynb
│
├── outputs/
│   ├── offline_demo_confusion_matrix.png
│   └── offline_demo_sample_predictions.png
│
├── src/
│   ├── offline_demo.py
│   └── train_cnn.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔧 Tech Stack

* **Python**
* **TensorFlow / Keras**
* **scikit-learn**
* **NumPy**
* **Matplotlib**
* **Seaborn**

---

## 🚀 Future Improvements

The project can be extended in several ways:

* Replace MNIST with **EMNIST** for recognizing a broader range of handwritten characters
* Develop a **Streamlit or Flask application** for real-time digit prediction
* Allow users to draw handwritten digits directly in a web interface
* Experiment with different CNN architectures and hyperparameters
* Extend the system toward handwritten word or character-sequence recognition

---

## 🎓 About the Project

This project was developed as part of the **CodeAlpha Machine Learning Virtual Internship — Task 3**.

It demonstrates the practical application of **deep learning, computer vision, image classification, and model evaluation** using a CNN-based approach.

---

## 👤 Author

**Ahmad Yar Daha**

[LinkedIn](https://www.linkedin.com/in/ahmad-yar-daha-6753bb423/) · [GitHub](https://github.com/codebyahmadd)

---

⭐ If you find this project useful, feel free to explore the repository and its implementation.
