# CodeAlpha_HandwrittenCharacterRecognition

**CodeAlpha Machine Learning Internship — Task 3**

## 📌 Objective
Identify handwritten characters (digits) using image processing and deep learning — specifically a **Convolutional Neural Network (CNN)**.

## 🧠 Approach
- Dataset: **MNIST** — 60,000 training / 10,000 test images of handwritten digits (0-9), 28x28 grayscale
- Model: **CNN** with 3 convolutional blocks (Conv2D + BatchNorm + MaxPooling), followed by a dense classifier head with dropout for regularization
- Framework: TensorFlow / Keras

## ⚠️ Important — How to Run
This project needs **TensorFlow** and **internet access** (to auto-download MNIST on first run). The easiest, zero-setup way to run it:

### Option A: Google Colab (recommended)
1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Upload `notebooks/Handwritten_Character_Recognition_CNN.ipynb`
3. Runtime → Change runtime type → GPU (optional, but faster)
4. Run all cells

### Option B: Run locally
```bash
pip install -r requirements.txt
python3 src/train_cnn.py
```

## 📁 Project Structure
```
CodeAlpha_HandwrittenCharacterRecognition/
├── notebooks/
│   └── Handwritten_Character_Recognition_CNN.ipynb   # main CNN notebook (Colab-ready)
├── src/
│   ├── train_cnn.py         # full CNN training script (TensorFlow/Keras + MNIST)
│   └── offline_demo.py      # lightweight offline pipeline demo (no internet/TF needed)
├── models/                  # trained model saved here after running
├── outputs/                 # plots: confusion matrix, training curves, sample predictions
├── requirements.txt
└── README.md
```

## 🧪 Offline Demo (bonus)
`src/offline_demo.py` is a small, fully offline sanity-check of the pipeline (data loading → train → evaluate) using scikit-learn's built-in 8x8 digit dataset and an MLP classifier — useful for quickly verifying the project logic without installing TensorFlow or needing internet. It achieved **96.7% accuracy** on this simpler dataset. The actual task submission is the CNN in `src/train_cnn.py` / the notebook.

```bash
python3 src/offline_demo.py
```

## 📈 Expected Results (CNN on MNIST)
A CNN of this depth typically reaches **~99% test accuracy** on MNIST — this is a well-known benchmark result for this style of architecture (documented extensively in deep learning literature). Run the notebook to generate your own confusion matrix, training curves, and sample prediction plots in `outputs/`.

## 🔧 Extending the Project
- Swap MNIST for **EMNIST** (`byclass` or `letters` split) to recognize full alphabet characters, not just digits
- Use a **CRNN** (CNN + RNN) architecture to extend to full word/sentence recognition
- Deploy with a simple Streamlit/Flask app where users can draw a digit and get a live prediction

## 🛠 Tech Stack
Python · TensorFlow/Keras · scikit-learn · matplotlib · seaborn

## 🎓 About
This project was built as part of the **CodeAlpha Machine Learning Virtual Internship**.

---
*Author: Ahmad Yar Daha · [LinkedIn](https://www.linkedin.com/in/ahmad-yar-daha-6753bb423/) · [GitHub](https://github.com/codebyahmadd)*
