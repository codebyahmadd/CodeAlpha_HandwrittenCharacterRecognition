from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DigitVision AI",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        /* Main application */
        .main {
            padding-top: 1rem;
        }

        /* Hero section */
        .hero {
            text-align: center;
            padding: 1.5rem 0 1rem 0;
        }

        .hero h1 {
            font-size: 3.2rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .hero p {
            font-size: 1.15rem;
            opacity: 0.75;
            margin-bottom: 0;
        }

        /* Section headings */
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0.5rem 0 0.8rem 0;
        }

        /* Prediction card */
        .prediction-card {
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-top: 0.5rem;
        }

        .prediction-label {
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            opacity: 0.65;
        }

        .prediction-digit {
            font-size: 6rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0.5rem 0;
        }

        .prediction-confidence {
            font-size: 1rem;
            opacity: 0.8;
        }

        .confidence-value {
            font-weight: 700;
        }

        /* Info cards */
        .info-card {
            padding: 1.2rem;
            border-radius: 15px;
            border: 1px solid rgba(128, 128, 128, 0.2);
            margin-bottom: 0.8rem;
        }

        /* Footer */
        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = Path("models/cnn_mnist_model.keras")
TEST_ACCURACY = 99.14
NUM_CLASSES = 10


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained CNN model once and cache it."""
    return keras.models.load_model(MODEL_PATH)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Convert uploaded image into the format expected by the MNIST CNN.
    """

    # Convert to grayscale
    image = image.convert("L")

    # Invert image so dark handwriting becomes bright
    image = ImageOps.invert(image)

    # Resize to MNIST dimensions
    image = image.resize((28, 28))

    # Normalize pixel values
    image_array = np.array(image).astype("float32") / 255.0

    # Add channel dimension
    image_array = np.expand_dims(image_array, axis=-1)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🔢 DigitVision AI</h1>
        <p>
            Handwritten Digit Recognition powered by a
            Convolutional Neural Network
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🔢 DigitVision AI")

    st.markdown(
        "### About the Project"
    )

    st.write(
        "DigitVision AI is a CNN-based handwritten digit "
        "recognition system trained on the MNIST dataset."
    )

    st.divider()

    st.markdown("### 🧠 Model")

    st.write("**Architecture:** Convolutional Neural Network")

    st.write("**Framework:** TensorFlow / Keras")

    st.write("**Dataset:** MNIST")

    st.write("**Classes:** 10 digits (0–9)")

    st.divider()

    st.markdown("### 📊 Performance")

    st.metric(
        "Test Accuracy",
        f"{TEST_ACCURACY:.2f}%"
    )

    st.divider()

    st.markdown("### 💡 Tips")

    st.write(
        "For better predictions, upload a clear image "
        "containing a single handwritten digit."
    )

    st.caption(
        "CodeAlpha Machine Learning Internship — Task 3"
    )


# ============================================================
# MODEL STATUS
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "❌ Trained model not found.\n\n"
        "Please make sure the following file exists:\n\n"
        "`models/cnn_mnist_model.keras`"
    )

    st.stop()


try:

    model = load_model()

except Exception as error:

    st.error(
        f"❌ Unable to load the trained model.\n\n"
        f"Error: {error}"
    )

    st.stop()


# ============================================================
# TOP METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "🤖 Model",
        "CNN"
    )

with metric2:
    st.metric(
        "🎯 Test Accuracy",
        "99.14%"
    )

with metric3:
    st.metric(
        "🔢 Classes",
        "10"
    )

with metric4:
    st.metric(
        "🖼️ Image Size",
        "28 × 28"
    )


st.divider()


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    "## 🖼️ Upload a Handwritten Digit"
)

st.write(
    "Upload an image containing a single handwritten digit "
    "to let the CNN model make a prediction."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"],
    help=(
        "For best results, use a clear image with "
        "one handwritten digit."
    ),
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file)

    # Preprocess
    processed_image = preprocess_image(image)

    # Predict
    prediction = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_digit = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_digit]
    ) * 100


    # --------------------------------------------------------
    # IMAGE + PREDICTION
    # --------------------------------------------------------

    left, right = st.columns(
        2,
        gap="large"
    )


    # --------------------------------------------------------
    # LEFT: INPUT IMAGE
    # --------------------------------------------------------

    with left:

        st.markdown(
            "### 🖼️ Input Image"
        )

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            "Uploaded handwritten digit"
        )


    # --------------------------------------------------------
    # RIGHT: AI PREDICTION
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### 🎯 AI Prediction"
        )

        st.success(
            f"### Predicted Digit: {predicted_digit}"
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


    st.divider()


    # ========================================================
    # CONFIDENCE DISTRIBUTION
    # ========================================================

    st.markdown(
        "## 📊 Prediction Confidence"
    )

    st.write(
        "Probability distribution across all ten digit classes."
    )

    probabilities = {
        str(i): float(prediction[i]) * 100
        for i in range(NUM_CLASSES)
    }

    st.bar_chart(
        probabilities
    )


    st.divider()


    # ========================================================
    # TOP PREDICTIONS
    # ========================================================

    st.markdown(
        "## 🏆 Top Predictions"
    )

    top_indices = np.argsort(
        prediction
    )[::-1][:3]


    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        probability = (
            float(prediction[index]) * 100
        )

        col1, col2, col3 = st.columns(
            [1, 3, 2]
        )

        with col1:

            st.write(
                f"**#{rank}**"
            )

        with col2:

            st.write(
                f"**Digit {index}**"
            )

        with col3:

            st.write(
                f"**{probability:.2f}%**"
            )

        st.progress(
            min(probability / 100, 1.0)
        )


else:

    st.info(
        "👆 Upload a handwritten digit image above "
        "to get an AI prediction."
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.markdown(
    "## 📌 About DigitVision AI"
)

info1, info2, info3 = st.columns(3)


with info1:

    st.markdown(
        """
        **🧠 Deep Learning**

        A Convolutional Neural Network learns
        visual patterns from handwritten digits.
        """
    )


with info2:

    st.markdown(
        """
        **📚 MNIST Dataset**

        The model is trained on 60,000 handwritten
        training images and evaluated on 10,000 test images.
        """
    )


with info3:

    st.markdown(
        """
        **🎯 High Accuracy**

        The trained CNN achieved **99.14% test accuracy**
        on the MNIST test dataset.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DigitVision AI • Built with Python, TensorFlow/Keras "
    "and Streamlit • CodeAlpha Machine Learning Internship — Task 3"
)