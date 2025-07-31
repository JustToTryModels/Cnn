# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import requests
import tempfile
import matplotlib.pyplot as plt

# --- Configuration ---
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Custom CSS for File-Uploader & Chip ---
st.markdown("""
<style>
/* ========== 1. 'Browse files' BUTTON ========== */
[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, #ff8a00, #e52e71);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 24px;
    font-size: 1.1em;
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

/* — KEEP TEXT WHITE IN *ALL* STATES — */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button:focus,
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:active,
[data-testid="stFileUploader"] button:visited,
[data-testid="stFileUploader"] button * {
    color: white !important;
}

[data-testid="stFileUploader"] button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 15px rgba(229, 46, 113, 0.4);
}
[data-testid="stFileUploader"] button:active {
    transform: scale(0.98);
}

/* ========== 2. UPLOADED FILE 'CHIP' ========== */
[data-testid="stFileUploaderFile"]{
    display: flex;
    align-items: center;
    background-color: #4A4A4A;
    color: white;
    border-radius: 25px;
    padding: 4px 12px;
    transition: box-shadow 0.2s ease;
}

/* Filename text */
[data-testid="stFileUploaderFile"] > div:first-of-type{
    color:white !important;
    font-size:0.9em;
    padding-right:10px;
}

/* Delete ('x') button */
[data-testid="stFileUploaderFile"] button{
    background-color:transparent;
    border:none;
}
[data-testid="stFileUploaderFile"] button svg{
    fill:white;
    transition:fill 0.2s ease;
}
[data-testid="stFileUploaderFile"] button:hover svg{
    fill:#ff8a00;
}

/* Chip focus outline */
[data-testid="stFileUploaderFile"]:focus-within{
    box-shadow:0 0 0 2px rgba(229,46,113,0.6);
    outline:none;
}
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_keras_model():
    """Loads the pre-trained Keras model from a GitHub URL."""
    model_url = "https://github.com/JustToTryModels/Cnn/raw/main/Model/fashion_mnist_best_model.keras"
    try:
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
        model = tf.keras.models.load_model(tmp_file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.exception(e)
        return None

model = load_keras_model()

# --- Class Names ---
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Image Pre-processing ---
def preprocess_image(image):
    """Converts to grayscale, resizes to 28×28, inverts & normalises."""
    grayscale_img = image.convert('L')
    resized_img   = grayscale_img.resize((28, 28), Image.Resampling.LANCZOS)
    inverted_img  = ImageOps.invert(resized_img)

    img_array = np.array(inverted_img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return inverted_img, img_array

# --- Streamlit UI ---
st.title("👗 Fashion MNIST Image Classifier")
st.markdown("Welcome to the Fashion Classifier! Upload an image of a clothing item, and the model will predict its category.")
st.markdown("💡 **Tip:** For best results, use centered images with plain backgrounds.")

st.sidebar.header("About")
st.sidebar.info("""
**Model:** Advanced CNN with Batch Normalization & Dropout.  
**Dataset:** Fashion MNIST  
**Frameworks:** TensorFlow/Keras • Streamlit  
**Source Code:** [GitHub](https://github.com/JustToTryModels/Cnn)
""")

uploaded_file = st.file_uploader("Choose an image of a fashion item…", type=["jpg", "jpeg", "png"])

# --- Main Logic ---
if uploaded_file is not None:
    if model is not None:
        original_image = Image.open(uploaded_file)

        with st.spinner("Classifying…"):
            processed_display_img, processed_model_img = preprocess_image(original_image)
            prediction = model.predict(processed_model_img)

            pred_probs       = prediction[0]
            top_class_index  = np.argmax(pred_probs)
            top_class_name   = class_names[top_class_index]
            top_confidence   = pred_probs[top_class_index] * 100

        # ----- Resize BOTH images to identical display size -----
        DISPLAY_SIZE = (300, 300)  # width, height in px

        original_display  = original_image.resize(DISPLAY_SIZE,  Image.Resampling.LANCZOS)
        processed_display = processed_display_img.resize(DISPLAY_SIZE, Image.NEAREST)

        # Row 1: Original & Processed Images
        st.header("Image Analysis")
        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.image(original_display,  caption="Original Uploaded Image",  width=DISPLAY_SIZE[0])
        with img_col2:
            st.image(processed_display, caption="Processed Image (28×28, inverted)", width=DISPLAY_SIZE[0])

        st.markdown("""<hr style="height:1px;border:none;color:#6E6E6E;background:#6E6E6E;">""", unsafe_allow_html=True)

        # Row 2: Prediction Result & Probability Bar Chart
        st.header("Prediction Results")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.subheader("Top Prediction")
            st.success(f"This looks like a **{top_class_name}**.")
            st.write(f"Confidence: **{top_confidence:.2f}%**")

        with res_col2:
            st.subheader("Confidence Scores")

            sorted_idx   = np.argsort(pred_probs)[::-1]
            sorted_names = [class_names[i] for i in sorted_idx]
            sorted_probs = pred_probs[sorted_idx]

            fig, ax = plt.subplots()
            bars = ax.barh(sorted_names, sorted_probs, color="skyblue")
            ax.set_xlabel("Probability")
            ax.set_xlim(0, 1)
            ax.invert_yaxis()

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f"{width:.1%}", va="center")

            st.pyplot(fig)

    else:
        st.error("The model is not available. Please check the deployment logs.")
