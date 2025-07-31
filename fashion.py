# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import requests
import tempfile

# --- Configuration ---
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="auto",
)

# --- Model Loading ---
@st.cache_resource
def load_keras_model():
    """
    Loads the pre-trained Keras model from a GitHub URL.
    Saves the model to a temporary file before loading.
    """
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
        return None

model = load_keras_model()

# --- Class Names ---
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Image Preprocessing (Replicating the Colab Logic) ---
def preprocess_uploaded_image(image_pil):
    """
    Preprocesses the user-uploaded image to exactly match the
    successful Colab inference script's logic.
    """
    # 1. Convert to grayscale ('L' mode)
    grayscale_img = image_pil.convert('L')

    # 2. Resize to 28x28 pixels using LANCZOS resampling
    resized_img = grayscale_img.resize((28, 28), Image.Resampling.LANCZOS)

    # 3. Invert the image colors. This is the crucial step.
    #    Fashion MNIST has white items on a black background.
    #    This step ensures user-uploaded images match that format.
    inverted_img = ImageOps.invert(resized_img)

    # 4. Convert to a numpy array and normalize to [0, 1]
    img_array = np.array(inverted_img).astype('float32') / 255.0

    # 5. Reshape the array to (1, 28, 28, 1) for the model
    img_batch = img_array.reshape(1, 28, 28, 1)

    # Return the processed image for display and the model input
    return inverted_img, img_batch

# --- Streamlit App Interface ---

st.title("👗 Fashion Item Classifier")
st.markdown("Upload an image of a clothing item. The model will classify it into one of the 10 Fashion-MNIST categories.")
st.markdown("💡 **Tip:** For best results, use images with a single, centered item and a plain background.")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a JPG or PNG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload an image using the sidebar to begin.")

if uploaded_file is not None and model is not None:
    # Open the user's image
    original_image = Image.open(uploaded_file)

    # Process the image using the corrected function
    processed_image_for_display, model_input = preprocess_uploaded_image(original_image)

    # Make prediction
    with st.spinner("🧠 Analyzing the image..."):
        prediction = model.predict(model_input)
        pred_probs = prediction[0]

    # Get the top prediction
    top_class_index = np.argmax(pred_probs)
    top_class_name = class_names[top_class_index]
    top_confidence = pred_probs[top_class_index]

    # --- Display Results ---
    st.header("Analysis Result")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image Processing")
        st.image(original_image, caption="1. Your Original Uploaded Image", use_column_width=True)
        st.image(processed_image_for_display, caption="2. Processed Image (Grayscale, 28x28, Inverted)", use_column_width=True)

    with col2:
        st.subheader("Prediction")
        st.success(f"**I'm {top_confidence:.1%} confident this is a {top_class_name}.**")

        # Sort predictions for the detailed report
        sorted_indices = np.argsort(pred_probs)[::-1]
        
        st.subheader("Detailed Confidence Report")
        
        results_data = {
            "Category": [class_names[i] for i in sorted_indices],
            "Confidence": [f"{pred_probs[i]*100:.2f}%" for i in sorted_indices]
        }
        st.dataframe(results_data, use_container_width=True)

elif model is None:
    st.error("Model could not be loaded. Please check deployment logs.")
