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
    layout="wide",  # Use wide layout for better column spacing
    initial_sidebar_state="auto",
)

# --- Model Loading ---
@st.cache_resource
def load_keras_model():
    """
    Loads the pre-trained Keras model from a GitHub URL.
    This version saves the model to a temporary file before loading.
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
        st.exception(e)
        return None

model = load_keras_model()

# --- Class Names ---
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Image Preprocessing (MODIFIED TO RETURN BOTH IMAGES) ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image and returns both the displayable
    processed image and the numpy array for the model.
    """
    grayscale_img = image.convert('L')
    resized_img = grayscale_img.resize((28, 28), Image.Resampling.LANCZOS)
    inverted_img = ImageOps.invert(resized_img)
    
    # Prepare the array for the model
    img_array = np.array(inverted_img)
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Return both the displayable PIL image and the model-ready array
    return inverted_img, img_array

# --- Streamlit App Interface ---
st.title("👗 Fashion MNIST Image Classifier")
st.markdown("""
    Welcome to the Fashion Classifier! Upload an image of a clothing item, and the model will predict its category.
""")

st.sidebar.header("About")
st.sidebar.info("""
    **Model:** Advanced CNN with Batch Normalization and Dropout.
    **Dataset:** Fashion MNIST
    **Frameworks:** TensorFlow/Keras & Streamlit
    **Source Code:** [GitHub Repository](https://github.com/JustToTryModels/Cnn)
""")

uploaded_file = st.file_uploader("Choose an image of a fashion item...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is not None:
        original_image = Image.open(uploaded_file)
        
        # Preprocess the image and make a prediction
        with st.spinner('Classifying...'):
            # Get both the displayable image and the model input
            processed_image_for_display, processed_image_for_model = preprocess_image(original_image)
            
            prediction = model.predict(processed_image_for_model)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            confidence = np.max(prediction) * 100

        # --- NEW LAYOUT STRUCTURE ---

        # Row 1: Original and Processed Images
        st.header("Image Analysis")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(original_image, caption="Original Uploaded Image", use_container_width=True)
        with img_col2:
            st.image(processed_image_for_display, caption="Processed Image (28x28, Inverted)", use_container_width=True)

        st.divider() # Adds a horizontal line for separation

        # Row 2: Prediction Result and Probabilities Table
        st.header("Prediction Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("Top Prediction")
            st.success(f"This looks like a **{predicted_class_name}**.")
            st.write(f"Confidence: **{confidence:.2f}%**")
        with res_col2:
            st.subheader("All Probabilities")
            prob_data = {
                "Fashion Item": class_names,
                "Probability": [f"{p*100:.2f}%" for p in prediction[0]]
            }
            st.dataframe(prob_data, use_container_width=True)
            
    else:
        st.error("The model is not available. Please check the deployment logs.")
