# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps # <-- Import ImageOps for the inversion step
import requests
import tempfile
import random

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

# --- Data Loading for Testing ---
@st.cache_data
def load_fashion_mnist_test_data():
    """Loads the original Fashion MNIST test set for demonstration."""
    (_, _), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Normalize and reshape just like in training
    X_test = X_test.astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)
    return X_test, y_test

model = load_keras_model()
X_test, y_test_labels = load_fashion_mnist_test_data()

# --- Class Names ---
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Image Preprocessing for User Uploads (THE FIX IS HERE) ---
def preprocess_uploaded_image(image_pil):
    """
    Preprocesses the user-uploaded image to exactly match the
    successful Colab inference script's logic.
    """
    # 1. Convert to grayscale ('L' mode)
    grayscale_img = image_pil.convert('L')

    # 2. Resize to 28x28 pixels using LANCZOS resampling
    resized_img = grayscale_img.resize((28, 28), Image.Resampling.LANCZOS)

    # 3. Invert the image colors. This is the crucial fix from your Colab script.
    inverted_img = ImageOps.invert(resized_img)

    # 4. Convert to a numpy array and normalize to [0, 1]
    img_array = np.array(inverted_img).astype('float32') / 255.0

    # 5. Reshape the array to (1, 28, 28, 1) for the model
    img_batch = img_array.reshape(1, 28, 28, 1)

    return img_batch

# --- Prediction Function ---
def predict(image_tensor):
    """Runs prediction on a processed image tensor."""
    if model is None:
        return None, None
    prediction = model.predict(image_tensor)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(prediction) * 100
    return predicted_class_name, confidence, prediction

# --- Streamlit App Interface (Layout Preserved) ---

st.title("👗 Fashion Item Classifier")
st.markdown("This app uses a CNN to classify images of clothing from the Fashion MNIST dataset.")

# --- Sidebar ---
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1.  **Test the model** with a sample image from the original dataset.
2.  **Upload your own image** to see how the model performs on real-world photos.
""")

st.sidebar.header("Test with a Sample Image")
if st.sidebar.button("Show Random Test Image"):
    # Select a random index and store data in session state
    random_index = random.randint(0, len(X_test) - 1)
    st.session_state.test_image = X_test[random_index]
    st.session_state.true_label_name = class_names[y_test_labels[random_index]]
    # Clear any previous user upload when testing a sample
    if 'user_upload' in st.session_state:
        del st.session_state['user_upload']

st.sidebar.header("Upload Your Own Image")
uploaded_file = st.sidebar.file_uploader(" ", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Store user upload in session state
    st.session_state.user_upload = uploaded_file
    # Clear any previous test image when a user uploads
    if 'test_image' in st.session_state:
        del st.session_state['test_image']

# --- Main Panel for Displaying Results ---
col1, col2 = st.columns(2)
image_to_predict = None
caption_text = ""

# Determine which image to show and process
if 'user_upload' in st.session_state:
    pil_image = Image.open(st.session_state.user_upload)
    image_to_predict = preprocess_uploaded_image(pil_image)
    caption_text = "Your Uploaded Image"
    with col1:
        st.image(pil_image, caption=caption_text, use_column_width=True)

elif 'test_image' in st.session_state:
    image_to_predict = st.session_state.test_image
    caption_text = f"Sample Test Image (True Label: {st.session_state.true_label_name})"
    with col1:
        st.image(image_to_predict, caption=caption_text, use_column_width=True)

# Perform prediction if an image has been selected
if image_to_predict is not None:
    with st.spinner('Classifying...'):
        predicted_class, confidence, probabilities = predict(image_to_predict)

    with col2:
        if predicted_class:
            st.subheader("Prediction Result")
            if 'true_label_name' in st.session_state and 'user_upload' not in st.session_state:
                st.write(f"**True Label:** {st.session_state.true_label_name}")

            st.success(f"**Predicted:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            st.subheader("Prediction Probabilities")
            prob_data = { "Fashion Item": class_names, "Probability": [f"{p*100:.2f}%" for p in probabilities[0]] }
            st.dataframe(prob_data, use_container_width=True)
        else:
            st.error("Model could not make a prediction.")
else:
    with col1:
        st.info("👈 Use the sidebar to test a sample image or upload your own.")
        st.image("https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png", caption="Examples from the Fashion-MNIST dataset the model was trained on.")
