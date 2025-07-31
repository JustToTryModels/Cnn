# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# --- Configuration ---
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👕",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Model Loading ---
# We use st.cache_resource to load the model only once and cache it.
@st.cache_resource
def load_keras_model():
    """
    Loads the pre-trained Keras model from a GitHub URL.
    Uses caching to avoid reloading the model on every interaction.
    """
    # The URL to the raw model file on GitHub
    model_url = "https://github.com/JustToTryModels/Cnn/raw/main/Model/fashion_mnist_best_model.keras"
    
    try:
        # Download the model file
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Load the model from the downloaded content
        model_file = BytesIO(response.content)
        model = tf.keras.models.load_model(model_file)
        return model
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading the model: {e}")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_keras_model()

# --- Class Names ---
# These must match the order from your training script
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Image Preprocessing ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    - Converts to grayscale
    - Resizes to 28x28
    - Normalizes pixel values
    - Reshapes for the model
    """
    # Convert the image to grayscale ('L' mode)
    img = image.convert('L')
    
    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Normalize the pixel values to the [0, 1] range
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape the array to (1, 28, 28, 1) to match the model's input shape
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

# --- Streamlit App Interface ---
st.title("👗 Fashion MNIST Image Classifier")
st.markdown("""
    Welcome to the Fashion Classifier! 
    
    Upload an image of a clothing item, and the model will predict its category.
    This app uses a Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset.
""")

st.sidebar.header("About")
st.sidebar.info("""
    **Model:** Advanced CNN with Batch Normalization and Dropout.
    **Dataset:** Fashion MNIST
    **Frameworks:** TensorFlow/Keras & Streamlit
    **Source Code:** [GitHub Repository](https://github.com/JustToTryModels/Cnn)
""")

uploaded_file = st.file_uploader("Choose an image of a fashion item...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image and make a prediction
    with st.spinner('Classifying...'):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(prediction) * 100

    # Display the prediction result
    with col2:
        st.subheader("Prediction Result")
        st.success(f"This looks like a **{predicted_class_name}**.")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.subheader("Prediction Probabilities")
        # Create a neat table for probabilities
        prob_data = {
            "Fashion Item": class_names,
            "Probability": [f"{p*100:.2f}%" for p in prediction[0]]
        }
        st.dataframe(prob_data, use_container_width=True)

elif model is None:
    st.error("Model could not be loaded. Please check the logs or contact the administrator.")
