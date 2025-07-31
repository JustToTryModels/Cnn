# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import requests
import tempfile
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* 1.  “Browse files” button  -------------------------------------------------*/
[data-testid="stFileUploader"] button{
    background:linear-gradient(90deg,#ff8a00,#e52e71);
    color:white;
    border:none;
    border-radius:25px;
    padding:10px 24px;
    font-size:1.1em;
    font-weight:bold;
    cursor:pointer;
    transition:transform .2s ease,box-shadow .2s ease;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:focus,
[data-testid="stFileUploader"] button:active,
[data-testid="stFileUploader"] button:visited,
[data-testid="stFileUploader"] button *{color:white!important;}

[data-testid="stFileUploader"] button:hover{
    transform:scale(1.05);
    box-shadow:0 5px 15px rgba(229,46,113,.4);
}
[data-testid="stFileUploader"] button:active{transform:scale(.98);}

/* 2.  Uploaded–file “chip” ---------------------------------------------------*/
[data-testid="stFileUploaderFile"]{
    display:flex;
    align-items:center;
    background:#4A4A4A;
    color:white;
    border-radius:25px;
    padding:4px 12px;
    transition:box-shadow .2s ease;
}
[data-testid="stFileUploaderFile"]>div:first-of-type{
    color:white!important;
    font-size:.9em;
    padding-right:10px;
}

/* 3.  DELETE (×) BUTTON — gradient colour, pill shape -----------------------*/
[data-testid="stFileUploaderFile"] button{
    background:linear-gradient(90deg,#ff8a00,#e52e71);
    border:none;
    border-radius:25px;            /*  <- pill (matches Browse button)  */
    padding:4px 8px;
    cursor:pointer;
    transition:transform .2s ease,box-shadow .2s ease;
    display:flex;align-items:center;justify-content:center;
}
[data-testid="stFileUploaderFile"] button:hover{
    transform:scale(1.05);
    box-shadow:0 5px 15px rgba(229,46,113,.4);
}
[data-testid="stFileUploaderFile"] button:active{transform:scale(.92);}

/*  × icon colour — always white                                              */
[data-testid="stFileUploaderFile"] button svg{fill:white!important;}

/* Chip focus outline ---------------------------------------------------------*/
[data-testid="stFileUploaderFile"]:focus-within{
    box-shadow:0 0 0 2px rgba(229,46,113,.6);
    outline:none;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_keras_model():
    """Load pre-trained model from GitHub."""
    url = ("https://github.com/JustToTryModels/Cnn/raw/main/Model/"
           "fashion_mnist_best_model.keras")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                for chunk in r.iter_content(8192):
                    tmp.write(chunk)
                path = tmp.name
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error("Error loading model.")
        st.exception(e)
        return None

model = load_keras_model()

# ── Class names ───────────────────────────────────────────────────────────────
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# ── Image helper ──────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image):
    """Convert to 28×28 grayscale, invert & normalise."""
    gray  = image.convert("L")
    small = gray.resize((28, 28), Image.Resampling.LANCZOS)
    inv   = ImageOps.invert(small)

    arr = np.asarray(inv).astype("float32")/255.0
    return inv, arr.reshape(1, 28, 28, 1)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("👗 Fashion MNIST Image Classifier")
st.markdown("Upload an image of a clothing item and the model will predict its category.")
st.markdown("💡 **Tip: Centred images with a plain background work best.**")

st.sidebar.header("About")
st.sidebar.info(
    "**Model**: CNN with batch norm & dropout  \n"
    "**Dataset**: Fashion-MNIST  \n"
    "**Frameworks**: TensorFlow/Keras & Streamlit  \n"
    "**Code**: [GitHub](https://github.com/JustToTryModels/Cnn)"
)

uploaded_file = st.file_uploader(
    "Choose an image of a fashion item…", type=["jpg", "jpeg", "png"]
)

# ── Main logic ───────────────────────────────────────────────────────────────
if uploaded_file:
    if model:
        orig_img = Image.open(uploaded_file)

        with st.spinner("Classifying…"):
            proc_disp_img, proc_for_model = preprocess_image(orig_img)
            preds      = model.predict(proc_for_model)[0]
            top_idx    = np.argmax(preds)
            top_name   = class_names[top_idx]
            top_conf   = preds[top_idx] * 100

        # identical display size
        DISP = (300, 300)
        orig_show = orig_img.resize(DISP, Image.Resampling.LANCZOS)
        proc_show = proc_disp_img.resize(DISP, Image.NEAREST)

        # ── Row 1 : images + arrow ───────────────────────────────────────────
        st.header("1. Image Analysis 🖼️")
        col1, col_arrow, col2 = st.columns([3, 1, 3])   # ← added middle column for arrow

        with col1:
            c1, c2, c3 = st.columns([1,3,1])
            with c2:
                st.image(orig_show, width=DISP[0])
                st.markdown(
                    '<p style="text-align:center;">Original Uploaded Image</p>',
                    unsafe_allow_html=True)

        # ----------  Arrow (new) ----------
        with col_arrow:
            st.markdown(
                "<h1 style='text-align:center; font-size: 64px; margin-top: 100px;'>"
                "➡️"
                "</h1>",
                unsafe_allow_html=True,
            )
        # -----------------------------------

        with col2:
            c1, c2, c3 = st.columns([1,3,1])
            with c2:
                st.image(proc_show, width=DISP[0])
                st.markdown(
                    '<p style="text-align:center;">Processed Image (28×28, inverted)</p>',
                    unsafe_allow_html=True)

        st.markdown('<hr style="height:1px;border:none;background:#6E6E6E;">',
                    unsafe_allow_html=True)

        # ── Row 2 : results ─────────────────────────────────────────────────
        st.header("2. Prediction Results ✨")
        # Create three columns: content, divider, content
        r1, r_divider, r2 = st.columns([2, 0.2, 2])

        with r1:
            st.subheader("2.1. Top Prediction")
            st.success(f"This looks like a **{top_name}**.")
            st.write(f"Confidence: **{top_conf:.2f}%**")

        # Add the vertical line in the middle column
        with r_divider:
            st.markdown(
                "<div style='border-left: 1px solid #6E6E6E; height: 350px; margin: auto;'></div>",
                unsafe_allow_html=True,
            )

        with r2:
            st.subheader("2.2. Confidence Scores")
            order  = np.argsort(preds)[::-1]
            names  = [class_names[i] for i in order]
            probs  = preds[order]

            fig, ax = plt.subplots()
            bars = ax.barh(names, probs, color="skyblue")
            ax.set_xlabel("Probability")
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            for bar in bars:
                w = bar.get_width()
                ax.text(w+0.01, bar.get_y()+bar.get_height()/2,
                        f"{w:.1%}", va="center")
            st.pyplot(fig)
    else:
        st.error("The model is not available. Please check the deployment logs.")
