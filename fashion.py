# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import requests, tempfile
import matplotlib.pyplot as plt

# ──────────────────────────────  CONFIG  ─────────────────────────────────────
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👕",
    layout="wide",
)

# ───────────────────────────  CUSTOM  CSS  ───────────────────────────────────
st.markdown(
    """
<style>
/* 1) “Browse files” button ---------------------------------------------------*/
[data-testid="stFileUploader"] button{
    background:linear-gradient(90deg,#ff8a00,#e52e71);
    color:#fff;border:none;border-radius:25px;
    padding:10px 24px;font-size:1.05rem;font-weight:600;
    cursor:pointer;transition:transform .2s,box-shadow .2s;
}
[data-testid="stFileUploader"] button:hover{
    transform:scale(1.05);
    box-shadow:0 5px 15px rgba(229,46,113,.4);
}
[data-testid="stFileUploader"] button:active{transform:scale(.97);}
[data-testid="stFileUploader"] button *,
[data-testid="stFileUploader"] button{color:#fff!important;}

/* 2) Uploaded-file “chip” ----------------------------------------------------*/
[data-testid="stFileUploaderFile"]{
    display:flex;align-items:center;
    background:#4a4a4a;color:#fff;
    border-radius:25px;padding:4px 12px;
    transition:box-shadow .2s;
}

/*    2a) File-name -----------------------------------------------------------*/
[data-testid="stFileUploaderFile"]>div:first-of-type{
    color:#fff!important;font-size:.9rem;padding-right:10px;
}

/*    2b) File-size  (THIS LINE FIXES THE ISSUE) -----------------------------*/
[data-testid="stFileUploaderFile"]>div:first-of-type span{
    color:#ffffff !important;
}

/* 3) Delete (×) button -------------------------------------------------------*/
[data-testid="stFileUploaderFile"] button{
    background:linear-gradient(90deg,#ff8a00,#e52e71);
    border:none;border-radius:25px;padding:4px 8px;
    cursor:pointer;transition:transform .2s,box-shadow .2s;
    display:flex;align-items:center;justify-content:center;
}
[data-testid="stFileUploaderFile"] button:hover{
    transform:scale(1.05);
    box-shadow:0 5px 15px rgba(229,46,113,.4);
}
[data-testid="stFileUploaderFile"] button:active{transform:scale(.92);}
[data-testid="stFileUploaderFile"] button svg{fill:#fff!important;}

/* 4) Focus outline -----------------------------------------------------------*/
[data-testid="stFileUploaderFile"]:focus-within{
    box-shadow:0 0 0 2px rgba(229,46,113,.6);
    outline:none;
}
</style>
""",
    unsafe_allow_html=True,
)

# ────────────────────────  MODEL  LOAD / CACHE  ──────────────────────────────
@st.cache_resource
def load_keras_model():
    url = ("https://github.com/JustToTryModels/Cnn/raw/main/Model/"
           "fashion_mnist_best_model.keras")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                for chunk in r.iter_content(8192):
                    tmp.write(chunk)
                model_path = tmp.name
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error("Error loading model."); st.exception(e)
        return None

model = load_keras_model()

# ─────────────────────────────  CONSTANTS  ───────────────────────────────────
CLASS_NAMES = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# ────────────────────────────  UTILITIES  ────────────────────────────────────
def preprocess_image(img: Image.Image):
    gray  = img.convert("L")
    small = gray.resize((28, 28), Image.Resampling.LANCZOS)
    inv   = ImageOps.invert(small)                     # white foreground
    arr   = np.asarray(inv).astype("float32")/255.0    # 0-1
    return inv, arr.reshape(1,28,28,1)

# ───────────────────────────────  UI  ────────────────────────────────────────
st.title("👗 Fashion-MNIST Image Classifier")
st.markdown("Upload a clothing image & let the CNN guess its category.")
st.markdown("💡 *Tip: centred items on plain backgrounds work best.*")

with st.sidebar:
    st.header("About")
    st.info(
        "**Model**: CNN (batch-norm + dropout)  \n"
        "**Dataset**: Fashion-MNIST  \n"
        "**Frameworks**: TensorFlow / Keras / Streamlit  \n"
        "**Repo**: [GitHub](https://github.com/JustToTryModels/Cnn)"
    )

uploaded = st.file_uploader("Choose an image…", type=["png","jpg","jpeg"])

# ───────────────────────────── MAIN LOGIC ────────────────────────────────────
if uploaded:
    if model:
        orig = Image.open(uploaded)

        with st.spinner("Classifying…"):
            disp_img, model_img = preprocess_image(orig)
            preds   = model.predict(model_img)[0]
            top_idx = int(np.argmax(preds))
            top_cls = CLASS_NAMES[top_idx]
            top_conf= preds[top_idx]*100

        # same display dims
        W,H = 300,300
        st.header("Image Analysis")
        c1,c2 = st.columns(2)

        with c1:
            st.image(orig.resize((W,H), Image.Resampling.LANCZOS))
            st.caption("Original upload")
        with c2:
            st.image(disp_img.resize((W,H), Image.NEAREST))
            st.caption("Pre-processed (28×28, inverted)")

        st.divider()

        st.header("Prediction")
        l,r = st.columns(2)
        with l:
            st.subheader("Top guess")
            st.success(f"Looks like a **{top_cls}**")
            st.write(f"Confidence: **{top_conf:.2f}%**")
        with r:
            st.subheader("All probabilities")
            order = np.argsort(preds)[::-1]
            fig, ax = plt.subplots()
            ax.barh([CLASS_NAMES[i] for i in order], preds[order],
                    color="#4da6ff")
            ax.set_xlim(0,1); ax.invert_yaxis(); ax.set_xlabel("Probability")
            for bar in ax.patches:
                w = bar.get_width()
                ax.text(w+0.01, bar.get_y()+bar.get_height()/2,
                        f"{w:.1%}", va="center")
            st.pyplot(fig)
    else:
        st.error("Model not available. Check deployment logs.")
