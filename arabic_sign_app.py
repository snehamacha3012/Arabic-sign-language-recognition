
import streamlit as st
import pickle, numpy as np, os, io, warnings, tensorflow as tf
from PIL import Image

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Arabic Sign Classifier", page_icon="🤟", layout="wide")

@st.cache_resource
def load_all():
    # Load model (keras preferred, h5 fallback)
    if os.path.exists("arabic_sign.keras"):
        model = tf.keras.models.load_model("arabic_sign.keras", compile=False)
    else:
        model = tf.keras.models.load_model("arabic_sign.h5", compile=False)

    # Metadata and ordered labels
    with open("arabic_sign.pkl", "rb") as f:
        meta = pickle.load(f)
    idx2label = {int(k): v for k, v in meta.get("idx2label").items()}
    labels = [lbl for _, lbl in sorted(idx2label.items(), key=lambda kv: kv[0])]  # correct order
    img_size = meta.get("img_size", 224)
    val_acc = meta.get("val_accuracy")

    # Warm-up
    dummy = tf.zeros((1, img_size, img_size, 3), dtype=tf.float32)
    _ = model.predict(dummy, verbose=0)

    return model, labels, img_size, val_acc

def preprocess(pil_img, img_size):
    # Only scale to [0,1]; model already handles preprocess_input internally
    img = pil_img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict(pil_img, model, labels, img_size):
    batch = preprocess(pil_img, img_size)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx]), probs

with st.spinner("Loading model (~20s first time, now warmed)..."):
    model, LABELS, IMG_SIZE, val_acc = load_all()

st.markdown("<h1 style='text-align:center'>🤟 Arabic Sign Classifier</h1>", unsafe_allow_html=True)
subtitle = "MobileNetV2 · 28 classes"
if val_acc is not None:
    subtitle += f" · Val Acc: {val_acc*100:.1f}%"
st.markdown(f"<p style='text-align:center;color:gray'>{subtitle}</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([1,1], gap="large")

with col1:
    st.subheader("Upload a hand sign")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","webp","bmp"], label_visibility="collapsed")
    use_img = Image.open(uploaded) if uploaded else None
    if use_img:
        st.image(use_img, caption="Selected image", use_column_width=True)
        st.caption(f"Resized to {IMG_SIZE}×{IMG_SIZE} for inference")

with col2:
    st.subheader("Prediction")
    if use_img:
        with st.spinner("Analysing..."):
            pred_cls, conf, probs = predict(use_img, model, LABELS, IMG_SIZE)
        st.success(f"Prediction: **{pred_cls}** — {conf*100:.2f}%")
        st.markdown("**Top 5 probabilities:**")
        top5 = sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)[:5]
        for cls, p in top5:
            c1, c2, c3 = st.columns([2,5,1])
            c1.write(cls)
            c2.progress(float(p))
            c3.write(f"{p*100:.1f}%")
    else:
        st.info("Upload an image to get a prediction.")

st.sidebar.title("Model Info")
st.sidebar.write(f"Classes: {len(LABELS)}")
st.sidebar.write("Backbone: MobileNetV2")
if val_acc is not None:
    st.sidebar.write(f"Val Accuracy: {val_acc*100:.2f}%")
st.sidebar.caption("For demo/education. Not a clinical tool.")
