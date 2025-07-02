import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# 1. Load model (cache agar hanya sekali saja)
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()

# 2. Hard-code class names
CLASS_NAMES = [
    "adas", "andaliman", "asam jawa", "bawang bombai", "bawang merah", "bawang putih",
    "biji ketumbar", "bukan rempah", "bunga lawang", "cengkeh", "daun jeruk", "daun kemangi",
    "daun ketumbar", "daun salam", "jahe", "jinten", "kapulaga", "kayu manis", "kayu secang",
    "kemiri", "kemukus", "kencur", "kluwek", "kunyit", "lada", "lengkuas", "pala",
    "saffron", "serai", "vanili", "wijen"
]

# 3. Prediction function menggunakan CLASS_NAMES
def predict_image(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    x = preprocess_input(np.array(img, dtype=np.float32))
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx])

# 4. Streamlit UI
st.set_page_config(page_title="Klasifikasi Rempah", layout="centered")

st.title("🔍 Klasifikasi Rempah Indonesia")
st.write("Gunakan upload gambar atau kamera untuk memprediksi jenis rempah.")

method = st.radio("Pilih metode input:", ["Upload Gambar", "Kamera"])

img = None
if method == "Upload Gambar":
    file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file)
else:
    img_data = st.camera_input("Ambil foto rempah")
    if img_data:
        img = Image.open(img_data)

if img:
    st.image(img, caption="Gambar input", use_container_width=True)
    st.write("⏳ Memproses...")
    label, score = predict_image(img)
    st.success(f"**Prediksi:** {label}  \n**Confidence:** {score:.2%}")
