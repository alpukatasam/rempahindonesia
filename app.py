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

# 2. Muat daftar kelas dari folder data_split/train
@st.cache_data
def load_class_names():
    split_train = os.path.join(os.path.dirname(__file__), "data_split", "train")
    classes = sorted([
        d for d in os.listdir(split_train)
        if os.path.isdir(os.path.join(split_train, d))
    ])
    return classes

class_names = load_class_names()

# 3. Fungsi preprocess & predict
def predict_image(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)  # shape (1,224,224,3)

    preds = model.predict(x)[0]
    top_idx = np.argmax(preds)
    return class_names[top_idx], float(preds[top_idx])

# 4. UI
st.set_page_config(page_title="Klasifikasi Rempah", layout="centered")

st.title("🔍 Klasifikasi Rempah Indonesia")
st.write("Upload gambar rempah atau ambil dengan kamera, lalu aplikasi akan memprediksi jenis rempah.")

# Pilih metode input
method = st.radio("Pilih metode input:", ["Upload Gambar", "Kamera"])

img = None
if method == "Upload Gambar":
    file = st.file_uploader("Pilih file gambar", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file)

else:  # Kamera
    cam = st.camera_input("Ambil foto rempah")
    if cam:
        img = Image.open(cam)

# Tampilkan dan prediksi
if img:
    st.image(img, caption="Gambar input", use_container_width=True)
    st.write("⏳ Memproses...")
    label, score = predict_image(img)
    st.success(f"**Prediksi:** {label}  \n**Confidence:** {score:.2%}")
