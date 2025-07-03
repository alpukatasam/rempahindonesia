import os
import time
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Imports khusus EfficientNetV2-S
from tensorflow.keras.applications.efficientnet_v2 import (
    EfficientNetV2S,
    preprocess_input
)

# 1. Load model dengan custom_objects dan compile=False
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.h5")
    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"EfficientNetV2S": EfficientNetV2S}
    )

# Panggil load_model() sekali
model = load_model()

# 2. Daftar kelas
CLASS_NAMES = [
    "adas", "andaliman", "asam jawa", "bawang bombai", "bawang merah", "bawang putih",
    "biji ketumbar", "bukan rempah", "bunga lawang", "cengkeh", "daun jeruk", "daun kemangi",
    "daun ketumbar", "daun salam", "jahe", "jinten", "kapulaga", "kayu manis", "kayu secang",
    "kemiri", "kemukus", "kencur", "kluwek", "kunyit", "lada", "lengkuas", "pala",
    "saffron", "serai", "vanili", "wijen"
]

# 3. Streamlit UI
st.set_page_config(page_title="Klasifikasi Rempah", layout="centered")
st.title("🔍 Klasifikasi Rempah Indonesia")
st.write("Upload gambar atau gunakan kamera untuk memprediksi jenis rempah.")

method    = st.radio("Metode input:", ["Upload Gambar", "Kamera"])
show_top3 = st.checkbox("Tampilkan Top-3 Prediksi", value=False)

# Ambil gambar dari file uploader atau kamera
img = None
if method == "Upload Gambar":
    file = st.file_uploader("Pilih file", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file)
else:
    img_data = st.camera_input("Ambil foto rempah")
    if img_data:
        img = Image.open(img_data)

# Prediksi dan tampilkan hasil
if img:
    st.image(img, caption="Input", use_container_width=True)
    with st.spinner("Memproses…"):
        start     = time.time()
        array_img = np.array(img.convert("RGB").resize((224,224)), dtype=np.float32)
        x         = preprocess_input(array_img)
        preds     = model.predict(np.expand_dims(x,0))[0]
        elapsed_ms = (time.time() - start) * 1000

    if not show_top3:
        idx = np.argmax(preds)
        st.success(f"**Prediksi:** {CLASS_NAMES[idx]}\n\n**Confidence:** {preds[idx]:.2%}")
    else:
        st.write("**Top-3 Prediksi:**")
        for rank, j in enumerate(preds.argsort()[-3:][::-1], start=1):
            st.write(f"{rank}. {CLASS_NAMES[j]} — {preds[j]:.2%}")

    st.write(f"⏱️ Waktu inferensi: {elapsed_ms:.1f} ms")
