import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from utils import download_model_from_gdrive

# --- Konfigurasi Model ---
model_path = 'efficientnetv2_rempahindo.keras'
file_id = '16mspKSZnXI3x2ENrW_FmOmei5Y_u0Z_8'

# --- Load model dengan cache + error handler ---
@st.cache_resource
def load_model_from_drive(file_id, model_path):
    if not os.path.exists(model_path):
        st.write("📥 Mengunduh model dari Google Drive...")
        download_model_from_gdrive(file_id, output_path=model_path)
        st.write(f"✅ Model berhasil diunduh: {model_path}")
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

model = load_model_from_drive(file_id, model_path)

# --- Daftar kelas ---
class_names = [
    "adas", "andaliman", "asam jawa", "bawang bombai", "bawang merah", "bawang putih", 
    "biji ketumbar", "bukan rempah", "bunga lawang", "cengkeh", "daun jeruk", "daun kemangi", 
    "daun ketumbar", "daun salam", "jahe", "jinten", "kapulaga", "kayu manis", "kayu secang", 
    "kemiri", "kemukus", "kencur", "kluwek", "kunyit", "lada", "lengkuas", "pala", 
    "saffron", "serai", "vanili", "wijen"
]

# --- UI Aplikasi ---
st.title("🌿 Aplikasi Prediksi Rempah Indonesia")
st.write("Pilih metode input gambar:")

input_method = st.radio("Metode input:", ["Gunakan Kamera", "Unggah Gambar"], index=1)

image_source = None
if input_method == "Gunakan Kamera":
    image_source = st.camera_input("Ambil gambar:")
elif input_method == "Unggah Gambar":
    image_source = st.file_uploader("Unggah gambar:", type=["jpg", "jpeg", "png"])

if image_source:
    try:
        image = Image.open(image_source)
        st.image(image, caption="🖼️ Gambar yang digunakan", use_column_width=True)
    except Exception:
        st.error("⚠️ Gagal membuka gambar. Pastikan formatnya benar.")
    else:
        # --- Preprocessing gambar ---
        image_resized = image.resize((224, 224))
        img_array = np.array(image_resized)

        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # --- Prediksi ---
        preds = model.predict(img_array)
        top_pred_index = np.argmax(preds[0])
        top_pred_score = float(preds[0][top_pred_index])
        top_pred_class = class_names[top_pred_index]

        sorted_indices = np.argsort(preds[0])[::-1][:3]
        top_classes = [class_names[i] for i in sorted_indices]
        top_scores = [preds[0][i] for i in sorted_indices]

        # --- Output Prediksi ---
        st.markdown("### 🧪 Hasil Prediksi (Top-1):")
        st.success(f"**{top_pred_class}** dengan confidence: **{top_pred_score:.2f}**")

        st.markdown("### 🔍 Top-3 Prediksi:")
        for i in range(3):
            st.write(f"{i+1}. {top_classes[i]}: {top_scores[i]:.2f}")

        # --- Koreksi dari user ---
        st.markdown("### ✏️ Koreksi (Feedback)")
        feedback = st.selectbox("Jika prediksi salah, pilih label yang benar:", class_names)
        if st.button("Kirim Feedback"):
            st.info(f"Feedback dikirim! Label seharusnya: **{feedback}**. Terima kasih.")