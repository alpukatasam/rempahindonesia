import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

# --- Konfigurasi Awal Aplikasi ---
st.set_page_config(
    page_title="Aplikasi Prediksi Rempah Indonesia",
    page_icon="🌿",
    layout="wide"
)

# --- Injeksi CSS Global untuk Sidebar ---
st.markdown(
    """
    <style>
    .sidebar-header {
        font-weight: bold !important;
        font-size: 16px !important;
        color: #444444 !important;
        margin-top: 1.2em !important;
        margin-bottom: 0.3em !important;
        line-height: 1.2 !important;
    }
    .sidebar-separator {
        margin-top: 1em !important;
        margin-bottom: 1em !important;
        border-bottom: 1px solid #cccccc !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Inisialisasi Navigasi dengan Radio (Lebih Sederhana) ---
pilihan = st.sidebar.radio(
    "Pilih Halaman / Metode Deteksi",
    ["Halaman Utama", "Upload Gambar", "Tangkap Gambar"]
)

# --- Header Banner ---
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image("img/spiceleft.png", width=100)
    with col2:
        st.markdown(
            "<h1 style='text-align: center;'>Aplikasi Prediksi Rempah Indonesia</h1>",
            unsafe_allow_html=True
        )
    with col3:
        st.image("img/spiceright.png", width=100)

# --- Load Model dengan Spinner dan Caching ---
model_path = "models/best_model.h5"

@st.cache_resource
def load_model_local(path):
    if not os.path.exists(path):
        st.error(f"❌ File model tidak ditemukan di {path}")
        st.stop()
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

with st.spinner("🔄 Memuat model deep learning..."):
    model = load_model_local(model_path)

# --- Daftar Kelas Rempah ---
class_names = [
    "adas", "andaliman", "asam jawa", "bawang bombai", "bawang merah", "bawang putih",
    "biji ketumbar", "bukan rempah", "bunga lawang", "cengkeh", "daun jeruk", "daun kemangi",
    "daun ketumbar", "daun salam", "jahe", "jinten", "kapulaga", "kayu manis", "kayu secang",
    "kemiri", "kemukus", "kencur", "kluwek", "kunyit", "lada", "lengkuas", "pala",
    "saffron", "serai", "vanili", "wijen"
]

# --- Fungsi Preprocessing dan Prediksi ---
def process_image(source):
    try:
        img = Image.open(source)
    except Exception:
        st.error("⚠️ Gagal membuka gambar. Pastikan file benar.")
        st.stop()
    st.image(img, caption="🖼️ Gambar yang Digunakan", use_container_width=True)
    img = img.resize((224, 224))
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack((arr,) * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_and_display(img_array):
    preds = model.predict(img_array)[0]
    # Ambil 3 prediksi tertinggi
    top3_idx = np.argsort(preds)[::-1][:3]
    df_top3 = pd.DataFrame({
        "Kelas": [class_names[i] for i in top3_idx],
        "Confidence (%)": preds[top3_idx] * 100  # ubah ke persentase
    })
    # Tampilkan Prediksi Utama
    st.success(
        f"🔍 Prediksi Utama: **{df_top3.loc[0, 'Kelas']}** "
        f"({df_top3.loc[0, 'Confidence (%)']:.2f}%)"
    )
    # Tabel Top-3
    st.markdown("### 🔍 Tiga Prediksi Teratas")
    st.table(df_top3.style.format({"Confidence (%)": "{:.2f}%"}))
    # Bar chart
    st.bar_chart(df_top3.set_index("Kelas"))
# --- Konten Halaman Berdasarkan Pilihan ---
if pilihan == "Halaman Utama":
    st.markdown("<h1 style='text-align: center;'>Selamat Datang</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: justify;">
        Aplikasi ini memanfaatkan deep learning untuk mendeteksi dan mengenali rempah-rempah khas Indonesia.
        Telusuri keanekaragaman rempah, pelajari kegunaan tradisional, dan uji model AI kami dengan foto Anda sendiri!
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image(
        "https://assets.corteva.com/is/image/Corteva/ar2-17mar20?$image_desktop$",
        caption="Berbagai Rempah Khas Indonesia",
        use_container_width=True
    )

elif pilihan == "Upload Gambar":
    st.header("Deteksi Rempah – Upload Gambar")
    upload_file = st.file_uploader("Unggah Gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
    if upload_file:
        img_arr = process_image(upload_file)
        predict_and_display(img_arr)

elif pilihan == "Tangkap Gambar":
    st.header("Deteksi Rempah – Kamera")
    camera_img = st.camera_input("Ambil Gambar")
    if camera_img:
        img_arr = process_image(camera_img)
        predict_and_display(img_arr)

