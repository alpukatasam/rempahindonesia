import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Konfigurasi awal Aplikasi ---
st.set_page_config(
    page_title="Aplikasi Prediksi Rempah Indonesia",
    page_icon="🌿",
    layout="wide"
)

# --- Injeksi CSS Global untuk sidebar ---
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

# --- Inisialisasi Session State untuk navigasi ---
if "nav" not in st.session_state:
    st.session_state.nav = "Halaman Utama"  # default halaman

# --- Sidebar ---
with st.sidebar:
    # Grup pertama: Pilih Halaman
    st.markdown("<div class='sidebar-header'>Pilih Halaman</div>", unsafe_allow_html=True)
    if st.button("Halaman Utama"):
        st.session_state.nav = "Halaman Utama"
    st.markdown("<div class='sidebar-separator'></div>", unsafe_allow_html=True)
    
    # Grup kedua: Pilih Metode Deteksi
    st.markdown("<div class='sidebar-header'>Pilih Metode Deteksi</div>", unsafe_allow_html=True)
    if st.button("Upload Gambar"):
        st.session_state.nav = "Upload Gambar"
    if st.button("Tangkap Gambar"):
        st.session_state.nav = "Tangkap Gambar"

# Ambil pilihan navigasi dari session state
pilihan = st.session_state.nav

# --- Header Banner ---
with st.container():
    # Menggunakan kolom dengan rasio 1:2:1 agar lebih simetris
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image("img/spiceleft.png", width=100)  # Gambar di kiri diperkecil
    with col2:
        st.markdown("<h1 style='text-align: center;'>Aplikasi Prediksi Rempah Indonesia</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #444;'>Sistem deteksi dan pengenalan rempah berbasis deep learning</p>", unsafe_allow_html=True)
    with col3:
        st.image("img/spiceright.png", width=100)  # Gambar di kanan diperkecil

# --- Load Model ---
model_path = r'models/best_model.h5'
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

model = load_model_local(model_path)

# --- Daftar Kelas Rempah ---
class_names = [
    "adas", "andaliman", "asam jawa", "bawang bombai", "bawang merah", "bawang putih",
    "biji ketumbar", "bukan rempah", "bunga lawang", "cengkeh", "daun jeruk", "daun kemangi",
    "daun ketumbar", "daun salam", "jahe", "jinten", "kapulaga", "kayu manis", "kayu secang",
    "kemiri", "kemukus", "kencur", "kluwek", "kunyit", "lada", "lengkuas", "pala",
    "saffron", "serai", "vanili", "wijen"
]

# --- Fungsi untuk Preprocessing Gambar & Prediksi ---
def process_image(source):
    try:
        image = Image.open(source)
    except Exception:
        st.error("⚠️ Gagal membuka gambar. Pastikan format gambar benar.")
        st.stop()
    st.image(image, caption="🖼️ Gambar yang Digunakan", use_container_width=True)
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized)
    if img_array.ndim == 2:  # jika grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # jika ada channel alpha
        img_array = img_array[..., :3]
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_and_display(img_array):
    preds = model.predict(img_array)
    top_index = np.argmax(preds[0])
    st.success(f"🔍 Prediksi: **{class_names[top_index]}** dengan confidence **{preds[0][top_index]:.2f}**")
    st.markdown("### 🔍 Top-3 Kelas:")
    sorted_indices = np.argsort(preds[0])[::-1][:3]
    for idx in sorted_indices:
        st.write(f"- {class_names[idx]}: {preds[0][idx]:.2f}")

# --- Konten Utama Berdasarkan Navigasi ---
# --- Konten Utama Berdasarkan Navigasi ---
if pilihan == "Halaman Utama":
    # Menggunakan st.markdown untuk menampilkan header dengan HTML agar bisa diatur posisi tengah
    st.markdown(
        """
        <h1 style="text-align: center;">Selamat Datang</h1>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="text-align: justify;">
            <p>
                Selamat datang di Aplikasi Prediksi Rempah Indonesia. Aplikasi ini dirancang dengan teknologi deep learning yang canggih untuk mendeteksi 
            dan mengenali berbagai jenis rempah yang ada di Nusantara. Melalui aplikasi ini, Anda dapat memahami keberagaman rempah-rempah yang tidak hanya memiliki 
            rasa dan aroma yang khas, tetapi juga manfaat kesehatan yang luar biasa. Di halaman ini, kami menyajikan sejumlah informasi yang mendalam mengenai 
            rempah-rempah, termasuk asal usul, kegunaan dalam pengobatan tradisional, dan peran penting rempah dalam dunia kuliner yang terus berkembang.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.image("https://assets.corteva.com/is/image/Corteva/ar2-17mar20?$image_desktop$",
             caption="Berbagai Rempah Khas Indonesia", use_container_width=True)


elif pilihan == "Upload Gambar":
    st.header("Deteksi Rempah – Upload Gambar")
    st.markdown("Silakan unggah gambar rempah dalam format **jpg**, **jpeg**, atau **png**.")
    upload_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    if upload_file:
        img_array = process_image(upload_file)
        predict_and_display(img_array)
    
elif pilihan == "Tangkap Gambar":
    st.header("Deteksi Rempah – Kamera")
    st.markdown("Silakan ambil gambar langsung menggunakan kamera perangkat Anda.")
    camera_image = st.camera_input("Ambil Gambar")
    if camera_image:
        img_array = process_image(camera_image)
        predict_and_display(img_array)
