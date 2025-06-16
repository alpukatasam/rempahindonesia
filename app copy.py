import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Memuat model dengan caching menggunakan st.cache_resource (dari Streamlit versi terbaru)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("D:/Skripsi/scv2/models/efficientnetv2_rempahindo.keras")  # Gunakan forward slash untuk path
    return model

# Memuat model
model = load_model()
st.success("Model berhasil dimuat!")

# Daftar nama kelas sesuai dataset Anda
class_names = [
    "adas", "andaliman", "asam jawa", "bawang bombai", "bawang merah", "bawang putih", 
    "biji ketumbar", "bukan rempah", "bunga lawang", "cengkeh", "daun jeruk", "daun kemangi", 
    "daun ketumbar", "daun salam", "jahe", "jinten", "kapulaga", "kayu manis", "kayu secang", 
    "kemiri", "kemukus", "kencur", "kluwek", "kunyit", "lada", "lengkuas", "pala", 
    "saffron", "serai", "vanili", "wijen"
]

# Judul aplikasi
st.title("Aplikasi Prediksi Rempah dengan EfficientNetV2-S")
st.write("Pilih metode input gambar:")

# Pilihan metode input dengan default "Unggah Gambar"
input_method = st.radio("Pilih metode:", ["Gunakan Kamera", "Unggah Gambar"], index=1)

# Proses pemilihan sumber gambar berdasarkan input_method
image_source = None
if input_method == "Gunakan Kamera":
    camera_image = st.camera_input("Ambil gambar menggunakan kamera")
    if camera_image is not None:
        image_source = camera_image
elif input_method == "Unggah Gambar":
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_source = uploaded_file

if image_source is not None:
    try:
        # Membaca gambar menggunakan PIL
        image = Image.open(image_source)
        st.image(image, caption="Gambar yang digunakan", use_column_width=True)
    except Exception as e:
        st.error("Gagal membuka gambar. Pastikan file adalah gambar yang valid.")
    else:
        # Preprocessing gambar: resize ke 224x224 (sesuai input model)
        image_resized = image.resize((224, 224))
        img_array = np.array(image_resized)

        # Jika gambar grayscale, ubah menjadi RGB
        if img_array.ndim == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        # Jika gambar memiliki channel 4 (misalnya, RGBA), ambil 3 channel pertama saja
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]

        # Terapkan preprocessing dari EfficientNetV2-S
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Bentuk menjadi (1,224,224,3)

        # Lakukan prediksi
        preds = model.predict(img_array)

        # Prediksi Top-1 (kelas dengan score tertinggi)
        top_pred_index = np.argmax(preds, axis=1)[0]
        top_pred_score = float(np.max(preds, axis=1)[0])
        top_pred_class = class_names[top_pred_index]

        # Prediksi Top-3
        sorted_indices = np.argsort(preds[0])[::-1][:3]
        top_classes = [class_names[i] for i in sorted_indices]
        top_scores = [preds[0][i] for i in sorted_indices]

        st.write("### Hasil Prediksi (Top-1):")
        st.write(f"**{top_pred_class}** dengan confidence score: **{top_pred_score:.2f}**")

        st.write("### Top-3 Prediksi:")
        for i in range(3):
            st.write(f"{i+1}. {top_classes[i]}: {top_scores[i]:.2f}")

        # Fitur Feedback: Koreksi hasil prediksi oleh pengguna
        st.write("### Koreksi Prediksi (Feedback)")
        feedback = st.selectbox("Jika prediksi salah, pilih label yang benar:", class_names)
        if st.button("Kirim Feedback"):
            # Anda dapat menambahkan mekanisme logging (misalnya menyimpan ke file atau database) di sini.
            st.success(f"Feedback diterima: gambar ini seharusnya '{feedback}'. Terima kasih!")
