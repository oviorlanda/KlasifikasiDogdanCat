import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐶🐱", layout="centered")

# --- Judul Utama ---
st.title("🐶🐱 Tebak Hewan Peliharaan Kamu: Kucing atau Anjing")
st.write(
    """
    Selamat datang di aplikasi klasifikasi **Anjing vs Kucing**.  
    Aplikasi ini menggunakan **CNN** yang sudah dilatih sebelumnya untuk memprediksi gambar
    yang Anda upload.  
    Silakan jelajahi halaman di sidebar untuk mempelajari model dan mencoba prediksi!
    """
)

# --- Sidebar Navigasi ---
page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "📈 Model Performance", "🐱🐶 Predict"],
    index=0
)

# --- Cache Model ---
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Oviorlanda/PrediksiDogCat",
        filename="cnn_scratch_best.h5"
    )
    model = tf.keras.models.load_model(model_path)
    return model

# --- Halaman: Home ---
if page == "🏠 Home":
    st.header("Tentang Aplikasi")
    st.markdown(
        """
        - **Tujuan**: Mengklasifikasikan gambar apakah termasuk **Anjing** atau **Kucing**.
        - **Dataset**: [Microsoft Cats vs Dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)
        - **Teknologi**: 
            - TensorFlow (untuk CNN)
            - Hugging Face Hub (untuk hosting model)
            - Streamlit (untuk UI interaktif)
        
        ---
        Gunakan menu di sidebar:
        - 📈 **Model Performance**: Lihat grafik akurasi dan F1 score model.
        - 🐱🐶 **Predict**: Upload gambar dan lihat hasil prediksi.
        """
    )

# --- Halaman: Model Performance ---
elif page == "📈 Model Performance":
    st.header("Performa Model")
    st.write(
        """
        Berikut adalah contoh grafik **akurasi** dan **F1 score** dari model yang sudah dilatih.
        (Contoh data ditampilkan untuk demo.)
        """
    )

    # Simulasi data akurasi dan F1 (contoh; ganti dengan data asli jika ada)
    epochs = list(range(1, 11))
    acc_scratch = [0.70, 0.78, 0.82, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91]
    acc_vgg16 = [0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.92, 0.93, 0.94, 0.94]

    fig, ax = plt.subplots()
    ax.plot(epochs, acc_scratch, label="CNN Scratch", marker="o")
    ax.plot(epochs, acc_vgg16, label="VGG16 Transfer Learning", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Perbandingan Akurasi")
    ax.legend()
    st.pyplot(fig)

    st.info("Grafik di atas menunjukkan CNN Scratch sedikit kalah dengan VGG16 dalam hal akurasi, tapi tetap cukup baik.")

# --- Halaman: Predict ---
elif page == "🐱🐶 Predict":
    st.header("Upload Gambar untuk Prediksi")
    uploaded_file = st.file_uploader("Upload gambar (JPG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

        # Load model hanya di halaman predict (biar cepat halaman lain)
        model = load_model()

        # Preprocessing
        img = image.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        with st.spinner("Sedang memproses gambar..."):
            prediction = model.predict(img_array)
        label = "Anjing 🐶" if prediction[0][0] > 0.5 else "Kucing 🐱"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        st.success(f"Hasil prediksi: **{label}** (Confidence: {confidence:.2%})")
