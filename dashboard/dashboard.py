import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import tensorflow as tf
import os

# Mengambil path folder tempat script ini dijalankan agar aman di server Linux
st.set_page_config(page_title="MindBalance - Anxiety Detection", page_icon="🧠", layout="wide")
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "mindbalance_model_new.keras")

# LOAD MODEL DENGAN ERROR HANDLING
@st.cache_resource
def load_ai_model():
    # compile=False mencegah Keras mencoba merekonstruksi optimizer/loss 
    # yang seringkali memicu error modul di environment cloud
    return tf.keras.models.load_model(model_path, compile=False)

# Menginisialisasi model
model = None
try:
    model = load_ai_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")

st.title("🧠 MindBalance — AI-Powered Anxiety Detection")
st.markdown("---")

# Simulasi Input (Ganti dengan input user yang sebenarnya)
# Pastikan bentuk input (shape) sesuai dengan input_layer modelmu [None, 21]
input_dummy = np.random.rand(1, 21).astype(np.float32)

if model:
    if st.button("Jalankan Prediksi"):
        try:
            # Prediksi
            prediction = model.predict(input_dummy)
            st.success("Prediksi berhasil dijalankan!")
            st.write(f"Output mentah: {prediction}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
else:
    st.warning("Model belum dimuat, silakan periksa file .keras di folder.")

# 1. KONFIGURASI HALAMAN & THEME
st.set_page_config(
    page_title="MindBalance - Anxiety Detection Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MindBalance — AI-Powered Anxiety Detection & Mental Wellness")
st.markdown("---")


# 2. LOAD DATASET (Untuk Keperluan EDA)
@st.cache_data
def load_data():
    csv_path = os.path.join(current_dir, "cleaned_anxiety_data.csv")
    df = pd.read_csv(csv_path) 
    return df

try:
    df = load_data()
except:
    st.warning("Dataset 'cleaned_anxiety_data.csv' tidak ditemukan. Menu EDA interaktif akan menggunakan placeholder data.")
    df = pd.DataFrame({
        'Anxiety_Category': ['Low', 'Medium', 'High', 'Low', 'High', 'Medium'],
        'Sleep_Duration': [7.5, 6.0, 5.0, 8.0, 4.5, 6.5],
        'Caffeine_Intake': [100, 250, 400, 50, 450, 200],
        'Stress_Level': [3, 6, 9, 2, 8, 5],
        'Heart_Rate': [70, 82, 95, 68, 90, 78]
    })


# 3. NAVIGASI MENU (SIDEBAR)
menu = st.sidebar.selectbox("Pilih Menu:", ["📊 Interactive EDA", "🔮 Anxiety Detection (Inference UI)"])


# MENU 1: INTERACTIVE EDA
if menu == "📊 Interactive EDA":
    st.header("Exploratory Data Analysis (EDA) Interaktif")
    st.write("Analisis hubungan antara gaya hidup, kondisi fisik, dan tingkat kecemasan pengguna.")

    selected_anxiety = st.multiselect(
        "Filter berdasarkan Tingkat Kecemasan:",
        options=df['Anxiety_Category'].unique(),
        default=df['Anxiety_Category'].unique()
    )
    filtered_df = df[df['Anxiety_Category'].isin(selected_anxiety)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Distribusi Tingkat Kecemasan")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=filtered_df, x='Anxiety_Category', palette='Set2', ax=ax)
        plt.xlabel("Tingkat Kecemasan")
        plt.ylabel("Jumlah Pengguna")
        st.pyplot(fig)
        st.caption("Grafik ini menunjukkan proporsi data pengguna berdasarkan tingkat kecemasan yang dipilih.")

    with col2:
        st.subheader("2. Hubungan Durasi Tidur & Kecemasan")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=filtered_df, x='Anxiety_Category', y='Sleep Hours', palette='Pastel1', ax=ax)
        plt.xlabel("Tingkat Kecemasan")
        plt.ylabel("Durasi Tidur (Jam)")
        st.pyplot(fig)
        st.caption("Terlihat kecenderungan bahwa durasi tidur yang lebih rendah berkorelasi dengan tingkat kecemasan yang lebih tinggi.")

    st.markdown("---")
    st.subheader("3. Matriks Korelasi Fitur Numerik")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)


# MENU 2: INFERENCE UI (FORM PREDIKSI)
elif menu == "🔮 Anxiety Detection (Inference UI)":
    st.header("Form Deteksi Dini Tingkat Kecemasan")
    st.write("Masukkan indikator profil klinis dan kebiasaan harian Anda untuk dianalisis oleh AI.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Profil Demografi & Kebiasaan")
            age = st.number_input("Umur Anda:", min_value=18, max_value=100, value=25)
            gender = st.selectbox("Jenis Kelamin:", ["Male", "Female", "Other"])
            occupation = st.selectbox("Pekerjaan:", ['Artist', 'Athlete', 'Chef', 'Doctor', 'Engineer', 'Freelancer', 'Lawyer', 'Musician', 'Nurse', 'Other', 'Scientist', 'Student', 'Teacher'])
            sleep_hours = st.slider("Durasi Tidur Harian (Jam):", 2.3, 11.3, 7.0, 0.1)
            physical_activity = st.slider("Aktivitas Fisik (Jam/Minggu):", 0.0, 10.1, 3.0, 0.1)
            caffeine = st.number_input("Konsumsi Kafein Harian (mg):", min_value=0, max_value=599, value=150)
            alcohol = st.number_input("Konsumsi Alkohol (Gelas/Minggu):", min_value=0, max_value=19, value=2)
            smoking = st.selectbox("Apakah Anda Merokok?", ["Yes", "No"])
            diet_quality = st.slider("Kualitas Pola Makan (Skala 1-10):", 1, 10, 7)

        with col2:
            st.markdown("### 🫀 Parameter Fisik & Medis")
            stress_level = st.slider("Tingkat Stres Psikologis (Skala 1-10):", 1, 10, 5)
            heart_rate = st.number_input("Detak Jantung Istirahat (BPM):", min_value=60, max_value=119, value=75)
            breathing_rate = st.number_input("Frekuensi Napas (Napas/Menit):", min_value=12, max_value=29, value=18)
            sweating_level = st.slider("Tingkat Keringat Berlebih (Skala 1-5):", 1, 5, 2)
            dizziness = st.selectbox("Apakah Sering Merasa Pusing?", ["Yes", "No"])
            medication = st.selectbox("Sedang Konsumsi Obat Kecemasan?", ["Yes", "No"])
            therapy_sessions = st.number_input("Sesi Terapi / Konseling Bulan Ini:", min_value=0, max_value=12, value=0)
            family_history = st.selectbox("Ada Riwayat Kecemasan di Keluarga?", ["Yes", "No"])
            recent_life_event = st.selectbox("Ada Kejadian Besar/Trauma 6 Bulan Terakhir?", ["Yes", "No"])

        submit_button = st.form_submit_button(label="Mulai Deteksi AI")


    if submit_button: # Pastikan variabel ini sudah didefinisikan di bagian input form-mu
        st.markdown("---")
        st.subheader("📋 Hasil Analisis")

        # [Logika perhitungan fitur tetap sama seperti kodingan aslimu...]
        # (Pastikan variabel sleep_hours, caffeine, dll sudah didefinisikan sebelumnya)

        if model_loaded:
            # [Logika input_vector tetap sama...]
            with st.spinner("MindBalance sedang menganalisis profil kesehatan Anda..."):
                class_out, reg_out = model(np.array([input_vector]), training=False)
                predicted_class_idx = int(np.argmax(class_out.numpy()[0]))
                labels = ["Low (Rendah)", "Medium (Sedang)", "High (Tinggi)"]
                anxiety_result = labels[predicted_class_idx]
        else:
            # Fallback logika rule-based
            if anxiety_composite > 0.5 or stress_level >= 8:
                predicted_class_idx = 2; anxiety_result = "High (Tinggi)"
            elif anxiety_composite > 0.3 or stress_level >= 5:
                predicted_class_idx = 1; anxiety_result = "Medium (Sedang)"
            else:
                predicted_class_idx = 0; anxiety_result = "Low (Rendah)"

        # Visual Output Awam Friendly
        if predicted_class_idx == 2:
            st.error(f"### 🚨 Tingkat Kecemasan: {anxiety_result}")
            st.progress(90)
            st.write("Hasil analisis menunjukkan tingkat kecemasan yang cukup tinggi. Sangat disarankan untuk mencari dukungan profesional.")
        elif predicted_class_idx == 1:
            st.warning(f"### ⚠️ Tingkat Kecemasan: {anxiety_result}")
            st.progress(50)
            st.write("Ada indikasi tekanan stres. Cobalah melakukan teknik relaksasi rutin.")
        else:
            st.success(f"### ✅ Tingkat Kecemasan: {anxiety_result}")
            st.progress(10)
            st.write("Kondisi mental Anda saat ini terlihat tenang dan stabil.")

        st.markdown("### 📊 Analisis Gaya Hidup")
        m1, m2, m3 = st.columns(3)
        m1.metric("Kualitas Tidur", f"{sleep_efficiency * 100:.0f}%")
        m2.metric("Risiko Gaya Hidup", f"{lifestyle_risk * 100:.0f}%")
        m3.metric("Skor Komposit", f"{anxiety_composite * 100:.0f}%")