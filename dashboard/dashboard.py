import streamlit as plt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import tensorflow as tf


# LOAD MODEL TENSORFLOW (AI)
@st.cache_resource # Menggunakan cache agar model tidak di-load berulang kali setiap web di-refresh
def load_ai_model():
    # Pastikan nama file sesuai dengan file .keras milik tim Anda
    return tf.keras.models.load_model("mindbalance_model_new.keras")

try:
    model = load_ai_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Gagal memuat model AI: {e}. Pastikan file 'mindbalance_model_new.keras' ada di folder.")
    model = None
    model_loaded = False


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
    # Pastikan file dataset Anda berada di folder yang sama atau sesuaikan path-nya
    df = pd.read_csv("cleaned_anxiety_data.csv") 
    return df

try:
    df = load_data()
except:
    st.warning("Dataset 'cleaned_anxiety_data.csv' tidak ditemukan. Menu EDA interaktif akan menggunakan placeholder data.")
    # Dummy data jika file csv belum di-upload ke server streamlit
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

    # Filter Interaktif berdasarkan Tingkat Kecemasan
    selected_anxiety = st.multiselect(
        "Filter berdasarkan Tingkat Kecemasan:",
        options=df['Anxiety_Category'].unique(),
        default=df['Anxiety_Category'].unique()
    )
    filtered_df = df[df['Anxiety_Category'].isin(selected_anxiety)]

    # Layout Kolom untuk Grafik
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
        sns.boxplot(data=filtered_df, x='Anxiety_Category', y='Sleep_Duration', palette='Pastel1', ax=ax)
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

    # Membuka Form di Streamlit
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        # --- KOLOM 1: DEFINISI VARIABEL INPUT DEMOGRAFI & GAYA HIDUP ---
        with col1:
            st.markdown("### 👤 Profil Demografi & Kebiasaan")
            age = st.number_input("Umur Anda:", min_value=18, max_value=100, value=25)
            gender = st.selectbox("Jenis Kelamin:", ["Male", "Female", "Other"])
            occupation = st.selectbox("Pekerjaan:", ['Artist', 'Athlete', 'Chef', 'Doctor', 'Engineer', 'Freelancer', 'Lawyer', 'Musician', 'Nurse', 'Other', 'Scientist', 'Student', 'Teacher'])
            sleep_hours = st.slider("Durasi Tidur Harian (Jam):", 2.3, 11.3, 7.0, 0.1) # <--- Variabel sleep_hours didefinisikan di sini
            physical_activity = st.slider("Aktivitas Fisik (Jam/Minggu):", 0.0, 10.1, 3.0, 0.1)
            caffeine = st.number_input("Konsumsi Kafein Harian (mg):", min_value=0, max_value=599, value=150)
            alcohol = st.number_input("Konsumsi Alkohol (Gelas/Minggu):", min_value=0, max_value=19, value=2)
            smoking = st.selectbox("Apakah Anda Merokok?", ["Yes", "No"])
            diet_quality = st.slider("Kualitas Pola Makan (Skala 1-10):", 1, 10, 7)

        # --- KOLOM 2: DEFINISI VARIABEL INPUT PARAMETER FISIK & MEDIS ---
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

        # Tombol submit untuk memicu proses pembacaan data dan prediksi
        submit_button = st.form_submit_button(label="Mulai Deteksi AI")

    # Jalankan perhitungan HANYA ketika tombol ditekan dan seluruh variabel di atas sudah terisi
    if submit_button:
        st.markdown("---")
        st.subheader("📋 Hasil Analisis")

        # 1. PROSES FEATURE ENGINEERING (Sekarang aman karena semua variabel input sudah terdefinisi di atas)
        sleep_norm = (sleep_hours - 2.3) / (11.3 - 2.3)
        caffeine_norm = caffeine / 599.0
        sleep_efficiency = round((sleep_norm * 0.7 + (1 - caffeine_norm) * 0.3), 4)

        binary_map = {"Yes": 1, "No": 0}
        stress_norm = (stress_level - 1) / 9
        alcohol_norm = alcohol / 19.0
        activity_norm = physical_activity / 10.1
        diet_norm = (diet_quality - 1) / 9

        lifestyle_risk = round((
            stress_norm * 0.25 + binary_map[smoking] * 0.10 + alcohol_norm * 0.10 +
            binary_map[family_history] * 0.15 + binary_map[recent_life_event] * 0.15 +
            (1 - activity_norm) * 0.10 + (1 - diet_norm) * 0.15
        ), 4)

        hr_norm = (heart_rate - 60) / (119 - 60)
        br_norm = (breathing_rate - 12) / (29 - 12)
        sweat_norm = (sweating_level - 1) / 4

        anxiety_composite = round((
            stress_norm * 0.30 + hr_norm * 0.20 + br_norm * 0.20 +
            sweat_norm * 0.15 + binary_map[family_history] * 0.15
        ), 4)

        # 2. PROSES PREDIKSI MODEL AI TENSORFLOW
        if model_loaded:
            gender_map = {"Male": 0, "Female": 1, "Other": 2}
            occ_list = ['Artist', 'Athlete', 'Chef', 'Doctor', 'Engineer', 'Freelancer', 'Lawyer', 'Musician', 'Nurse', 'Other', 'Scientist', 'Student', 'Teacher']
            occ_map = {occ: i for i, occ in enumerate(occ_list)}

            # Menggabungkan 21 komponen variabel menjadi matriks siap prediksi
            input_vector = np.array([[
                age, gender_map[gender], occ_map[occupation],
                sleep_hours, physical_activity, caffeine,
                alcohol, binary_map[smoking], binary_map[family_history],
                stress_level, heart_rate, breathing_rate,
                sweating_level, binary_map[dizziness], binary_map[medication],
                therapy_sessions, binary_map[recent_life_event], diet_quality,
                sleep_efficiency, lifestyle_risk, anxiety_composite
            ]], dtype=np.float32)

            with st.spinner("Model AI sedang memproses matriks klinis Anda..."):
                class_out, reg_out = model(tf.constant(input_vector), training=False)
                predicted_class_idx = int(np.argmax(class_out.numpy()[0]))
                
            labels = ["Low (Rendah)", "Medium (Sedang)", "High (Tinggi)"]
            anxiety_result = labels[predicted_class_idx]
        else:
            # Mode Cadangan (Simulasi Cerdas) jika file .keras tidak ada di folder
            if anxiety_composite > 0.5 or stress_level >= 8:
                predicted_class_idx = 2
                anxiety_result = "High (Tinggi)"
            elif anxiety_composite > 0.3 or stress_level >= 5:
                predicted_class_idx = 1
                anxiety_result = "Medium (Sedang)"
            else:
                predicted_class_idx = 0
                anxiety_result = "Low (Rendah)"

        # 3. MENAMPILKAN HASILNYA KE LAYAR DASHBOARD
        if predicted_class_idx == 2:
            st.error(f"**Hasil Analisis Tingkat Kecemasan: {anxiety_result}**")
            st.info("**Rekomendasi Coping:** Ambil waktu jeda istirahat, batasi asupan kopi/kafein harian, lakukan teknik *box breathing*, dan sangat disarankan untuk berdiskusi dengan psikolog atau tenaga profesional.")
        elif predicted_class_idx == 1:
            st.warning(f"**Hasil Analisis Tingkat Kecemasan: {anxiety_result}**")
            st.info("**Rekomendasi Coping:** Tingkatkan durasi tidur harian, luangkan waktu 15 menit untuk jalan santai/olahraga, serta kurangi stimulan di malam hari.")
        else:
            st.success(f"**Hasil Analisis Tingkat Kecemasan: {anxiety_result}**")
            st.info("**Rekomendasi Coping:** Tingkat kecemasan Anda sangat baik dan stabil. Pertahankan kombinasi pola hidup dan manajemen stres yang sudah Anda miliki saat ini.")

        # Tampilkan Nilai Skor Tambahan di Dashboard hasil kalkulasi Feature Engineering
        st.markdown("### Skor Indikator Gabungan (Feature Engineering):")
        m1, m2, m3 = st.columns(3)
        m1.metric("Sleep Efficiency Score", f"{sleep_efficiency:.4f}")
        m2.metric("Lifestyle Risk Index", f"{lifestyle_risk:.4f}")
        m3.metric("Anxiety Composite Score", f"{anxiety_composite:.4f}")