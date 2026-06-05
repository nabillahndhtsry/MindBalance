import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf

try:
    import keras
    from keras.layers import Dense as KerasDense
    from keras.utils import register_keras_serializable
except ImportError:
    keras = None
    KerasDense = None
    register_keras_serializable = None

# Kustom Dense untuk menerima quantization_config saat model lama diserialisasi
custom_objects = {}
if KerasDense is not None and register_keras_serializable is not None:
    @register_keras_serializable(package='keras.layers', name='Dense')
    class DenseWithQuantization(KerasDense):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.quantization_config = quantization_config

        def get_config(self):
            config = super().get_config()
            config.pop('quantization_config', None)
            return config

    custom_objects['Dense'] = DenseWithQuantization

# Mengambil path folder tempat script ini dijalankan agar aman di server Linux
st.set_page_config(page_title="MindBalance - Anxiety Detection Dashboard", page_icon="🧠", layout="wide")
current_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(current_dir, "mindbalance_model_new.keras")
data_path = os.path.join(current_dir, "cleaned_anxiety_data.csv")

# LOAD MODEL DENGAN ERROR HANDLING
@st.cache_resource
def load_ai_model():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {model_path}")
    loader = keras.models if keras is not None else tf.keras.models
    return loader.load_model(model_path, compile=False, custom_objects=custom_objects)

# Menginisialisasi model
model = None
model_error = None
try:
    model = load_ai_model()
except Exception as e:
    model_error = e
    st.warning(
        "⚠️ Model tidak dapat dimuat. UI inference akan berjalan dengan fallback aturan."
    )
    st.caption(f"Detil error: {type(e).__name__}: {e}")

st.title("🧠 MindBalance — AI-Powered Anxiety Detection & Mental Wellness")
st.markdown("---")

# helper encoding untuk inference
GENDER_CATEGORIES = ["Male", "Female", "Other"]
OCCUPATION_CATEGORIES = [
    'Artist', 'Athlete', 'Chef', 'Doctor', 'Engineer', 'Freelancer',
    'Lawyer', 'Musician', 'Nurse', 'Other', 'Scientist', 'Student', 'Teacher'
]


def encode_gender(gender: str) -> list:
    return [1 if gender == cat else 0 for cat in GENDER_CATEGORIES]


def encode_yes_no(value: str) -> int:
    return 1 if value == "Yes" else 0


def encode_occupation(occupation: str) -> list:
    return [1 if occupation == cat else 0 for cat in OCCUPATION_CATEGORIES]


def build_input_vector(
    age: int,
    gender: str,
    occupation: str,
    sleep_hours: float,
    physical_activity: float,
    caffeine: int,
    alcohol: int,
    smoking: str,
    stress_level: int,
    heart_rate: int,
    breathing_rate: int,
    sweating_level: int,
    dizziness: str,
    medication: str,
    therapy_sessions: int,
    family_history: str,
    recent_life_event: str,
    diet_quality: int,
) -> np.ndarray:
    vector = [
        age,
        sleep_hours,
        physical_activity,
        caffeine,
        alcohol,
        stress_level,
        heart_rate,
        breathing_rate,
        sweating_level,
        therapy_sessions,
        diet_quality,
        encode_yes_no(smoking),
        encode_yes_no(dizziness),
        encode_yes_no(medication),
        encode_yes_no(family_history),
        encode_yes_no(recent_life_event),
    ]
    vector.extend(encode_gender(gender))
    vector.extend(encode_occupation(occupation))
    return np.array(vector, dtype=np.float32)


def calculate_risk_metrics(
    sleep_hours: float,
    physical_activity: float,
    caffeine: int,
    alcohol: int,
    stress_level: int,
) -> tuple[float, float, float]:
    sleep_efficiency = max(0.0, min(1.0, (sleep_hours - 4) / 6))
    lifestyle_risk = min(1.0, (0.2 * max(0, 7 - sleep_hours) + 0.1 * max(0, 400 - physical_activity * 40) + 0.001 * caffeine + 0.02 * alcohol + 0.08 * stress_level) / 10)
    anxiety_composite = min(1.0, (0.35 * (1.0 - sleep_efficiency) + 0.25 * stress_level / 10 + 0.15 * caffeine / 600 + 0.15 * alcohol / 20 + 0.1 * (1.0 - min(1.0, physical_activity / 10))))
    return sleep_efficiency, lifestyle_risk, anxiety_composite


# 2. LOAD DATASET (Untuk Keperluan EDA)
@st.cache_data
def load_data():
    return pd.read_csv(data_path)

try:
    df = load_data()
except Exception:
    st.warning("Dataset 'cleaned_anxiety_data.csv' tidak ditemukan. Menu EDA interaktif akan menggunakan placeholder data.")
    df = pd.DataFrame({
        'Anxiety_Category': ['Low', 'Medium', 'High', 'Low', 'High', 'Medium'],
        'Sleep Hours': [7.5, 6.0, 5.0, 8.0, 4.5, 6.5],
        'Caffeine Intake': [100, 250, 400, 50, 450, 200],
        'Stress Level': [3, 6, 9, 2, 8, 5],
        'Heart Rate': [70, 82, 95, 68, 90, 78]
    })


# 3. NAVIGASI MENU (SIDEBAR)
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["📊 Interactive EDA", "🔮 Anxiety Detection (Inference UI)"],
    key="main_menu"
)


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
        if 'Sleep Hours' in filtered_df.columns:
            sns.boxplot(data=filtered_df, x='Anxiety_Category', y='Sleep Hours', palette='Pastel1', ax=ax)
            plt.xlabel("Tingkat Kecemasan")
            plt.ylabel("Durasi Tidur (Jam)")
            st.pyplot(fig)
        else:
            ax.text(0.5, 0.5, 'Kolom Sleep Hours tidak tersedia', ha='center', va='center')
            st.pyplot(fig)
        st.caption("Terlihat kecenderungan bahwa durasi tidur yang lebih rendah berkorelasi dengan tingkat kecemasan yang lebih tinggi.")

    st.markdown("---")
    st.subheader("3. Matriks Korelasi Fitur Numerik")
    corr = df.corr(numeric_only=True)
    if corr.empty:
        st.warning("Tidak ada fitur numerik yang tersedia untuk menghitung korelasi.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(corr, annot=True, cmap='Coolwarm', fmt='.2f', ax=ax)
        fig.tight_layout()
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
            gender = st.selectbox("Jenis Kelamin:", ["Male", "Female", "Other"], key="gender")
            occupation = st.selectbox("Pekerjaan:", OCCUPATION_CATEGORIES, key="occupation")
            sleep_hours = st.slider("Durasi Tidur Harian (Jam):", 2.3, 11.3, 7.0, 0.1)
            physical_activity = st.slider("Aktivitas Fisik (Jam/Minggu):", 0.0, 10.1, 3.0, 0.1)
            caffeine = st.number_input("Konsumsi Kafein Harian (mg):", min_value=0, max_value=599, value=150)
            alcohol = st.number_input("Konsumsi Alkohol (Gelas/Minggu):", min_value=0, max_value=19, value=2)
            smoking = st.selectbox("Apakah Anda Merokok?", ["Yes", "No"], key="smoking")
            diet_quality = st.slider("Kualitas Pola Makan (Skala 1-10):", 1, 10, 7)

        with col2:
            st.markdown("### 🫀 Parameter Fisik & Medis")
            stress_level = st.slider("Tingkat Stres Psikologis (Skala 1-10):", 1, 10, 5)
            heart_rate = st.number_input("Detak Jantung Istirahat (BPM):", min_value=60, max_value=119, value=75)
            breathing_rate = st.number_input("Frekuensi Napas (Napas/Menit):", min_value=12, max_value=29, value=18)
            sweating_level = st.slider("Tingkat Keringat Berlebih (Skala 1-5):", 1, 5, 2)
            dizziness = st.selectbox("Apakah Sering Merasa Pusing?", ["Yes", "No"], key="dizziness")
            medication = st.selectbox("Sedang Konsumsi Obat Kecemasan?", ["Yes", "No"], key="medication")
            therapy_sessions = st.number_input("Sesi Terapi / Konseling Bulan Ini:", min_value=0, max_value=12, value=0)
            family_history = st.selectbox("Ada Riwayat Kecemasan di Keluarga?", ["Yes", "No"], key="family_history")
            recent_life_event = st.selectbox("Ada Kejadian Besar/Trauma 6 Bulan Terakhir?", ["Yes", "No"], key="recent_life_event")

        submit_button = st.form_submit_button(label="Mulai Deteksi AI")

    if submit_button:
        st.markdown("---")
        st.subheader("📋 Hasil Analisis")

        input_vector = build_input_vector(
            age,
            gender,
            occupation,
            sleep_hours,
            physical_activity,
            caffeine,
            alcohol,
            smoking,
            stress_level,
            heart_rate,
            breathing_rate,
            sweating_level,
            dizziness,
            medication,
            therapy_sessions,
            family_history,
            recent_life_event,
            diet_quality,
        )

        input_array = np.expand_dims(input_vector, axis=0)
        predicted_class_idx = None
        anxiety_result = None
        model_predicted = False

        if model is not None:
            expected_shape = None
            try:
                expected_shape = model.input_shape[1] if hasattr(model, 'input_shape') else None
            except Exception:
                expected_shape = None

            if expected_shape == input_array.shape[1]:
                try:
                    prediction = model.predict(input_array, verbose=0)
                    model_predicted = True
                    if prediction.ndim == 2 and prediction.shape[1] > 1:
                        predicted_class_idx = int(np.argmax(prediction[0]))
                    else:
                        predicted_class_idx = int(np.round(prediction[0][0])) if prediction.ndim == 2 else int(np.round(prediction[0]))
                except Exception as e:
                    st.warning(f"Model terpasang tetapi prediksi gagal: {e}")
            else:
                st.warning(f"Model ditemukan, tetapi bentuk input berbeda (diharapkan {expected_shape}, diteruskan {input_array.shape[1]}). Menggunakan fallback aturan.")

        sleep_efficiency, lifestyle_risk, anxiety_composite = calculate_risk_metrics(
            sleep_hours, physical_activity, caffeine, alcohol, stress_level
        )

        if not model_predicted:
            if anxiety_composite > 0.5 or stress_level >= 8:
                predicted_class_idx = 2
                anxiety_result = "High (Tinggi)"
            elif anxiety_composite > 0.3 or stress_level >= 5:
                predicted_class_idx = 1
                anxiety_result = "Medium (Sedang)"
            else:
                predicted_class_idx = 0
                anxiety_result = "Low (Rendah)"
        else:
            labels = ["Low (Rendah)", "Medium (Sedang)", "High (Tinggi)"]
            anxiety_result = labels[predicted_class_idx] if 0 <= predicted_class_idx < len(labels) else "Unknown"

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