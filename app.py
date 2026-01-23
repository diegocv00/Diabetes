import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Predicción de riesgo de diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_resources():
    model = joblib.load('modelo_regresion_logistica_diabetes.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_resources()
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos del modelo ('modelo_regresion_logistica_diabetes.pkl') o el escalador ('scaler.pkl'). Asegúrate de haber ejecutado el script de entrenamiento.")
    st.stop()

# Header
st.title("🩺 Predicción de riesgo de diabetes")
st.markdown("### Ingrese los datos clínicos del paciente para evaluar el riesgo.")

# Create columns for layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Estilo de vida y antecedentes")
    st.markdown("<hr style='height:1px;border:none;color:#e0e0e0;background-color:#e0e0e0;margin-top: 5px; margin-bottom: 15px;' />", unsafe_allow_html=True)
    alcohol_consumption = st.slider("Consumo de alcohol (veces/semana)", 0, 20, 0, help="Frecuencia de consumo de alcohol semanal")
    physical_activity = st.number_input("Actividad física (min/semana)", min_value=0, max_value=1000, value=30, step=10)
    family_history = st.selectbox("Antecedentes familiares de diabetes", ["No", "Sí"])
    bmi = st.number_input("IMC (Índice de Masa Corporal)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    st.markdown("#### Perfil lipídico")
    st.markdown("<hr style='height:1px;border:none;color:#e0e0e0;background-color:#e0e0e0;margin-top: 5px; margin-bottom: 15px;' />", unsafe_allow_html=True)
    cholesterol_total = st.number_input("Colesterol Total (mg/dL)", min_value=100, max_value=400, value=200)
    hdl_cholesterol = st.number_input("Colesterol HDL (mg/dL)", min_value=20, max_value=100, value=50)
    ldl_cholesterol = st.number_input("Colesterol LDL (mg/dL)", min_value=50, max_value=300, value=100)
    triglycerides = st.number_input("Triglicéridos (mg/dL)", min_value=10, max_value=600, value=150)

with col3:
    st.markdown("#### Glucosa e insulina")
    st.markdown("<hr style='height:1px;border:none;color:#e0e0e0;background-color:#e0e0e0;margin-top: 5px; margin-bottom: 15px;' />", unsafe_allow_html=True)
    glucose_fasting = st.number_input("Glucosa en ayunas (mg/dL)", min_value=50, max_value=400, value=90)
    glucose_postprandial = st.number_input("Glucosa postprandial (mg/dL)", min_value=50, max_value=500, value=120)
    insulin_level = st.number_input("Nivel de insulina (µU/mL)", min_value=0, max_value=100, value=10)
    hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=5.5, step=0.1)

# Convert categorical input
family_history_val = 1 if family_history == "Sí" else 0

# Prepare input array for prediction
# Order: alcohol_consumption_per_week, physical_activity_minutes_per_week, family_history_diabetes, bmi, 
# cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides, glucose_fasting, glucose_postprandial, 
# insulin_level, hba1c
input_features = np.array([[
    alcohol_consumption,
    physical_activity,
    family_history_val,
    bmi,
    cholesterol_total,
    hdl_cholesterol,
    ldl_cholesterol,
    triglycerides,
    glucose_fasting,
    glucose_postprandial,
    insulin_level,
    hba1c
]])

st.write("---")

# Prediction button
if st.button("Calcular riesgo"):
    try:
        # Scale input
        input_scaled = scaler.transform(input_features)
        
        # Predict probability
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Threshold (using 0.6 as defined in the training script)
        threshold = 0.6
        prediction = 1 if probability >= threshold else 0
        
        st.write("### Resultados")
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.error(f"**ALTO RIESGO DETECTADO**")
                st.markdown(f"El modelo predice una alta probabilidad de diabetes.")
            else:
                st.success(f"**BAJO RIESGO**")
                st.markdown(f"El modelo no detecta diabetes con los datos proporcionados.")
        
        with result_col2:
            st.metric(label="Probabilidad estimada", value=f"{probability:.2%}")
            st.progress(probability)
            


    except Exception as e:
        st.error(f"Ocurrió un error al procesar los datos: {e}")
