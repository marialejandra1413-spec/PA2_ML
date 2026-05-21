# ============================================================
# app.py — Aplicación web Titanic con Streamlit
# ============================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Configuración de la página ────────────────────────────
st.set_page_config(page_title="Predictor Titanic", page_icon="🚢", layout="centered")

# ── Encabezado ────────────────────────────────────────────
st.title("🚢 Predictor de Supervivencia del Titanic")
st.markdown("---")

# ── Datos del estudiante ──────────────────────────────────
st.markdown(
    """
**👤 Nombre:** TU NOMBRE COMPLETO  
**🎓 Código ISIL:** TU_CODIGO_ISIL  
**📓 Cuaderno Colab:** [Ver cuaderno aquí](https://colab.research.google.com/drive/1jjCVQLmSF-AZA1UPCBAVa1coJ0UGTZuY)
"""
)
st.markdown("---")


# ── Cargar el modelo ──────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelos/modelo_random_forest.pkl")
    columnas = joblib.load("modelos/columnas.pkl")
    return modelo, columnas


try:
    modelo, columnas = cargar_modelo()
    st.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()

# ── Formulario de entrada ─────────────────────────────────
st.subheader("📋 Ingresa los datos del pasajero")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Clase del boleto (Pclass)", [1, 2, 3], help="1=Primera, 2=Segunda, 3=Tercera"
    )
    sex = st.selectbox("Sexo", ["male", "female"])
    age = st.slider("Edad", min_value=1, max_value=80, value=30)
    sibsp = st.number_input(
        "Hermanos/Cónyuge a bordo (SibSp)", min_value=0, max_value=8, value=0
    )

with col2:
    parch = st.number_input(
        "Padres/Hijos a bordo (Parch)", min_value=0, max_value=6, value=0
    )
    fare = st.number_input(
        "Tarifa del boleto (Fare)", min_value=0.0, max_value=600.0, value=32.0
    )
    embarked = st.selectbox(
        "Puerto de embarque (Embarked)",
        ["S", "C", "Q"],
        help="S=Southampton, C=Cherbourg, Q=Queenstown",
    )


# ── Preparar los datos ────────────────────────────────────
def preparar_datos(pclass, sex, age, sibsp, parch, fare, embarked):
    datos = {
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Sex_male": 1 if sex == "male" else 0,
        "Embarked_Q": 1 if embarked == "Q" else 0,
        "Embarked_S": 1 if embarked == "S" else 0,
    }
    df_pred = pd.DataFrame([datos])
    # Asegurar que tenga todas las columnas del modelo
    for col in columnas:
        if col not in df_pred.columns:
            df_pred[col] = 0
    return df_pred[columnas]


# ── Botón de predicción ───────────────────────────────────
st.markdown("---")
if st.button("🔮 Predecir supervivencia", type="primary", use_container_width=True):

    datos_entrada = preparar_datos(pclass, sex, age, sibsp, parch, fare, embarked)
    prediccion = modelo.predict(datos_entrada)[0]
    probabilidad = modelo.predict_proba(datos_entrada)[0]

    st.markdown("---")
    st.subheader("📊 Resultado de la predicción")

    if prediccion == 1:
        st.success("## 🟢 ¡El pasajero SOBREVIVIÓ!")
        prob = probabilidad[1] * 100
        st.metric("Probabilidad de supervivencia", f"{prob:.1f}%")
    else:
        st.error("## 🔴 El pasajero NO sobrevivió")
        prob = probabilidad[0] * 100
        st.metric("Probabilidad de no supervivencia", f"{prob:.1f}%")

    # Barra de progreso
    st.progress(float(probabilidad[1]))
    st.caption(
        f"Probabilidad de sobrevivir: {probabilidad[1]*100:.1f}% | "
        f"No sobrevivir: {probabilidad[0]*100:.1f}%"
    )

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.caption("Evaluación PA2 — Curso de Machine Learning | ISIL")
