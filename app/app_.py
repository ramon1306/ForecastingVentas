"""Streamlit entry point for the sales forecasting application."""

import streamlit as st

st.set_page_config(page_title="Forecasting Ventas", layout="wide")

st.title("Forecasting de Ventas")
st.write("Carga un modelo entrenado para consultar pronosticos de ventas.")

uploaded_model = st.file_uploader("Modelo entrenado (.joblib)", type=["joblib"])

if uploaded_model:
    st.success(f"Modelo recibido: {uploaded_model.name}")
else:
    st.info("Todavia no hay un modelo cargado. Entrena uno desde los notebooks.")
