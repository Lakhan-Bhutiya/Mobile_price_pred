# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature names
model = joblib.load("price_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("📱 Smartphone Price Predictor")

st.write("Enter the smartphone specifications below to predict its price:")

# User inputs (exclude Sale)
weight = st.number_input("Weight (grams)", min_value=80.0, max_value=250.0, value=120.0, step=0.1)
resolution = st.number_input("Screen Size (inches)", min_value=3.5, max_value=7.5, value=5.0, step=0.1)
ppi = st.number_input("PPI (pixels per inch)", min_value=100, max_value=600, value=300)
cpu_core = st.number_input("CPU Cores", min_value=1, max_value=16, value=4)
cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.5, max_value=4.0, value=1.5, step=0.1)
internal_mem = st.number_input("Internal Memory (GB)", min_value=2, max_value=512, value=16)
ram = st.number_input("RAM (GB)", min_value=0.5, max_value=32.0, value=2.0, step=0.5)
rear_cam = st.number_input("Rear Camera (MP)", min_value=0, max_value=200, value=13)
front_cam = st.number_input("Front Camera (MP)", min_value=0, max_value=100, value=5)
battery = st.number_input("Battery (mAh)", min_value=800, max_value=10000, value=3000)
thickness = st.number_input("Thickness (mm)", min_value=4.0, max_value=15.0, value=8.0, step=0.1)

# Build input dataframe
input_data = pd.DataFrame([[
    weight, resolution, ppi, cpu_core, cpu_freq,
    internal_mem, ram, rear_cam, front_cam, battery, thickness
]], columns=feature_names)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Price: ₹{prediction:.2f}")
