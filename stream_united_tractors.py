import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Perbandingan Kinerja FOA dan PSO dalam Optimasi SVR untuk Peramalan Harga Saham United Tractors")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset (Excel)", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.write("Dataset yang diupload:")
    st.write(data.head())

    # Preprocessing data
    data = data.dropna()
    data = data.sort_values(by='Date')
    X = data[['lag1']].values
    y = data['y'].values

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False, random_state=42)

    # Pilih algoritma
    algorithm = st.selectbox("Pilih algoritma optimasi:", ["FOA", "PSO"])

    if algorithm == "FOA":
        st.header("Fruit Fly Optimization (FOA)")
        model = joblib.load('foa_model.sav')
        y_pred_test = model.predict(X_test)
        mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
        st.write(f"MAPE pada data testing: {mape_test:.4f}%")

        # Plot hasil prediksi
        fig, ax = plt.subplots()
        ax.plot(y_test, label='Aktual', marker='o')
        ax.plot(y_pred_test, label='Prediksi', marker='x')
        ax.set_title('Perbandingan Nilai Aktual dan Prediksi (FOA)')
        ax.set_xlabel('Data Point')
        ax.set_ylabel('Harga Saham')
        ax.legend()
        st.pyplot(fig)

    elif algorithm == "PSO":
        st.header("Particle Swarm Optimization (PSO)")
        model = joblib.load('pso_model.sav')
        y_pred_test = model.predict(X_test)
        mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
        st.write(f"MAPE pada data testing: {mape_test:.4f}%")

        # Plot hasil prediksi
        fig, ax = plt.subplots()
        ax.plot(y_test, label='Aktual', marker='o')
        ax.plot(y_pred_test, label='Prediksi', marker='x')
        ax.set_title('Perbandingan Nilai Aktual dan Prediksi (PSO)')
        ax.set_xlabel('Data Point')
        ax.set_ylabel('Harga Saham')
        ax.legend()
        st.pyplot(fig)
        