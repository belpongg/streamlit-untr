import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Judul aplikasi
st.title("Perbandingan Kinerja FOA dan PSO dalam Optimasi SVR untuk Peramalan Harga Saham United Tractors")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset (Excel)", type=["xlsx"])
if uploaded_file is not None:
    # Baca dataset
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
        model = joblib.load('foa_model.sav')  # Memuat model FOA
        y_pred_test = model.predict(X_test)  # Prediksi menggunakan data testing
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
        model = joblib.load('pso_model.sav')  # Memuat model PSO
        y_pred_test = model.predict(X_test)  # Prediksi menggunakan data testing
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

    # Prediksi 15 hari ke depan mulai dari 1 Januari 2025
    st.header("Prediksi 15 Hari ke Depan (Mulai 1 Januari 2025)")

    # Buat data input untuk prediksi 15 hari ke depan
    last_lag1 = X_test[-1][0]  # Ambil nilai terakhir dari X_test
    future_predictions = []
    for i in range(15):
        # Prediksi nilai y untuk periode berikutnya
        next_prediction = model.predict([[last_lag1]])[0]
        future_predictions.append(next_prediction)
        # Update last_lag1 untuk prediksi berikutnya
        last_lag1 = next_prediction

    # Buat DataFrame untuk menyimpan hasil prediksi
    future_dates = pd.date_range(start='2025-01-01', periods=15, freq='D')  # Mulai dari 1 Januari 2025
    forecast_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi_Harga': future_predictions
    })

    # Tampilkan hasil prediksi
    st.write("Hasil Prediksi 15 Hari ke Depan (Mulai 1 Januari 2025):")
    st.write(forecast_df)

    # Plot hasil prediksi
    fig, ax = plt.subplots()
    ax.plot(future_dates, future_predictions, marker='o', linestyle='-', color='blue')
    ax.set_title('Prediksi Harga Saham 15 Hari ke Depan (Mulai 1 Januari 2025)')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Prediksi Harga')
    ax.grid(True)
    st.pyplot(fig)

    # Tabel Perbandingan Nilai Aktual dan Prediksi
    st.header("Perbandingan Nilai Aktual dan Prediksi")

    # Buat DataFrame untuk perbandingan
    comparison_df = pd.DataFrame({
        'Tanggal': data['Date'].iloc[-len(y_test):],  # Ambil tanggal sesuai dengan data testing
        'Aktual': y_test,
        'Prediksi': y_pred_test
    })

    # Tampilkan tabel
    st.write("Tabel Perbandingan Nilai Aktual dan Prediksi:")
    st.write(comparison_df)

    # Plot perbandingan
    fig, ax = plt.subplots()
    ax.plot(comparison_df['Tanggal'], comparison_df['Aktual'], label='Aktual', marker='o')
    ax.plot(comparison_df['Tanggal'], comparison_df['Prediksi'], label='Prediksi', marker='x')
    ax.set_title('Perbandingan Nilai Aktual dan Prediksi')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Saham')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
