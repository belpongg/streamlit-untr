import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib
from pandas.tseries.offsets import BDay
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
import datetime

# Judul aplikasi
st.title("Perbandingan Kinerja FOA dan PSO dalam Optimasi SVR untuk Peramalan Harga Saham United Tractors")

# 1. Kalender Libur Khusus
class IDXTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1),
        Holiday('New Year Eve', month=12, day=31),
        Holiday('Christmas', month=12, day=25),
        # Tambahkan hari libur bursa Indonesia lainnya sesuai kebutuhan
    ]

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
    dates = data['Date'].values

    # Split data (60:20:20)
    X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
        X, y, dates, test_size=0.4, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test, dates_val, dates_test = train_test_split(
        X_temp, y_temp, dates_temp, test_size=0.5, shuffle=False, random_state=42)

    # Pilih algoritma
    algorithm = st.selectbox("Pilih algoritma optimasi:", ["FOA", "PSO"])

    if algorithm == "FOA":
        st.header("Fruit Fly Optimization (FOA)")
        
        # Load model (asumsi model sudah di-train dengan kode FOA terbaru)
        try:
            model = joblib.load('foa_model.sav')
            
            # Evaluasi model
            y_pred_test = model.predict(X_test)
            mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
            
            st.subheader("Hasil Evaluasi")
            st.write(f"MAPE pada data testing: {mape_test:.4f}%")
            
            # Tampilkan parameter terbaik
            st.subheader("Parameter Terbaik")
            st.write(f"- C: {model.C:.6f}")
            st.write(f"- Gamma: {model.gamma:.6f}")
            st.write(f"- Epsilon: {model.epsilon:.6f}")

            # Plot hasil prediksi
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(dates_test, y_test, label='Aktual', marker='o')
            ax.plot(dates_test, y_pred_test, label='Prediksi', marker='x')
            ax.set_title('Perbandingan Nilai Aktual dan Prediksi (FOA)')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga Saham')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Gagal memuat model FOA: {str(e)}")

    elif algorithm == "PSO":
        st.header("Particle Swarm Optimization (PSO)")
        
        # Load model (asumsi model sudah di-train dengan kode PSO terbaru)
        try:
            model = joblib.load('pso_model.sav')
            
            # Evaluasi model
            y_pred_test = model.predict(X_test)
            mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
            
            st.subheader("Hasil Evaluasi")
            st.write(f"MAPE pada data testing: {mape_test:.4f}%")
            
            # Tampilkan parameter terbaik
            st.subheader("Parameter Terbaik")
            st.write(f"- C: {model.C:.6f}")
            st.write(f"- Gamma: {model.gamma:.6f}")
            st.write(f"- Epsilon: {model.epsilon:.6f}")

            # Plot hasil prediksi
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(dates_test, y_test, label='Aktual', marker='o')
            ax.plot(dates_test, y_pred_test, label='Prediksi', marker='x')
            ax.set_title('Perbandingan Nilai Aktual dan Prediksi (PSO)')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga Saham')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Gagal memuat model PSO: {str(e)}")

    # ====================================================
    # PREDIKSI 15 HARI TRADING KE DEPAN
    # ====================================================
    st.header("Prediksi 15 Hari Trading ke Depan")
    
    # Dapatkan tanggal terakhir dari data
    last_date = data['Date'].iloc[-1]
    st.write(f"Tanggal terakhir dalam dataset: {last_date.strftime('%Y-%m-%d')}")
    
    # Generate kalender libur
    cal = IDXTradingCalendar()
    holidays = cal.holidays(start=last_date, end=last_date + datetime.timedelta(days=60))
    
    # Fungsi validasi hari trading
    def is_trading_day(date):
        return date.weekday() < 5 and date not in holidays
    
    # Generate 15 hari trading VALID
    future_dates = []
    current_date = last_date
    found_dates = 0
    
    with st.spinner('Menghitung hari trading yang valid...'):
        while found_dates < 15:
            current_date += BDay(1)
            if is_trading_day(current_date):
                future_dates.append(current_date)
                found_dates += 1
    
    # Lakukan prediksi
    current_lag = data['y'].iloc[-1]
    future_predictions = []
    for _ in range(15):
        next_pred = model.predict([[current_lag]])[0]
        future_predictions.append(next_pred)
        current_lag = next_pred
    
    # Buat DataFrame hasil prediksi
    forecast_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi_Harga': future_predictions
    })
    
    # Tampilkan hasil prediksi
    st.write("Hasil Prediksi 15 Hari Trading ke Depan:")
    st.dataframe(forecast_df.style.format({
        'Prediksi_Harga': '{:.2f}'
    }))
    
    # Plot hasil prediksi
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'][-30:], data['y'][-30:], 'b-', label='Data Historis')
    ax.plot(forecast_df['Tanggal'], forecast_df['Prediksi_Harga'], 'ro--', label='Prediksi')
    ax.set_title('Prediksi Harga Saham 15 Hari Trading ke Depan')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Saham')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Tabel Perbandingan Nilai Aktual dan Prediksi
    st.header("Perbandingan Nilai Aktual dan Prediksi")
    
    comparison_df = pd.DataFrame({
        'Tanggal': dates_test,
        'Aktual': y_test,
        'Prediksi': y_pred_test
    })
    
    st.write("Tabel Perbandingan Nilai Aktual dan Prediksi:")
    st.dataframe(comparison_df.style.format({
        'Aktual': '{:.2f}',
        'Prediksi': '{:.2f}'
    }))
    
    # Plot perbandingan
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(comparison_df['Tanggal'], comparison_df['Aktual'], label='Aktual', marker='o')
    ax.plot(comparison_df['Tanggal'], comparison_df['Prediksi'], label='Prediksi', marker='x')
    ax.set_title('Perbandingan Nilai Aktual dan Prediksi')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga Saham')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)
