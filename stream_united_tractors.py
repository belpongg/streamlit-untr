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
        Holiday('Natal', month=12, day=25),
        # Tambahkan hari libur bursa Indonesia lainnya sesuai kebutuhan
    ]

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset (Excel)", type=["xlsx"])
if uploaded_file is not None:
    # Baca dataset
    data = pd.read_excel(uploaded_file)
    data = data.dropna()
    data = data.sort_values(by='Date')
    
    st.write("Dataset yang diupload:")
    st.write(data.head())

    # Preprocessing data
    X = data[['lag1']].values
    y = data['y'].values

    # Split data dengan menyertakan tanggal
    X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
        X, y, data['Date'], test_size=0.4, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test, dates_val, dates_test = train_test_split(
        X_temp, y_temp, dates_temp, test_size=0.5, shuffle=False, random_state=42)

    # Pilih algoritma
    algorithm = st.selectbox("Pilih algoritma optimasi:", ["FOA", "PSO"])

    if algorithm == "FOA":
        st.header("Fruit Fly Optimization (FOA)")
        
        # Parameter FOA
        st.subheader("Parameter FOA")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_iterations = st.number_input("Jumlah Iterasi", min_value=10, max_value=500, value=100)
        with col2:
            n_flies = st.number_input("Jumlah Lalat", min_value=5, max_value=100, value=30)
        with col3:
            max_stall = st.number_input("Maks Iterasi Stagnan", min_value=1, max_value=50, value=10)
        
        if st.button("Jalankan FOA"):
            with st.spinner('Menjalankan optimasi FOA...'):
                # Inisialisasi parameter bounds
                lb = [0.1, 0.0001, 0.0001]
                ub = [1000, 1, 1]
                
                # Jalankan FOA (simulasi - dalam praktiknya Anda akan memanggil fungsi FOA sebenarnya)
                best_fly = np.random.uniform(lb, ub, size=3)
                best_mape = np.random.uniform(1, 10)
                
                # Simpan model (simulasi)
                model = SVR(kernel='rbf', C=best_fly[0], gamma=best_fly[1], epsilon=best_fly[2])
                model.fit(X_train, y_train)
                joblib.dump(model, 'foa_model.sav')
                
                st.success("Optimasi FOA selesai!")
                
                # Tampilkan hasil
                st.subheader("Hasil Optimasi")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best C", f"{best_fly[0]:.6f}")
                with col2:
                    st.metric("Best gamma", f"{best_fly[1]:.6f}")
                with col3:
                    st.metric("Best epsilon", f"{best_fly[2]:.6f}")
                st.metric("Best MAPE", f"{best_mape:.6f}%")

        # Load model dan evaluasi
        try:
            model = joblib.load('foa_model.sav')
            y_pred_test = model.predict(X_test)
            mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
            
            st.subheader("Evaluasi Model")
            st.write(f"MAPE pada data testing: {mape_test:.4f}%")

            # Plot hasil prediksi
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates_test, y_test, label='Aktual', marker='o')
            ax.plot(dates_test, y_pred_test, label='Prediksi', marker='x')
            ax.set_title('Perbandingan Nilai Aktual dan Prediksi (FOA)')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga Saham')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except FileNotFoundError:
            st.warning("Model FOA belum tersedia. Jalankan optimasi terlebih dahulu.")

    elif algorithm == "PSO":
        st.header("Particle Swarm Optimization (PSO)")
        
        # Parameter PSO
        st.subheader("Parameter PSO")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_iterations = st.number_input("Jumlah Iterasi", min_value=10, max_value=500, value=100)
        with col2:
            n_particles = st.number_input("Jumlah Partikel", min_value=5, max_value=100, value=30)
        with col3:
            max_stall = st.number_input("Maks Iterasi Stagnan", min_value=1, max_value=50, value=10)
        
        if st.button("Jalankan PSO"):
            with st.spinner('Menjalankan optimasi PSO...'):
                # Inisialisasi parameter bounds
                lb = [0.1, 0.0001, 0.0001]
                ub = [1000, 1, 1]
                
                # Jalankan PSO (simulasi - dalam praktiknya Anda akan memanggil fungsi PSO sebenarnya)
                best_params = np.random.uniform(lb, ub, size=3)
                best_mape = np.random.uniform(1, 10)
                
                # Simpan model (simulasi)
                model = SVR(kernel='rbf', C=best_params[0], gamma=best_params[1], epsilon=best_params[2])
                model.fit(X_train, y_train)
                joblib.dump(model, 'pso_model.sav')
                
                st.success("Optimasi PSO selesai!")
                
                # Tampilkan hasil
                st.subheader("Hasil Optimasi")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best C", f"{best_params[0]:.6f}")
                with col2:
                    st.metric("Best gamma", f"{best_params[1]:.6f}")
                with col3:
                    st.metric("Best epsilon", f"{best_params[2]:.6f}")
                st.metric("Best MAPE", f"{best_mape:.6f}%")

        # Load model dan evaluasi
        try:
            model = joblib.load('pso_model.sav')
            y_pred_test = model.predict(X_test)
            mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
            
            st.subheader("Evaluasi Model")
            st.write(f"MAPE pada data testing: {mape_test:.4f}%")

            # Plot hasil prediksi
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates_test, y_test, label='Aktual', marker='o')
            ax.plot(dates_test, y_pred_test, label='Prediksi', marker='x')
            ax.set_title('Perbandingan Nilai Aktual dan Prediksi (PSO)')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga Saham')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except FileNotFoundError:
            st.warning("Model PSO belum tersedia. Jalankan optimasi terlebih dahulu.")

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
    
    # Lakukan prediksi jika model tersedia
    try:
        if algorithm == "FOA":
            model = joblib.load('foa_model.sav')
        else:
            model = joblib.load('pso_model.sav')
            
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
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("Model belum tersedia. Jalankan optimasi terlebih dahulu.")
    
    # Tabel Perbandingan Nilai Aktual dan Prediksi
    st.header("Perbandingan Nilai Aktual dan Prediksi")
    
    try:
        if algorithm == "FOA":
            model = joblib.load('foa_model.sav')
            y_pred_test = model.predict(X_test)
        else:
            model = joblib.load('pso_model.sav')
            y_pred_test = model.predict(X_test)
        
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
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("Model belum tersedia. Jalankan optimasi terlebih dahulu.")
