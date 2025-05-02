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
        Holiday('Natal', month=12, day=25)
    ]

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset (Excel)", type=["xlsx"])
if uploaded_file is not None:
    # Baca dataset
    data = pd.read_excel(uploaded_file)
    data = data.dropna()
    data = data.sort_values(by='Date')
    
    st.write("Dataset yang diupload (5 data terakhir):")
    st.write(data.tail())

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
            max_stall_iterations = st.number_input("Iterasi Stagnasi Maks", min_value=1, max_value=50, value=10)
        
        # Batas parameter
        st.subheader("Batas Parameter")
        col1, col2, col3 = st.columns(3)
        with col1:
            c_min = st.number_input("C min", value=0.1, format="%.4f")
            c_max = st.number_input("C max", value=1000.0, format="%.4f")
        with col2:
            gamma_min = st.number_input("gamma min", value=0.0001, format="%.4f")
            gamma_max = st.number_input("gamma max", value=1.0, format="%.4f")
        with col3:
            epsilon_min = st.number_input("epsilon min", value=0.0001, format="%.4f")
            epsilon_max = st.number_input("epsilon max", value=1.0, format="%.4f")
        
        lb = [c_min, gamma_min, epsilon_min]
        ub = [c_max, gamma_max, epsilon_max]
        
        if st.button("Jalankan FOA"):
            with st.spinner('Menjalankan optimasi FOA...'):
                # Inisialisasi
                dim = len(lb)
                flies = np.random.uniform(lb, ub, (n_flies, dim))
                best_fly = flies[0]
                best_mape = float('inf')
                convergence_history = []
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for iteration in range(n_iterations):
                    # Update progress
                    progress = (iteration + 1) / n_iterations
                    progress_bar.progress(progress)
                    status_text.text(f"Iterasi {iteration + 1}/{n_iterations} | MAPE Terbaik: {best_mape:.4f}%")
                    
                    # Update posisi lalat
                    for i in range(n_flies):
                        flies[i] = best_fly + np.random.normal(0, 1, dim) * (np.array(ub) - np.array(lb)) * 0.1
                        flies[i] = np.clip(flies[i], lb, ub)
                    
                    # Hitung MAPE baru
                    current_mape = []
                    for i in range(n_flies):
                        C, gamma, epsilon = flies[i]
                        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                        model.fit(X_train, y_train)
                        y_pred_val = model.predict(X_val)
                        mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100
                        current_mape.append(mape)
                        if mape < best_mape:
                            best_mape = mape
                            best_fly = flies[i].copy()
                    
                    convergence_history.append(best_mape)
                
                # Hasil optimasi
                st.success("Optimasi FOA selesai!")
                
                # Plot konvergensi
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(convergence_history, 'b-o')
                ax.set_title('Konvergensi MAPE FOA')
                ax.set_xlabel('Iterasi')
                ax.set_ylabel('MAPE (%)')
                ax.grid(True)
                st.pyplot(fig)
                
                # Parameter terbaik
                st.subheader("Parameter Terbaik FOA")
                st.write(f"- C: {best_fly[0]:.6f}")
                st.write(f"- gamma: {best_fly[1]:.6f}")
                st.write(f"- epsilon: {best_fly[2]:.6f}")
                st.write(f"- MAPE Validation: {best_mape:.4f}%")
                
                # Evaluasi pada test set
                best_model = SVR(kernel='rbf', C=best_fly[0], gamma=best_fly[1], epsilon=best_fly[2])
                best_model.fit(X_train, y_train)
                y_pred_test = best_model.predict(X_test)
                mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
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
            max_stall_iterations = st.number_input("Iterasi Stagnasi Maks", min_value=1, max_value=50, value=10)
        
        # Batas parameter
        st.subheader("Batas Parameter")
        col1, col2, col3 = st.columns(3)
        with col1:
            c_min = st.number_input("C min", value=0.1, format="%.4f")
            c_max = st.number_input("C max", value=1000.0, format="%.4f")
        with col2:
            gamma_min = st.number_input("gamma min", value=0.0001, format="%.4f")
            gamma_max = st.number_input("gamma max", value=1.0, format="%.4f")
        with col3:
            epsilon_min = st.number_input("epsilon min", value=0.0001, format="%.4f")
            epsilon_max = st.number_input("epsilon max", value=1.0, format="%.4f")
        
        lb = [c_min, gamma_min, epsilon_min]
        ub = [c_max, gamma_max, epsilon_max]
        
        if st.button("Jalankan PSO"):
            with st.spinner('Menjalankan optimasi PSO...'):
                # Inisialisasi
                dim = len(lb)
                particles = np.random.uniform(lb, ub, (n_particles, dim))
                velocities = np.random.uniform(-np.array(ub)/10, np.array(ub)/10, (n_particles, dim))
                
                pbest_positions = particles.copy()
                pbest_scores = np.full(n_particles, np.inf)
                gbest_position = particles[0].copy()
                gbest_score = np.inf
                
                convergence_history = []
                
                # Parameter adaptif
                w_max = 0.9
                w_min = 0.2
                c1_initial = 2.5
                c1_final = 1.5
                c2_initial = 1.5
                c2_final = 2.5
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for iteration in range(n_iterations):
                    # Update progress
                    progress = (iteration + 1) / n_iterations
                    progress_bar.progress(progress)
                    status_text.text(f"Iterasi {iteration + 1}/{n_iterations} | MAPE Terbaik: {gbest_score:.4f}%")
                    
                    # Update parameter adaptif
                    w = w_max - (w_max - w_min) * iteration / n_iterations
                    c1 = c1_initial - (c1_initial - c1_final) * iteration / n_iterations
                    c2 = c2_initial + (c2_final - c2_initial) * iteration / n_iterations
                    
                    # Update setiap partikel
                    for i in range(n_particles):
                        # Update kecepatan
                        r1, r2 = np.random.rand(dim), np.random.rand(dim)
                        cognitive = c1 * r1 * (pbest_positions[i] - particles[i])
                        social = c2 * r2 * (gbest_position - particles[i])
                        velocities[i] = w * velocities[i] + cognitive + social
                        velocities[i] = np.clip(velocities[i], -np.array(ub)/5, np.array(ub)/5)
                        
                        # Update posisi
                        particles[i] += velocities[i]
                        particles[i] = np.clip(particles[i], lb, ub)
                        
                        # Evaluasi
                        model = SVR(kernel='rbf', C=particles[i][0], gamma=particles[i][1], epsilon=particles[i][2])
                        model.fit(X_train, y_train)
                        y_pred_val = model.predict(X_val)
                        mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100
                        
                        # Update personal best
                        if mape < pbest_scores[i]:
                            pbest_scores[i] = mape
                            pbest_positions[i] = particles[i].copy()
                        
                        # Update global best
                        if mape < gbest_score:
                            gbest_score = mape
                            gbest_position = particles[i].copy()
                    
                    convergence_history.append(gbest_score)
                
                # Hasil optimasi
                st.success("Optimasi PSO selesai!")
                
                # Plot konvergensi
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(convergence_history, 'b-o')
                ax.set_title('Konvergensi MAPE PSO')
                ax.set_xlabel('Iterasi')
                ax.set_ylabel('MAPE (%)')
                ax.grid(True)
                st.pyplot(fig)
                
                # Parameter terbaik
                st.subheader("Parameter Terbaik PSO")
                st.write(f"- C: {gbest_position[0]:.6f}")
                st.write(f"- gamma: {gbest_position[1]:.6f}")
                st.write(f"- epsilon: {gbest_position[2]:.6f}")
                st.write(f"- MAPE Validation: {gbest_score:.4f}%")
                
                # Evaluasi pada test set
                best_model = SVR(kernel='rbf', C=gbest_position[0], gamma=gbest_position[1], epsilon=gbest_position[2])
                best_model.fit(X_train, y_train)
                y_pred_test = best_model.predict(X_test)
                mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
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
        next_pred = best_model.predict([[current_lag]])[0]
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
