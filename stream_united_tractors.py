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

# Set random seed untuk reproduktibilitas
np.random.seed(42)

# Judul aplikasi
st.title("Perbandingan Kinerja FOA dan PSO dalam Optimasi SVR untuk Peramalan Harga Saham United Tractors")

# 1. Kalender Libur Khusus
class IDXTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1),
        Holiday('New Year Eve', month=12, day=31),
        Holiday('Natal', month=12, day=25),
    ]

# Implementasi FOA
def foa_optimization(X_train, y_train, X_val, y_val, n_iterations=100, n_flies=30, 
                    lb=[0.1, 0.0001, 0.0001], ub=[1000, 1, 1], 
                    tolerance=0.0001, max_stall_iterations=10):
    
    lb = np.array(lb)
    ub = np.array(ub)
    dim = len(lb)
    
    # Inisialisasi lalat
    flies = np.random.uniform(lb, ub, (n_flies, dim))
    best_fly = flies[0]
    best_mape = float('inf')
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    history = {'best_mape': [], 'best_params': []}
    
    # Hitung nilai fitness awal
    initial_mape = []
    for i in range(n_flies):
        model = SVR(kernel='rbf', C=flies[i][0], gamma=flies[i][1], epsilon=flies[i][2])
        model.fit(X_train, y_train)
        mape = mean_absolute_percentage_error(y_val, model.predict(X_val)) * 100
        initial_mape.append(mape)
        if mape < best_mape:
            best_mape = mape
            best_fly = flies[i].copy()
    
    history['best_mape'].append(best_mape)
    history['best_params'].append(best_fly.copy())
    
    stall_counter = 0
    
    for iteration in range(1, n_iterations + 1):
        previous_best = best_mape
        
        # Update posisi lalat
        for i in range(n_flies):
            flies[i] = best_fly + np.random.normal(0, 1, dim) * (ub - lb) * 0.1
            flies[i] = np.clip(flies[i], lb, ub)
        
        # Hitung MAPE baru
        current_mape = []
        for i in range(n_flies):
            model = SVR(kernel='rbf', C=flies[i][0], gamma=flies[i][1], epsilon=flies[i][2])
            model.fit(X_train, y_train)
            mape = mean_absolute_percentage_error(y_val, model.predict(X_val)) * 100
            current_mape.append(mape)
            if mape < best_mape:
                best_mape = mape
                best_fly = flies[i].copy()
        
        # Update progress
        progress = (iteration + 1) / n_iterations
        progress_bar.progress(progress)
        status_text.text(f"Iterasi {iteration}/{n_iterations} - Best MAPE: {best_mape:.4f}%")
        
        history['best_mape'].append(best_mape)
        history['best_params'].append(best_fly.copy())
        
        # Cek konvergensi
        mape_change = previous_best - best_mape
        if abs(mape_change) < tolerance:
            stall_counter += 1
            if stall_counter >= max_stall_iterations:
                break
        else:
            stall_counter = 0
    
    return best_fly, best_mape, history

# Implementasi PSO
def pso_optimization(X_train, y_train, X_val, y_val, n_iterations=100, n_particles=30, 
                    lb=[0.1, 0.0001, 0.0001], ub=[1000, 1, 1], 
                    tolerance=0.0001, max_stall_iterations=10):
    
    dimensi = len(lb)
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Parameter PSO
    w_max = 0.9
    w_min = 0.2
    c1_awal = 2.5
    c1_akhir = 1.5
    c2_awal = 1.5  
    c2_akhir = 2.5
    v_maks = (ub - lb) * 0.3
    
    # Inisialisasi partikel
    particles = np.random.uniform(lb, ub, (n_particles, dimensi))
    velocities = np.random.uniform(-v_maks, v_maks, (n_particles, dimensi))
    
    # Inisialisasi best
    pbest_positions = particles.copy()
    pbest_scores = np.full(n_particles, np.inf)
    gbest_position = particles[0].copy()
    gbest_score = np.inf
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    history = {'best_mape': [], 'best_params': []}
    
    for iteration in range(n_iterations):
        # Parameter adaptif
        w = w_max - (w_max - w_min) * iteration / n_iterations
        c1 = c1_awal - (c1_awal - c1_akhir) * iteration / n_iterations
        c2 = c2_awal + (c2_akhir - c2_awal) * iteration / n_iterations
        
        previous_best = gbest_score
        
        # Update setiap partikel
        for i in range(n_particles):
            # Update kecepatan
            r1, r2 = np.random.rand(dimensi), np.random.rand(dimensi)
            cognitive = c1 * r1 * (pbest_positions[i] - particles[i])
            social = c2 * r2 * (gbest_position - particles[i])
            velocities[i] = w * velocities[i] + cognitive + social
            velocities[i] = np.clip(velocities[i], -v_maks, v_maks)
            
            # Update posisi
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], lb, ub)
            
            # Evaluasi
            model = SVR(kernel='rbf', C=particles[i][0], gamma=particles[i][1], epsilon=particles[i][2])
            model.fit(X_train, y_train)
            mape = mean_absolute_percentage_error(y_val, model.predict(X_val)) * 100
            
            # Update personal best
            if mape < pbest_scores[i]:
                pbest_scores[i] = mape
                pbest_positions[i] = particles[i].copy()
            
            # Update global best
            if mape < gbest_score:
                gbest_score = mape
                gbest_position = particles[i].copy()
        
        # Update progress
        progress = (iteration + 1) / n_iterations
        progress_bar.progress(progress)
        status_text.text(f"Iterasi {iteration + 1}/{n_iterations} - Best MAPE: {gbest_score:.4f}%")
        
        history['best_mape'].append(gbest_score)
        history['best_params'].append(gbest_position.copy())
        
        # Cek konvergensi
        improvement = previous_best - gbest_score
        if abs(improvement) < tolerance:
            stall_counter += 1
            if stall_counter >= max_stall_iterations:
                break
        else:
            stall_counter = 0
    
    return gbest_position, gbest_score, history

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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_iterations = st.number_input("Jumlah Iterasi", min_value=10, max_value=500, value=100)
        with col2:
            n_flies = st.number_input("Jumlah Lalat", min_value=5, max_value=100, value=30)
        with col3:
            tolerance = st.number_input("Toleransi", min_value=0.00001, max_value=0.1, value=0.0001, format="%.5f")
        with col4:
            max_stall = st.number_input("Maks Iterasi Stagnan", min_value=1, max_value=50, value=10)
        
        if st.button("Jalankan FOA"):
            with st.spinner('Menjalankan optimasi FOA...'):
                best_params, best_mape, history = foa_optimization(
                    X_train, y_train, X_val, y_val,
                    n_iterations=n_iterations,
                    n_flies=n_flies,
                    tolerance=tolerance,
                    max_stall_iterations=max_stall
                )
                
                # Simpan model
                model = SVR(kernel='rbf', C=best_params[0], gamma=best_params[1], epsilon=best_params[2])
                model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))
                joblib.dump(model, 'foa_model.sav')
                
                st.success("Optimasi FOA selesai!")
                
                # Tampilkan hasil
                st.subheader("Hasil Optimasi")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best C", f"{best_params[0]:.6f}")
                with col2:
                    st.metric("Best gamma", f"{best_params[1]:.6f}")
                with col3:
                    st.metric("Best epsilon", f"{best_params[2]:.6f}")
                st.metric("Best MAPE (Validation)", f"{best_mape:.6f}%")

                # Plot konvergensi
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(history['best_mape'], 'b-o')
                ax.set_title('Konvergensi FOA')
                ax.set_xlabel('Iterasi')
                ax.set_ylabel('MAPE (%)')
                ax.grid(True)
                st.pyplot(fig)

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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_iterations = st.number_input("Jumlah Iterasi", min_value=10, max_value=500, value=100)
        with col2:
            n_particles = st.number_input("Jumlah Partikel", min_value=5, max_value=100, value=30)
        with col3:
            tolerance = st.number_input("Toleransi", min_value=0.00001, max_value=0.1, value=0.0001, format="%.5f")
        with col4:
            max_stall = st.number_input("Maks Iterasi Stagnan", min_value=1, max_value=50, value=10)
        
        if st.button("Jalankan PSO"):
            with st.spinner('Menjalankan optimasi PSO...'):
                best_params, best_mape, history = pso_optimization(
                    X_train, y_train, X_val, y_val,
                    n_iterations=n_iterations,
                    n_particles=n_particles,
                    tolerance=tolerance,
                    max_stall_iterations=max_stall
                )
                
                # Simpan model
                model = SVR(kernel='rbf', C=best_params[0], gamma=best_params[1], epsilon=best_params[2])
                model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))
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
                st.metric("Best MAPE (Validation)", f"{best_mape:.6f}%")

                # Plot konvergensi
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(history['best_mape'], 'b-o')
                ax.set_title('Konvergensi PSO')
                ax.set_xlabel('Iterasi')
                ax.set_ylabel('MAPE (%)')
                ax.grid(True)
                st.pyplot(fig)

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

    # PREDIKSI 15 HARI TRADING KE DEPAN
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
