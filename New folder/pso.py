#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay  # Business Day

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
file_path = 'C:/Users/HP PAVILION/Downloads/utut.xlsx'
try:
    data = pd.read_excel(file_path)
    data = data.dropna()
    data = data.sort_values(by='Date')
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Features and target
X = data[['lag1']].values
y = data['y'].values

# Split data into training, validation, and testing sets (60:20:20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False, random_state=42)

# Define the PSO algorithm for SVR parameter optimization
def pso_svr(n_iterations, n_particles, lb, ub):
    dim = len(lb)  # Dimensi (C, gamma, epsilon)
    w_max = 0.9  # Inertia weight max
    w_min = 0.4  # Inertia weight min
    c1 = 2.0  # Cognitive parameter
    c2 = 2.0  # Social parameter
    v_max = (np.array(ub) - np.array(lb)) * 0.2  # Maximum velocity (20% dari rentang parameter)
    
    # Inisialisasi posisi dan kecepatan partikel
    particles = np.random.uniform(lb, ub, (n_particles, dim))
    velocities = np.random.uniform(-v_max, v_max, (n_particles, dim))  # Kecepatan diinisialisasi secara acak
    
    # Tampilkan kecepatan awal
    print("\nKecepatan Awal Partikel:")
    print("Partikel\tKecepatan C\tKecepatan gamma\tKecepatan epsilon")
    for i, velocity in enumerate(velocities):
        print(f"{i+1}\t\t{velocity[0]:.4f}\t\t{velocity[1]:.4f}\t\t{velocity[2]:.4f}")
    
    # Inisialisasi best positions dan best scores
    pbest_positions = particles.copy()
    pbest_scores = np.array([float('inf')] * n_particles)
    gbest_position = pbest_positions[0]
    gbest_score = float('inf')
    gbest_scores_history = []
    
    # Tabel 4.2: Nilai Awal Parameter Partikel
    print("\nTabel 4.2: Nilai Awal Parameter Partikel")
    print("Partikel\tC\t\tgamma\t\tepsilon")
    for i, particle in enumerate(particles):
        print(f"{i+1}\t\t{particle[0]:.4f}\t{particle[1]:.4f}\t{particle[2]:.4f}")
    
    # Tabel 4.3: Nilai Fitness (MAPE) Awal
    print("\nTabel 4.3: Nilai Fitness (MAPE) Awal")
    print("Partikel\tFitness (MAPE)")
    for i, particle in enumerate(particles):
        C, gamma, epsilon = particle
        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        model.fit(X_train, y_train)  # Melatih model menggunakan data training
        y_pred_val = model.predict(X_val)  # Memprediksi menggunakan data validation
        mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100  # Menghitung MAPE dari data validation
        pbest_scores[i] = mape
        print(f"{i+1}\t\t{mape:.4f}%")
        if mape < gbest_score:
            gbest_score = mape
            gbest_position = particle
    
    # Tabel 4.4: Nilai Px_best untuk Setiap Parameter dan Pf_best
    print("\nTabel 4.4: Nilai Px_best untuk Setiap Parameter dan Pf_best")
    print("Partikel\tPx_best C\tPx_best gamma\tPx_best epsilon\tPf_best (MAPE)")
    for i in range(n_particles):
        print(f"{i+1}\t\t{pbest_positions[i][0]:.4f}\t\t{pbest_positions[i][1]:.4f}\t\t{pbest_positions[i][2]:.4f}\t\t{pbest_scores[i]:.4f}%")
    
    # Iterasi PSO
    for iteration in range(n_iterations):
        w = w_max - (w_max - w_min) * iteration / n_iterations  # Update inertia weight
        for i in range(n_particles):
            # Update kecepatan
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - particles[i]) +
                             c2 * r2 * (gbest_position - particles[i]))
            velocities[i] = np.clip(velocities[i], -v_max, v_max)  # Batasi kecepatan
            
            # Update posisi
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], lb, ub)  # Batasi posisi dalam lb dan ub
            
            # Hitung fitness baru
            C, gamma, epsilon = particles[i]
            model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_train, y_train)  # Melatih model menggunakan data training
            y_pred_val = model.predict(X_val)  # Memprediksi menggunakan data validation
            mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100  # Menghitung MAPE dari data validation
            
            # Update pbest dan gbest
            if mape < pbest_scores[i]:
                pbest_scores[i] = mape
                pbest_positions[i] = particles[i]
            if mape < gbest_score:
                gbest_score = mape
                gbest_position = particles[i]
        
        # Tampilkan kecepatan pada setiap iterasi (opsional)
        if (iteration + 1) % 10 == 0:  # Cetak setiap 10 iterasi
            print(f"\nKecepatan Partikel pada Iterasi {iteration + 1}:")
            print("Partikel\tKecepatan C\tKecepatan gamma\tKecepatan epsilon")
            for i, velocity in enumerate(velocities):
                print(f"{i+1}\t\t{velocity[0]:.4f}\t\t{velocity[1]:.4f}\t\t{velocity[2]:.4f}")
        
        gbest_scores_history.append(gbest_score)
        print(f"Iteration {iteration+1}/{n_iterations}, Best MAPE: {gbest_score:.4f}%")
    
    # Tampilkan kecepatan terakhir
    print("\nKecepatan Terakhir Partikel:")
    print("Partikel\tKecepatan C\tKecepatan gamma\tKecepatan epsilon")
    for i, velocity in enumerate(velocities):
        print(f"{i+1}\t\t{velocity[0]:.4f}\t\t{velocity[1]:.4f}\t\t{velocity[2]:.4f}")
    
    # Informasi Posisi Terbaik Partikel (Menggantikan Tabel 4.6)
    print("\nPosisi Terbaik Partikel Setelah Iterasi:")
    print(f"C = {gbest_position[0]:.4f}, gamma = {gbest_position[1]:.4f}, epsilon = {gbest_position[2]:.4f}")
    
    # Tabel 4.7: Partikel dan MAPE Konvergen pada Suatu Nilai
    print("\nTabel 4.7: Partikel dan MAPE Konvergen pada Suatu Nilai")
    print("Partikel\tC\t\tgamma\t\tepsilon\tMAPE (Validation)")
    for i in range(n_particles):
        print(f"{i+1}\t\t{pbest_positions[i][0]:.4f}\t{pbest_positions[i][1]:.4f}\t{pbest_positions[i][2]:.4f}\t{pbest_scores[i]:.4f}%")
    
    # Plot convergence
    plt.plot(range(n_iterations), gbest_scores_history)
    plt.xlabel('Iteration')
    plt.ylabel('Best MAPE')
    plt.title('Convergence of PSO')
    plt.show()
    
    return gbest_position, gbest_score

# Define the bounds for C, gamma, and epsilon
lb = [0.1, 0.0001, 0.0001]
ub = [1000, 1, 1]

# Perform PSO optimization
best_params, best_mape = pso_svr(n_iterations=100, n_particles=30, lb=lb, ub=ub)

# Train the SVR model with the best parameters found by PSO
best_C, best_gamma, best_epsilon = best_params
best_model = SVR(kernel='rbf', C=best_C, gamma=best_gamma, epsilon=best_epsilon)
best_model.fit(X_train, y_train)  # Melatih model menggunakan data training

# Predict on the test set
y_pred_test = best_model.predict(X_test)  # Memprediksi menggunakan data testing

# Calculate MAPE for the test set
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
print(f"\nBest Parameters (C, gamma, epsilon): {best_params}")
print(f"MAPE with PSO-optimized SVR on Test Set: {mape_test:.4f}%")

# Create a DataFrame to store the comparison results
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_test
})

# Save the comparison results to an Excel file
output_file_path = 'C:/Users/HP PAVILION/Downloads/comparison_results.xlsx'
comparison_df.to_excel(output_file_path, index=False)
print(f"\nComparison results saved to {output_file_path}")

# Plot perbandingan actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Values', color='blue', marker='o', linestyle='-', linewidth=1, markersize=5)
plt.plot(y_pred_test, label='Predicted Values', color='red', marker='x', linestyle='--', linewidth=1, markersize=5)
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Actual vs Predicted Values (PSO-Optimized SVR)')
plt.legend()
plt.grid(True)
plt.show()

# ====================================================
# 6. Peramalan 15 Hari Trading ke Depan (FIXED VERSION)
# ====================================================

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
import datetime

# 1. Definisikan Kalender Libur yang Akurat
class ExactHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        # Libur Desember
        Holiday('New Year Eve', month=12, day=31),
        
        # Libur Januari
        Holiday('New Year', month=1, day=1),
        # Holiday('Libur Contoh', month=1, day=2),  # Hapus komentar jika 2 Jan libur
        # Tambahkan semua libur resmi sesuai bursa
    ]

# 2. Dapatkan tanggal terakhir dari data
last_date = data['Date'].iloc[-1]
print(f"\nTanggal terakhir dalam dataset: {last_date}")

# 3. Generate kalender libur
cal = ExactHolidayCalendar()
holidays = cal.holidays(start=last_date, end=last_date + datetime.timedelta(days=60))

# 4. Fungsi validasi hari trading
def is_trading_day(date):
    # Senin-Jumat dan bukan hari libur
    return date.weekday() < 5 and date not in holidays

# 5. Generate 15 hari trading VALID
future_dates = []
current_date = last_date
found_dates = 0

while found_dates < 15:
    current_date += BDay(1)
    if is_trading_day(current_date):
        future_dates.append(current_date)
        found_dates += 1
        print(f"Menambahkan {current_date.date()} sebagai hari trading ke-{found_dates}")

# 6. Lakukan prediksi
current_lag = data['y'].iloc[-1]
forecast_values = []
for _ in range(15):
    forecast_values.append(best_model.predict([[current_lag]])[0])
    current_lag = forecast_values[-1]

# 7. Hasil akhir
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast': forecast_values
})

# 8. Simpan dan tampilkan
forecast_df.to_excel('forecast_FIXED.xlsx', index=False)

print("\nHasil Peramalan 15 Hari Trading yang Benar:")
print(forecast_df.to_string())

# Plotting
plt.figure(figsize=(12,6))
plt.plot(data['Date'][-30:], data['y'][-30:], 'b-', label='Data Historis')
plt.plot(forecast_df['Date'], forecast_df['Forecast'], 'ro--', label='Prediksi')
plt.title('Peramalan 15 Hari Trading (Fix)')
plt.legend()
plt.grid(True)
plt.show()

# In[ ]:




