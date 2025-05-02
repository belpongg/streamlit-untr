#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Tetapkan seed untuk hasil yang konsisten
np.random.seed(42)

# Load dataset
file_path = 'C:/Users/HP PAVILION/Downloads/utut.xlsx'
data = pd.read_excel(file_path)
data = data.dropna()
data = data.sort_values(by='Date')

# Features and target
X = data[['lag1']].values
y = data['y'].values

# Split data into training, validation, and testing sets (60:20:20)
# Pastikan tanggal juga ikut terbagi
X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
    X, y, data['Date'], test_size=0.4, shuffle=False, random_state=42
)
X_val, X_test, y_val, y_test, dates_val, dates_test = train_test_split(
    X_temp, y_temp, dates_temp, test_size=0.5, shuffle=False, random_state=42
)

# Define the Fruit Fly Optimization Algorithm
def foa(n_iterations, n_flies, lb, ub):
    lb = np.array(lb)  # Convert lb to NumPy array
    ub = np.array(ub)  # Convert ub to NumPy array
    dim = len(lb)
    
    # Initialize flies
    flies = np.random.uniform(lb, ub, (n_flies, dim))
    best_fly = flies[0]
    best_mape = float('inf')
    
    # Store convergence history
    convergence_history = []
    
    # Tabel 1: Nilai Awal Parameter Lalat
    print("Tabel 1: Nilai Awal Parameter Lalat")
    print("Lalat\tC\t\tgamma\t\tepsilon")
    for i, fly in enumerate(flies):
        print(f"{i+1}\t\t{fly[0]:.4f}\t{fly[1]:.4f}\t{fly[2]:.4f}")
    
    # Hitung nilai fitness (MAPE) untuk setiap lalat
    mape_values = []
    for i, fly in enumerate(flies):
        C, gamma, epsilon = fly
        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        model.fit(X_train, y_train)  # Melatih model menggunakan data training
        y_pred_val = model.predict(X_val)  # Memprediksi menggunakan data validation
        mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100  # Menghitung MAPE dari data validation
        mape_values.append(mape)
        if mape < best_mape:
            best_mape = mape
            best_fly = fly.copy()
    
    # Tabel 2: Nilai Kelayakan (Fitness) Iterasi 0
    print("\nTabel 2: Nilai Kelayakan (Fitness) Iterasi 0")
    print("Lalat\tCurrent Fitness (MAPE)")
    for i, mape in enumerate(mape_values):
        print(f"{i+1}\t\t{mape:.4f}%")
    
    # Tampilkan best smell
    print(f"\nBest Smell (Nilai Fitness Terbaik) Sebelum Iterasi: {best_mape:.4f}%")
    print(f"Posisi Lalat Terbaik: C={best_fly[0]:.4f}, gamma={best_fly[1]:.4f}, epsilon={best_fly[2]:.4f}")
    
    # Mulai iterasi FOA
    for iteration in range(n_iterations):
        # Update flies position based on best fly
        for i in range(n_flies):
            flies[i] = best_fly + np.random.normal(0, 1, dim) * (ub - lb) * 0.1
            flies[i] = np.clip(flies[i], lb, ub)  # Pastikan dalam batas
        
        # Calculate MAPE for new flies
        for i, fly in enumerate(flies):
            C, gamma, epsilon = fly
            model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_train, y_train)  # Melatih model menggunakan data training
            y_pred_val = model.predict(X_val)  # Memprediksi menggunakan data validation
            mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100  # Menghitung MAPE dari data validation
            if mape < best_mape:
                best_mape = mape
                best_fly = fly.copy()
        
        # Simpan nilai MAPE terbaik untuk visualisasi konvergensi
        convergence_history.append(best_mape)
    
    # Tabel 3: Posisi Lalat yang Diperbarui
    print("\nTabel 3: Posisi Lalat yang Diperbarui")
    print("Lalat\tC\t\tgamma\t\tepsilon")
    for i, fly in enumerate(flies):
        print(f"{i+1}\t\t{fly[0]:.4f}\t{fly[1]:.4f}\t{fly[2]:.4f}")
    
    # Tabel 4: Lalat dan MAPE Konvergen pada Suatu Nilai
    print("\nTabel 4: Lalat dan MAPE Konvergen pada Suatu Nilai")
    print("Lalat\tC\t\tgamma\t\tepsilon\tMAPE")
    for i, fly in enumerate(flies):
        print(f"{i+1}\t\t{fly[0]:.4f}\t{fly[1]:.4f}\t{fly[2]:.4f}\t{best_mape:.4f}%")
    
    # Plot konvergensi MAPE
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history, marker='o', linestyle='-', color='blue')
    plt.title('Konvergensi MAPE Selama Iterasi FOA')
    plt.xlabel('Iterasi')
    plt.ylabel('Best MAPE (%)')
    plt.grid(True)
    plt.show()
    
    return best_fly, best_mape

# Define the bounds for C, gamma, and epsilon
lb = [0.1, 0.0001, 0.0001]
ub = [1000, 1, 1]

# Perform FOA optimization
best_params, best_mape = foa(n_iterations=100, n_flies=30, lb=lb, ub=ub)

# Train the SVR model with the best parameters found by FOA
best_C, best_gamma, best_epsilon = best_params
best_model = SVR(kernel='rbf', C=best_C, gamma=best_gamma, epsilon=best_epsilon)
best_model.fit(X_train, y_train)  # Melatih model menggunakan data training

# Predict on the test set
y_pred_test = best_model.predict(X_test)  # Memprediksi menggunakan data testing

# Calculate MAPE for the test set
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
print(f"\nBest Parameters (C, gamma, epsilon): {best_params}")
print(f"MAPE with FOA-optimized SVR on Test Set: {mape_test:.4f}%")

# Plot perbandingan nilai aktual dan prediksi
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Nilai Aktual', marker='o', linestyle='-', color='blue')
plt.plot(y_pred_test, label='Nilai Prediksi', marker='x', linestyle='--', color='red')
plt.title('Perbandingan Nilai Aktual dan Prediksi pada Data Testing')
plt.xlabel('Data Point')
plt.ylabel('Nilai')
plt.legend()
plt.grid(True)
plt.show()

# Simpan hasil perbandingan data aktual dan prediksi ke dalam file Excel
results = pd.DataFrame({
    'Tanggal': dates_test,  # Tambahkan kolom tanggal
    'Aktual': y_test,
    'Prediksi': y_pred_test
})
results.to_excel('C:/Users/HP PAVILION/Downloads/hasil_prediksi.xlsx', index=False)
print("\nHasil perbandingan data aktual dan prediksi telah disimpan ke dalam file 'hasil_prediksi.xlsx'.")

# In[ ]:




