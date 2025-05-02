#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Set random seed for reproducibility
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
X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
    X, y, data['Date'], test_size=0.4, shuffle=False, random_state=42
)
X_val, X_test, y_val, y_test, dates_val, dates_test = train_test_split(
    X_temp, y_temp, dates_temp, test_size=0.5, shuffle=False, random_state=42
)

def print_fly_table(flies, mape_values, iteration, best_fly=None):
    """Print detailed table for all flies"""
    print(f"\n{'='*120}")
    print(f"ITERATION {iteration} | TOTAL FLIES: {len(flies)}")
    print('='*120)
    print("| {:<6} | {:<12} | {:<12} | {:<12} | {:<20} | {:<15} |".format(
        "Fly", "C", "gamma", "epsilon", "MAPE (%)", "Status"))
    print('-'*120)
    
    for i in range(len(flies)):
        status = "BEST" if best_fly is not None and np.array_equal(flies[i], best_fly) else ""
        print("| {:<6} | {:<12.6f} | {:<12.6f} | {:<12.6f} | {:<20.6f} | {:<15} |".format(
            i+1, flies[i][0], flies[i][1], flies[i][2], mape_values[i], status))
    print('='*120)

def foa(n_iterations, n_flies, lb, ub, tolerance=0.0001, max_stall_iterations=10):
    lb = np.array(lb)
    ub = np.array(ub)
    dim = len(lb)
    
    # Initialize flies
    flies = np.random.uniform(lb, ub, (n_flies, dim))
    best_fly = flies[0]
    best_mape = float('inf')
    
    # Store convergence history
    convergence_history = []
    all_iterations_data = []
    stall_counter = 0
    
    # ==============================================
    # Tabel 1: Nilai Awal Parameter Lalat
    # ==============================================
    print("\n" + "="*50)
    print("TABEL 1: NILAI AWAL PARAMETER LALAT")
    print("="*50)
    print("-" * 100)
    print("| {:<6} | {:<12} | {:<12} | {:<12} |".format("Lalat", "C", "gamma", "epsilon"))
    print("-" * 100)
    for i in range(n_flies):
        print("| {:<6} | {:<12.6f} | {:<12.6f} | {:<12.6f} |".format(
            i+1, flies[i][0], flies[i][1], flies[i][2]))
    print("-" * 100)
    
    # Hitung nilai fitness awal
    initial_mape = []
    for i in range(n_flies):
        C, gamma, epsilon = flies[i]
        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100
        initial_mape.append(mape)
        if mape < best_mape:
            best_mape = mape
            best_fly = flies[i].copy()
    
    # ==============================================
    # Tabel 2: Nilai Fitness Awal
    # ==============================================
    print("\n\n" + "="*50)
    print("TABEL 2: NILAI FITNESS AWAL (ITERASI 0)")
    print("="*50)
    print("-" * 100)
    print("| {:<6} | {:<20} | {:<12} | {:<12} | {:<12} |".format(
        "Lalat", "MAPE (%)", "C", "gamma", "epsilon"))
    print("-" * 100)
    for i in range(n_flies):
        print("| {:<6} | {:<20.6f} | {:<12.6f} | {:<12.6f} | {:<12.6f} |".format(
            i+1, initial_mape[i], flies[i][0], flies[i][1], flies[i][2]))
    print("-" * 100)
    
    # Tampilkan detail semua lalat di iterasi 0
    print_fly_table(flies, initial_mape, 0, best_fly)
    
    # Simpan data awal
    convergence_history.append(best_mape)
    all_iterations_data.append({
        'iteration': 0,
        'flies': flies.copy(),
        'mape_values': initial_mape.copy(),
        'best_fly': best_fly.copy(),
        'best_mape': best_mape
    })
    
    # ==============================================
    # Proses Iterasi FOA
    # ==============================================
    print("\n\n" + "="*50)
    print("PROSES ITERASI FOA (MENAMPILKAN SEMUA 30 LALAT)")
    print("="*50)
    
    for iteration in range(1, n_iterations + 1):
        previous_best = best_mape
        
        # Update posisi lalat
        for i in range(n_flies):
            flies[i] = best_fly + np.random.normal(0, 1, dim) * (ub - lb) * 0.1
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
        
        # Tampilkan detail semua lalat di iterasi ini
        print_fly_table(flies, current_mape, iteration, best_fly)
        
        # Simpan data iterasi
        convergence_history.append(best_mape)
        all_iterations_data.append({
            'iteration': iteration,
            'flies': flies.copy(),
            'mape_values': current_mape.copy(),
            'best_fly': best_fly.copy(),
            'best_mape': best_mape
        })
        
        # Cek konvergensi
        mape_change = previous_best - best_mape
        if abs(mape_change) < tolerance:
            stall_counter += 1
        else:
            stall_counter = 0
            
        if stall_counter >= max_stall_iterations:
            print(f"\n🔥 KONVERGEN pada iterasi {iteration}!")
            print(f"Perubahan MAPE < {tolerance} selama {max_stall_iterations} iterasi berturut-turut")
            break
    
    # ==============================================
    # Tabel 3: Posisi Lalat Terakhir
    # ==============================================
    print("\n\n" + "="*50)
    print("TABEL 3: POSISI LALAT TERAKHIR")
    print("="*50)
    print("-" * 100)
    print("| {:<6} | {:<12} | {:<12} | {:<12} |".format("Lalat", "C", "gamma", "epsilon"))
    print("-" * 100)
    for i in range(n_flies):
        print("| {:<6} | {:<12.6f} | {:<12.6f} | {:<12.6f} |".format(
            i+1, flies[i][0], flies[i][1], flies[i][2]))
    print("-" * 100)
    
    # ==============================================
    # Tabel 4: Nilai Fitness Terakhir
    # ==============================================
    print("\n\n" + "="*50)
    print("TABEL 4: NILAI FITNESS TERAKHIR")
    print("="*50)
    print("-" * 100)
    print("| {:<6} | {:<20} | {:<12} | {:<12} | {:<12} |".format(
        "Lalat", "MAPE (%)", "C", "gamma", "epsilon"))
    print("-" * 100)
    for i in range(n_flies):
        print("| {:<6} | {:<20.6f} | {:<12.6f} | {:<12.6f} | {:<12.6f} |".format(
            i+1, current_mape[i], flies[i][0], flies[i][1], flies[i][2]))
    print("-" * 100)
    
    # ==============================================
    # Tabel 5: Ringkasan Konvergensi
    # ==============================================
    print("\n\n" + "="*50)
    print("TABEL 5: RINGKASAN KONVERGENSI")
    print("="*50)
    print("-" * 80)
    print("| {:<15} | {:<30} |".format("Metric", "Value"))
    print("-" * 80)
    print("| {:<15} | {:<30} |".format("Total Iterasi", f"{iteration}/{n_iterations}"))
    print("| {:<15} | {:<30.6f}% |".format("MAPE Awal", convergence_history[0]))
    print("| {:<15} | {:<30.6f}% |".format("MAPE Akhir", best_mape))
    print("| {:<15} | {:<30.6f}% |".format("Perbaikan", convergence_history[0] - best_mape))
    print("| {:<15} | {:<30} |".format("Status", "Konvergen" if stall_counter >= max_stall_iterations else "Maksimum Iterasi"))
    print("| {:<15} | {:<30.6f} |".format("Best C", best_fly[0]))
    print("| {:<15} | {:<30.6f} |".format("Best gamma", best_fly[1]))
    print("| {:<15} | {:<30.6f} |".format("Best epsilon", best_fly[2]))
    print("-" * 80)
    
    # ==============================================
    # Visualisasi Konvergensi
    # ==============================================
    plt.figure(figsize=(15, 10))
    
    # Plot MAPE convergence
    plt.subplot(2, 2, 1)
    plt.plot(convergence_history, 'b-o')
    plt.title('Konvergensi MAPE')
    plt.xlabel('Iterasi')
    plt.ylabel('MAPE (%)')
    plt.grid(True)
    
    # Plot parameter C
    plt.subplot(2, 2, 2)
    plt.plot([x['best_fly'][0] for x in all_iterations_data], 'g-o')
    plt.title('Konvergensi Parameter C')
    plt.xlabel('Iterasi')
    plt.ylabel('Nilai C')
    plt.grid(True)
    
    # Plot parameter gamma
    plt.subplot(2, 2, 3)
    plt.plot([x['best_fly'][1] for x in all_iterations_data], 'r-o')
    plt.title('Konvergensi Parameter gamma')
    plt.xlabel('Iterasi')
    plt.ylabel('Nilai gamma')
    plt.grid(True)
    
    # Plot parameter epsilon
    plt.subplot(2, 2, 4)
    plt.plot([x['best_fly'][2] for x in all_iterations_data], 'm-o')
    plt.title('Konvergensi Parameter epsilon')
    plt.xlabel('Iterasi')
    plt.ylabel('Nilai epsilon')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # ==============================================
    # Simpan Data ke Excel
    # ==============================================
    # Buat DataFrame untuk semua iterasi
    convergence_df = pd.DataFrame({
        'Iterasi': [x['iteration'] for x in all_iterations_data],
        'Best_MAPE': [x['best_mape'] for x in all_iterations_data],
        'Best_C': [x['best_fly'][0] for x in all_iterations_data],
        'Best_gamma': [x['best_fly'][1] for x in all_iterations_data],
        'Best_epsilon': [x['best_fly'][2] for x in all_iterations_data]
    })
    
    # Buat DataFrame detail untuk semua lalat
    detailed_data = []
    for iter_data in all_iterations_data:
        for fly_idx in range(n_flies):
            detailed_data.append({
                'Iterasi': iter_data['iteration'],
                'Lalat': fly_idx + 1,
                'C': iter_data['flies'][fly_idx][0],
                'gamma': iter_data['flies'][fly_idx][1],
                'epsilon': iter_data['flies'][fly_idx][2],
                'MAPE': iter_data['mape_values'][fly_idx],
                'Status': 'Best' if np.array_equal(iter_data['flies'][fly_idx], iter_data['best_fly']) else ''
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # Simpan ke file Excel
    with pd.ExcelWriter('C:/Users/HP PAVILION/Downloads/hasil_optimasi_detail.xlsx') as writer:
        convergence_df.to_excel(writer, sheet_name='Konvergensi', index=False)
        detailed_df.to_excel(writer, sheet_name='Detail_Lalat', index=False)
    
    print("\nData hasil optimasi telah disimpan ke 'hasil_optimasi_detail.xlsx'")
    
    return best_fly, best_mape

# Define parameter bounds
lb = [0.1, 0.0001, 0.0001]
ub = [1000, 1, 1]

# Run optimization
best_params, best_mape = foa(
    n_iterations=100,
    n_flies=30,
    lb=lb,
    ub=ub,
    tolerance=0.0001,
    max_stall_iterations=10
)

# Train final model
best_model = SVR(
    kernel='rbf',
    C=best_params[0],
    gamma=best_params[1],
    epsilon=best_params[2]
)
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred_test = best_model.predict(X_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

# ==============================================
# Hasil Evaluasi Akhir
# ==============================================
print("\n" + "="*50)
print("HASIL EVALUASI MODEL AKHIR")
print("="*50)
print(f"Parameter Terbaik:")
print(f"  C     : {best_params[0]:.6f}")
print(f"  gamma : {best_params[1]:.6f}")
print(f"  epsilon: {best_params[2]:.6f}")
print(f"MAPE pada Data Test: {mape_test:.6f}%")
print("="*50)

# Plot hasil prediksi
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Aktual', marker='o')
plt.plot(dates_test, y_pred_test, label='Prediksi', marker='x')
plt.title('Perbandingan Nilai Aktual dan Prediksi')
plt.xlabel('Tanggal')
plt.ylabel('Nilai')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Simpan hasil prediksi
results_df = pd.DataFrame({
    'Tanggal': dates_test,
    'Aktual': y_test,
    'Prediksi': y_pred_test,
    'Error': y_test - y_pred_test
})
results_df.to_excel('C:/Users/HP PAVILION/Downloads/hasil_prediksi_final.xlsx', index=False)
print("\nHasil prediksi telah disimpan ke 'hasil_prediksi_final.xlsx'")

# In[ ]:




