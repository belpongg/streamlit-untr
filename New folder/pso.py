#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
from mpl_toolkits.mplot3d import Axes3D
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
import datetime

# 1. SETUP DAN PREPROSES DATA ==============================================

np.random.seed(42)

def muat_data(filepath):
    try:
        data = pd.read_excel(filepath)
        data = data.dropna()
        data = data.sort_values(by='Date')
        print("Dataset berhasil dimuat, bentuk:", data.shape)
        return data
    except Exception as e:
        print(f"Gagal memuat dataset: {e}")
        exit()

file_path = 'C:/Users/HP PAVILION/Downloads/utut.xlsx'
data = muat_data(file_path)

X = data[['lag1']].values  
y = data['y'].values

def pisah_data(X, y, dates):
    X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
        X, y, dates, test_size=0.4, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test, dates_val, dates_test = train_test_split(
        X_temp, y_temp, dates_temp, test_size=0.5, shuffle=False, random_state=42)
    return (X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test)

(X_train, X_val, X_test, 
 y_train, y_val, y_test,
 dates_train, dates_val, dates_test) = pisah_data(X, y, data['Date'])

# 2. IMPLEMENTASI PSO-SVR =================================================

def pso_svr(n_iterasi, n_partikel, batas_bawah, batas_atas, toleransi=1e-4, maks_iterasi_tanpa_perbaikan=10):
    
    dimensi = len(batas_bawah)
    lb = np.array(batas_bawah)
    ub = np.array(batas_atas)
    
    # Parameter PSO
    w_max = 0.9
    w_min = 0.2
    c1_awal = 2.5
    c1_akhir = 1.5
    c2_awal = 1.5  
    c2_akhir = 2.5
    v_maks = (ub - lb) * 0.3
    
    # Inisialisasi partikel
    partikel = np.random.uniform(lb, ub, (n_partikel, dimensi))
    kecepatan = np.random.uniform(-v_maks, v_maks, (n_partikel, dimensi))
    
    # Inisialisasi best
    pbest_posisi = partikel.copy()
    pbest_skor = np.full(n_partikel, np.inf)
    gbest_posisi = partikel[0].copy()
    gbest_skor = np.inf
    
    # Penyimpanan history
    history = {
        'gbest_skor': [],
        'gbest_params': [],
        'semua_partikel': [],
        'semua_kecepatan': []
    }

    # 2.1 INISIALISASI ====================================================
    
    def print_pemisah(lebar=80):
        print("=" * lebar)
    
    def print_tabel_partikel(partikel, kecepatan, iterasi=None):
        if iterasi is not None:
            print(f"\nITERASI {iterasi} - POSISI DAN KECEPATAN PARTIKEL")
        print_pemisah()
        print("| Part | C              | gamma          | epsilon        | Kecepatan C    | Kecepatan gamma| Kecepatan eps  |")
        print_pemisah()
        for i in range(len(partikel)):
            print(f"| {i+1:4} | {partikel[i][0]:<14.6f} | {partikel[i][1]:<14.6f} | {partikel[i][2]:<14.6f} | "
                  f"{kecepatan[i][0]:<14.6f} | {kecepatan[i][1]:<14.6f} | {kecepatan[i][2]:<14.6f} |")
        print_pemisah()
    
    print("\nINISIALISASI")
    print_tabel_partikel(partikel, kecepatan, "Awal")
    
    print("\nEVALUASI FITNESS AWAL")
    print_pemisah()
    print("| Part | MAPE (%)       |")
    print_pemisah()
    
    for i in range(n_partikel):
        model = SVR(kernel='rbf', C=partikel[i][0], 
                   gamma=partikel[i][1], epsilon=partikel[i][2])
        model.fit(X_train, y_train)
        mape = mean_absolute_percentage_error(y_val, model.predict(X_val)) * 100
        pbest_skor[i] = mape
        
        if mape < gbest_skor:
            gbest_skor = mape
            gbest_posisi = partikel[i].copy()
        
        print(f"| {i+1:4} | {mape:<14.6f} |")
    
    print_pemisah()
    
    history['gbest_skor'].append(gbest_skor)
    history['gbest_params'].append(gbest_posisi.copy())
    history['semua_partikel'].append(partikel.copy())
    history['semua_kecepatan'].append(kecepatan.copy())
    
    # 2.2 ITERASI UTAMA ===================================================
    
    print("\nMEMULAI OPTIMASI")
    penghitung_stagnan = 0
    
    for iterasi in range(n_iterasi):
        # Parameter adaptif
        w = w_max - (w_max - w_min) * iterasi / n_iterasi
        c1 = c1_awal - (c1_awal - c1_akhir) * iterasi / n_iterasi
        c2 = c2_awal + (c2_akhir - c2_awal) * iterasi / n_iterasi
        
        skor_sebelumnya = gbest_skor
        
        # Update setiap partikel
        for i in range(n_partikel):
            # Update kecepatan
            r1, r2 = np.random.rand(dimensi), np.random.rand(dimensi)
            kognitif = c1 * r1 * (pbest_posisi[i] - partikel[i])
            sosial = c2 * r2 * (gbest_posisi - partikel[i])
            kecepatan[i] = w * kecepatan[i] + kognitif + sosial
            kecepatan[i] = np.clip(kecepatan[i], -v_maks, v_maks)
            
            # Update posisi
            partikel[i] += kecepatan[i]
            partikel[i] = np.clip(partikel[i], lb, ub)
            
            # Evaluasi
            model = SVR(kernel='rbf', C=partikel[i][0],
                       gamma=partikel[i][1], epsilon=partikel[i][2])
            model.fit(X_train, y_train)
            mape = mean_absolute_percentage_error(y_val, model.predict(X_val)) * 100
            
            # Update personal best
            if mape < pbest_skor[i]:
                pbest_skor[i] = mape
                pbest_posisi[i] = partikel[i].copy()
            
            # Update global best
            if mape < gbest_skor:
                gbest_skor = mape
                gbest_posisi = partikel[i].copy()
        
        # Simpan history
        history['gbest_skor'].append(gbest_skor)
        history['gbest_params'].append(gbest_posisi.copy())
        history['semua_partikel'].append(partikel.copy())
        history['semua_kecepatan'].append(kecepatan.copy())
        
        # Cek konvergensi
        perbaikan = skor_sebelumnya - gbest_skor
        if abs(perbaikan) < toleransi:
            penghitung_stagnan += 1
        else:
            penghitung_stagnan = 0
        
        konvergen = penghitung_stagnan >= maks_iterasi_tanpa_perbaikan
        
        # 2.3 TRACKING ITERASI ============================================
        
        if (iterasi < 10) or (iterasi % 5 == 0) or konvergen:
            print(f"\nITERASI {iterasi + 1}")
            print(f"MAPE Terbaik: {gbest_skor:.6f}%")
            print(f"Parameter: C={gbest_posisi[0]:.6f}, gamma={gbest_posisi[1]:.6f}, epsilon={gbest_posisi[2]:.6f}")
            print(f"Perbaikan: {perbaikan:.6f}%")
            print(f"Status: {'KONVERGEN' if konvergen else 'LANJUT'}")
            
            if iterasi < 10 or iterasi % 10 == 0 or konvergen:
                print_tabel_partikel(partikel, kecepatan, iterasi + 1)
        
        if konvergen:
            break
    
    # 2.4 HASIL AKHIR =====================================================
    
    print("\nOPTIMASI SELESAI")
    print(f"MAPE Terbaik: {gbest_skor:.6f}%")
    print(f"Parameter Optimal: C={gbest_posisi[0]:.6f}, gamma={gbest_posisi[1]:.6f}, epsilon={gbest_posisi[2]:.6f}")
    
    # Hitung perbaikan
    mape_awal = history['gbest_skor'][0]
    mape_akhir = gbest_skor
    perbaikan_total = mape_awal - mape_akhir
    
    # 2.5 TABEL RINGKASAN KONVERGENSI =====================================
    
    print("\n")
    print("=" * 50)
    print("TABEL 5: RINGKASAN KONVERGENSI")
    print("=" * 50)
    print("-" * 80)
    print("| {:<15} | {:<30} |".format("Metric", "Value"))
    print("-" * 80)
    print("| {:<15} | {:<30} |".format("Total Iterasi", str(iterasi + 1)))
    print("| {:<15} | {:<30.6f}% |".format("MAPE Awal", mape_awal))
    print("| {:<15} | {:<30.6f}% |".format("MAPE Akhir", mape_akhir))
    print("| {:<15} | {:<30.6f}% |".format("Perbaikan", perbaikan_total))
    print("| {:<15} | {:<30} |".format("Status", "Konvergen" if konvergen else "Tidak Konvergen"))
    print("| {:<15} | {:<30.6f} |".format("Best C", gbest_posisi[0]))
    print("| {:<15} | {:<30.6f} |".format("Best gamma", gbest_posisi[1]))
    print("| {:<15} | {:<30.6f} |".format("Best epsilon", gbest_posisi[2]))
    print("-" * 80)

    # 3. VISUALISASI ======================================================
    
    # 3.1 Plot Konvergensi
    plt.figure(figsize=(12, 6))
    plt.plot(history['gbest_skor'], 'b-o', linewidth=1, markersize=4)
    plt.title('Konvergensi PSO: MAPE Terbaik vs Iterasi')
    plt.xlabel('Iterasi')
    plt.ylabel('MAPE (%)')
    plt.grid(True)
    plt.show()
    
    # 3.2 Plot Parameter
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    ax1.plot([p[0] for p in history['gbest_params']], 'r-o', markersize=4)
    ax1.set_ylabel('Nilai C')
    ax1.set_title('Evolusi Parameter C')
    ax1.grid(True)
    
    ax2.plot([p[1] for p in history['gbest_params']], 'g-o', markersize=4)
    ax2.set_ylabel('Nilai Gamma')
    ax2.set_title('Evolusi Parameter Gamma')
    ax2.grid(True)
    
    ax3.plot([p[2] for p in history['gbest_params']], 'b-o', markersize=4)
    ax3.set_ylabel('Nilai Epsilon')
    ax3.set_xlabel('Iterasi')
    ax3.set_title('Evolusi Parameter Epsilon')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 3.3 Visualisasi 3D
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for p in range(n_partikel):
        ax.plot([h[p][0] for h in history['semua_partikel']],
                [h[p][1] for h in history['semua_partikel']],
                [h[p][2] for h in history['semua_partikel']],
                alpha=0.3, linewidth=0.5)
    
    ax.scatter(gbest_posisi[0], gbest_posisi[1], gbest_posisi[2],
              c='red', s=200, marker='*', label='Global Best')
    
    ax.set_xlabel('C')
    ax.set_ylabel('Gamma')
    ax.set_zlabel('Epsilon')
    ax.set_title('Trajektori Partikel dalam Ruang Parameter')
    ax.legend()
    plt.show()
    
    return gbest_posisi, gbest_skor, history

# 4. JALANKAN OPTIMASI ====================================================

batas_bawah = [0.1, 0.0001, 0.0001]
batas_atas = [1000, 1, 1]

print("\nMEMULAI OPTIMASI PSO")
best_params, best_mape, history = pso_svr(
    n_iterasi=100,
    n_partikel=30,
    batas_bawah=batas_bawah,
    batas_atas=batas_atas,
    toleransi=0.0001,
    maks_iterasi_tanpa_perbaikan=10
)

# 5. EVALUASI MODEL FINAL =================================================

model_final = SVR(kernel='rbf', C=best_params[0],
                 gamma=best_params[1], epsilon=best_params[2])
model_final.fit(X_train, y_train)

y_pred_test = model_final.predict(X_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

print("\nEVALUASI MODEL FINAL")
print(f"Test MAPE: {mape_test:.4f}%")

# 6. PERAMALAN ============================================================

class KalenderHariLibur(AbstractHolidayCalendar):
    rules = [
        Holiday('Tahun Baru', month=1, day=1),
        Holiday('Natal', month=12, day=25)
    ]

def buat_tanggal_ramalan(tanggal_terakhir, n_hari=15):
    kalender = KalenderHariLibur()
    libur = kalender.holidays(start=tanggal_terakhir, end=tanggal_terakhir + datetime.timedelta(days=365))
    
    tanggal_ramalan = []
    tanggal_sekarang = tanggal_terakhir
    
    while len(tanggal_ramalan) < n_hari:
        tanggal_sekarang += BDay(1)
        if tanggal_sekarang not in libur:
            tanggal_ramalan.append(tanggal_sekarang)
    
    return tanggal_ramalan

tanggal_terakhir = data['Date'].iloc[-1]
tanggal_ramalan = buat_tanggal_ramalan(tanggal_terakhir)

ramalan = []
lag_sekarang = y[-1]

for _ in range(15):
    prediksi = model_final.predict([[lag_sekarang]])[0]
    ramalan.append(prediksi)
    lag_sekarang = prediksi

df_ramalan = pd.DataFrame({
    'Tanggal': tanggal_ramalan,
    'Ramalan': ramalan
})

print("\nRAMALAN 15 HARI KE DEPAN")
print(df_ramalan.to_string(index=False))

# 7. SIMPAN HASIL =========================================================

# Simpan history konvergensi
df_history = pd.DataFrame({
    'iterasi': range(len(history['gbest_skor'])),
    'best_mape': history['gbest_skor'],
    'best_C': [p[0] for p in history['gbest_params']],
    'best_gamma': [p[1] for p in history['gbest_params']],
    'best_epsilon': [p[2] for p in history['gbest_params']]
})

df_history.to_excel('history_konvergensi_pso.xlsx', index=False)

# Simpan ramalan
df_ramalan.to_excel('ramalan_15_hari.xlsx', index=False)

print("\nHasil disimpan ke file:")
print("- history_konvergensi_pso.xlsx")
print("- ramalan_15_hari.xlsx")

# In[ ]:




