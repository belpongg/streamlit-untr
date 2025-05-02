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
st.title("Optimasi SVR dengan FOA dan PSO untuk Prediksi Saham")

# 1. Kalender Libur Khusus (Gabungan FOA dan PSO)
class TradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1),
        Holiday('Christmas', month=12, day=25),
        Holiday('New Year Eve', month=12, day=31)
    ]

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset saham (Excel)", type=["xlsx"])
if uploaded_file is not None:
    # Baca dan preprocess data
    data = pd.read_excel(uploaded_file)
    data = data.dropna()
    data = data.sort_values(by='Date')
    
    st.write("Preview Dataset:")
    st.dataframe(data.head())

    # Split data 60:20:20 (sesuai kode FOA dan PSO)
    X = data[['lag1']].values
    y = data['y'].values
    dates = data['Date'].values
    
    X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
        X, y, dates, test_size=0.4, shuffle=False, random_state=42)
    X_val, X_test, y_val, y_test, dates_val, dates_test = train_test_split(
        X_temp, y_temp, dates_temp, test_size=0.5, shuffle=False, random_state=42)

    # Pilih algoritma
    algorithm = st.radio("Pilih algoritma optimasi:", ["FOA", "PSO"], horizontal=True)

    if algorithm == "FOA":
        st.header("🪰 Fruit Fly Optimization Algorithm (FOA)")
        try:
            # Load model FOA
            model = joblib.load('foa_model.sav')
            
            # Evaluasi
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
            mape_val = mean_absolute_percentage_error(y_val, y_pred_val) * 100
            mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
            
            st.success(f"**Hasil FOA**\n\n"
                      f"- MAPE Validation: `{mape_val:.4f}%`\n"
                      f"- MAPE Testing: `{mape_test:.4f}%`\n"
                      f"- Parameter Terbaik:\n"
                      f"  - C: `{model.C:.6f}`\n"
                      f"  - gamma: `{model.gamma:.6f}`\n"
                      f"  - epsilon: `{model.epsilon:.6f}`")

            # Plot hasil
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(dates_test, y_test, 'b-', label='Aktual', marker='o')
            ax.plot(dates_test, y_pred_test, 'r--', label='Prediksi FOA', marker='x')
            ax.set_title('Perbandingan Aktual vs Prediksi (FOA)')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga Saham')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Gagal memuat model FOA. Pastikan file 'foa_model.sav' ada!\nError: {str(e)}")

    elif algorithm == "PSO":
        st.header("🐦 Particle Swarm Optimization (PSO)")
        try:
            # Load model PSO dengan parameter yang sesuai dengan implementasi asli
            model = joblib.load('pso_model.sav')
            
            # Evaluasi
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
            mape_val = mean_absolute_percentage_error(y_val, y_pred_val) * 100
            mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
            
            st.success(f"**Hasil PSO**\n\n"
                      f"- MAPE Validation: `{mape_val:.4f}%`\n"
                      f"- MAPE Testing: `{mape_test:.4f}%`\n"
                      f"- Parameter Terbaik:\n"
                      f"  - C: `{model.C:.6f}` (range: 0.1-1000)\n"
                      f"  - gamma: `{model.gamma:.6f}` (range: 0.0001-1)\n"
                      f"  - epsilon: `{model.epsilon:.6f}` (range: 0.0001-1)")
            
            # Plot konvergensi (jika tersedia dalam model)
            if hasattr(model, 'convergence_history_'):
                fig_conv, ax_conv = plt.subplots(figsize=(12,6))
                ax_conv.plot(model.convergence_history_['gbest_skor'], 'b-o', linewidth=1, markersize=4)
                ax_conv.set_title('Konvergensi PSO: MAPE Terbaik vs Iterasi')
                ax_conv.set_xlabel('Iterasi')
                ax_conv.set_ylabel('MAPE (%)')
                ax_conv.grid(True)
                st.pyplot(fig_conv)

            # Plot hasil prediksi
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(dates_test, y_test, 'g-', label='Aktual', marker='o')
            ax.plot(dates_test, y_pred_test, 'm--', label='Prediksi PSO', marker='x')
            ax.set_title('Perbandingan Aktual vs Prediksi (PSO)')
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga Saham')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Gagal memuat model PSO. Pastikan file 'pso_model.sav' ada!\nError: {str(e)}")

    # =============================================
    # FITUR PREDIKSI 15 HARI (SAMA UNTUK FOA/PSO)
    # =============================================
    st.header("🔮 Prediksi 15 Hari ke Depan")
    
    def generate_trading_dates(last_date, n_days=15):
        cal = TradingCalendar()
        holidays = cal.holidays(start=last_date, end=last_date + datetime.timedelta(days=60))
        trading_dates = []
        current_date = last_date
        
        while len(trading_dates) < n_days:
            current_date += BDay(1)
            if current_date.weekday() < 5 and current_date not in holidays:
                trading_dates.append(current_date)
        return trading_dates
    
    last_date = data['Date'].iloc[-1]
    future_dates = generate_trading_dates(last_date)
    
    # Prediksi
    current_lag = y[-1]
    predictions = []
    for _ in range(15):
        pred = model.predict([[current_lag]])[0]
        predictions.append(pred)
        current_lag = pred
    
    # Tampilkan hasil
    forecast_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Harga Prediksi': predictions
    })
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(forecast_df.style.format({'Harga Prediksi': '{:.2f}'}))
    
    with col2:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(data['Date'][-30:], data['y'][-30:], 'b-', label='Historis')
        ax.plot(forecast_df['Tanggal'], forecast_df['Harga Prediksi'], 'ro--', label='Prediksi')
        ax.set_title('Trend Prediksi 15 Hari')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # =============================================
    # TABEL PERBANDINGAN
    # =============================================
    st.header("📊 Tabel Perbandingan")
    comparison_df = pd.DataFrame({
        'Tanggal': dates_test,
        'Aktual': y_test,
        'Prediksi': y_pred_test,
        'Selisih': y_test - y_pred_test
    })
    st.dataframe(comparison_df.style.format({
        'Aktual': '{:.2f}',
        'Prediksi': '{:.2f}',
        'Selisih': '{:.2f}'
    }))

# Petunjuk penggunaan
with st.expander("ℹ️ Petunjuk Penggunaan"):
    st.write("""
    1. Upload dataset Excel dengan kolom: `Date`, `lag1`, dan `y`
    2. Pilih algoritma optimasi (FOA atau PSO)
    3. Pastikan file model (`foa_model.sav`/`pso_model.sav`) ada di folder yang sama
    4. Hasil akan menampilkan:
       - Metrik evaluasi (MAPE)
       - Parameter terbaik
       - Grafik prediksi
       - Prediksi 15 hari ke depan
    """)
