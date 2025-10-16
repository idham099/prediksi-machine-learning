import numpy as np
from scipy.stats import norm
import os

# Define target columns consistent with 1.lstm.py and 2.xgboost.py
target_columns = ['Pesawat_DTG', 'Pesawat_BRK', 'Penumpang_DTG', 'Penumpang_BRK',
                  'Bagasi_DTG', 'Bagasi_BRK', 'Cargo_DTG', 'Cargo_BRK', 'Pos_DTG', 'Pos_BRK']

def diebold_mariano_test(e1, e2, h=1, alternative='two-sided'):
    """
    Melakukan Uji Diebold-Mariano (DM) antara dua deret error.
    e1 dan e2: array-like, residual atau error prediksi dari dua model (harus 1D array)
    h: horizon prediksi (1 untuk one-step ahead forecast)
    alternative: 'two-sided', 'less', 'greater'
    """
    e1 = np.array(e1)
    e2 = np.array(e2)

    if e1.ndim > 1 or e2.ndim > 1:
        raise ValueError("e1 and e2 must be 1D arrays for Diebold-Mariano test when passed to function.")

    d = (e1 ** 2) - (e2 ** 2)
    d_mean = np.mean(d)

    gamma = [np.cov(d[:-lag], d[lag:])[0, 1] if lag != 0 else np.var(d, ddof=1) for lag in range(h)]
    V_d = gamma[0] + 2 * sum(gamma[1:])

    if V_d <= 0:
        print(f"Warning: Variance V_d is non-positive ({V_d}). DM statistic cannot be computed or is infinite.")
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(V_d / len(d))

    if alternative == 'two-sided':
        p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    elif alternative == 'less':
        p_value = norm.cdf(dm_stat)
    elif alternative == 'greater':
        p_value = 1 - norm.cdf(dm_stat)
    else:
        raise ValueError("alternative harus salah satu dari: 'two-sided', 'less', atau 'greater'.")

    return dm_stat, p_value

# -------------------------
# KODE UTAMA
# -------------------------

# Direktori utama tempat file .npy dari LSTM dan y_test disimpan
base_npy_dir = '/content/drive/MyDrive/prediksi/AiTesis/'

# Direktori tempat file .npy dari XGBoost disimpan
xgb_npy_dir = os.path.join(base_npy_dir, '2.XGBoost', 'grafik_prediksi_ramalan', 'npy_predictions')

# Memuat data y_test dan y_pred_lstm (diasumsikan masih dalam satu file 2D)
try:
    y_test_full = np.load(os.path.join(base_npy_dir, 'y_test.npy'))
    y_pred_lstm_full = np.load(os.path.join(base_npy_dir, 'y_pred_lstm.npy'))
except FileNotFoundError as e:
    print(f"Error: {e}. Pastikan Anda telah menjalankan 1.lstm.py dan file 'y_test.npy' dan 'y_pred_lstm.npy' ada di '{base_npy_dir}'.")
    exit()

print("\n--- Hasil Uji Diebold-Mariano per Kolom Target ---")

# Iterasi melalui setiap kolom target
for i, col_name in enumerate(target_columns):
    # Ambil data aktual dan prediksi LSTM untuk kolom saat ini (array 1D)
    y_test_col = y_test_full[:, i]
    y_pred_lstm_col = y_pred_lstm_full[:, i]

    # Muat prediksi XGBoost untuk kolom saat ini dari file terpisah (ini akan menjadi array 1D)
    xgb_pred_filename = f"y_pred_xgb_{col_name}.npy"
    xgb_pred_filepath = os.path.join(xgb_npy_dir, xgb_pred_filename)

    try:
        y_pred_xgb_col = np.load(xgb_pred_filepath)
    except FileNotFoundError:
        print(f"\nError: File prediksi XGBoost untuk '{col_name}' tidak ditemukan di '{xgb_pred_filepath}'.")
        print("Pastikan Anda sudah menjalankan 2.xgboost.py dengan benar.")
        continue # Lanjut ke kolom berikutnya jika file tidak ditemukan

    # --- PENANGANAN PERBEDAAN PANJANG ARRAY (SOLUSI SEMENTARA) ---
    # Temukan panjang minimum dari ketiga array untuk kolom saat ini
    min_len = min(len(y_test_col), len(y_pred_lstm_col), len(y_pred_xgb_col))

    # Potong array yang lebih panjang agar sesuai dengan panjang minimum
    if len(y_test_col) != min_len:
        print(f"  Warning: Panjang y_test_col ({len(y_test_col)}) dipotong menjadi {min_len} untuk '{col_name}'.")
        y_test_col = y_test_col[:min_len]
    if len(y_pred_lstm_col) != min_len:
        print(f"  Warning: Panjang y_pred_lstm_col ({len(y_pred_lstm_col)}) dipotong menjadi {min_len} untuk '{col_name}'.")
        y_pred_lstm_col = y_pred_lstm_col[:min_len]
    if len(y_pred_xgb_col) != min_len:
        print(f"  Warning: Panjang y_pred_xgb_col ({len(y_pred_xgb_col)}) dipotong menjadi {min_len} untuk '{col_name}'.")
        y_pred_xgb_col = y_pred_xgb_col[:min_len]
    # --- AKHIR PENANGANAN PERBEDAAN PANJANG ARRAY ---

    # Hitung error untuk kolom spesifik ini
    errors_lstm_col = y_test_col - y_pred_lstm_col
    errors_xgb_col = y_test_col - y_pred_xgb_col

    print(f"\nKolom Target: {col_name}")
    try:
        # Lakukan uji DM untuk error kolom saat ini
        dm_stat, p_value = diebold_mariano_test(errors_lstm_col, errors_xgb_col)

        print(f"  DM Statistic: {dm_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")

        # Interpretasi hasil
        alpha = 0.05
        if p_value < alpha:
            print(f"  Karena p-value ({p_value:.4f}) < alpha ({alpha}), ada perbedaan signifikan dalam akurasi prediksi antara model LSTM dan XGBoost untuk {col_name}.")
            if dm_stat > 0:
                print(f"  Model XGBoost (e2) lebih akurat daripada LSTM (e1) untuk {col_name}.")
            else:
                print(f"  Model LSTM (e1) lebih akurat daripada XGBoost (e2) untuk {col_name}.")
        else:
            print(f"  Karena p-value ({p_value:.4f}) >= alpha ({alpha}), tidak ada perbedaan signifikan dalam akurasi prediksi antara model LSTM dan XGBoost untuk {col_name}.")
    except ValueError as ve:
        print(f"  Error saat menjalankan DM test untuk {col_name}: {ve}")
    except Exception as e:
        print(f"  Terjadi error tak terduga untuk {col_name}: {e}")

print("\n----------------------------------------------------")
print("Uji Diebold-Mariano selesai.")