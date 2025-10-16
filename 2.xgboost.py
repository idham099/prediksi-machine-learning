import pandas as pd
import shutil
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import sklearn
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

print(f"XGBoost version: {xgb.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# --- Global Variables & Initializations ---
# Definisikan kolom waktu
all_time_features = ['Bulan', 'Hari_dalam_Minggu', 'Hari_dalam_Bulan', 'Hari_dalam_Tahun',
                     'Minggu_dalam_Tahun', 'Kuartal', 'Tahun']

# Definisikan kolom target
target_columns = ['Pesawat_DTG', 'Pesawat_BRK', 'Penumpang_DTG', 'Penumpang_BRK',
                  'Bagasi_DTG', 'Bagasi_BRK', 'Cargo_DTG', 'Cargo_BRK', 'Pos_DTG', 'Pos_BRK']

# --- Helper Functions ---
def tampilkan_menu():
    """Menampilkan menu utama aplikasi."""
    print(" Prediksi XGBoost : ")
    print("+=========================================+")
    print("| 1. Copy ke data XGBoost                 |")
    print("| 2. Extreme Gradient Boosting (XGBoost)  |")
    print("| 3. Keluar Aplikasi                      |")
    print("+=========================================+")

def copy_data_file(source_path, destination_path):
    """Menyalin file data dari lokasi sumber ke lokasi tujuan."""
    print("+ INFO : Sedang mengkopi ke data XGBoost, Mohon menunggu..")
    print(f"Mencoba menyalin file dari:\n'{source_path}'")
    print(f"Ke:\n'{destination_path}'")
    try:
        # Memastikan direktori tujuan ada
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Cek jika data sumber ada dan terbaca
        with pd.ExcelFile(source_path) as xls:
            if 'Data Harian' not in xls.sheet_names:
                print(f"Peringatan: Sheet 'Data Harian' tidak ditemukan dalam file sumber '{source_path}'.")

        shutil.copy(source_path, destination_path)
        print(f"\nBerhasil menyalin file.")
        return True
    except FileNotFoundError:
        print(f"Error: File '{source_path}' tidak ditemukan. Pastikan path sumber sudah benar.")
    except Exception as e:
        print(f"Terjadi error saat menyalin file: {e}")
    return False

def preprocess_data(df, output_path):
    """
    Melakukan pra-pemrosesan data.
    """
    print("\n--- Tahap 1: Pengumpulan dan Pembersihan Data ---")

    if 'Tanggal' not in df.columns:
        print("Peringatan: Kolom 'Tanggal' tidak ditemukan. Pastikan nama kolom tanggal Anda benar.")
        return None, None, None, None

    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    initial_date_rows = df.shape[0]
    df.dropna(subset=['Tanggal'], inplace=True)
    if initial_date_rows - df.shape[0] > 0:
        print(f"- Dihapus {initial_date_rows - df.shape[0]} baris dengan tanggal tidak valid.")
    df.sort_values(by='Tanggal', inplace=True, ignore_index=True)
    print("- Kolom 'Tanggal' telah dikonversi ke format datetime dan diurutkan.")

    initial_rows_dup = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows_dup - df.shape[0]
    if removed_duplicates > 0:
        print(f"- {removed_duplicates} baris duplikat telah dihapus.")
    else:
        print("- Tidak ditemukan baris duplikat.")

    # --- Membuat Fitur Waktu ---
    print("\n--- Membuat Fitur Waktu dari Kolom 'Tanggal' ---")
    df['Bulan'] = df['Tanggal'].dt.month.astype(int)
    df['Hari_dalam_Minggu'] = df['Tanggal'].dt.dayofweek.astype(int)
    df['Hari_dalam_Bulan'] = df['Tanggal'].dt.day.astype(int)
    df['Hari_dalam_Tahun'] = df['Tanggal'].dt.dayofyear.astype(int)
    df['Minggu_dalam_Tahun'] = df['Tanggal'].dt.isocalendar().week.astype(int)
    df['Kuartal'] = df['Tanggal'].dt.quarter.astype(int)
    df['Tahun'] = df['Tanggal'].dt.year.astype(int)
    print("- Fitur waktu (Bulan, Hari_dalam_Minggu, Kuartal, dll.) telah dibuat.")

    relevant_numeric_cols = [col for col in target_columns if col in df.columns]
    if not relevant_numeric_cols:
        print("Error: Tidak ada kolom numerik relevan yang ditemukan untuk diproses.")
        return None, None, None, None

    print("\n--- Memulai Penanganan Inkonsistensi dan Kesalahan Entri ---")
    for col in relevant_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    print("- Kolom numerik relevan telah dikonversi ke numerik dan NaN diisi dengan 0.")

    for col in relevant_numeric_cols:
        if (df[col] < 0).sum() > 0:
            df[col] = df[col].clip(lower=0)
            print(f"   Nilai negatif di kolom '{col}' telah diubah menjadi 0.")

    # Rekayasa Fitur (Lagged & Rolling)
    print("\nc) Rekayasi Fitur (Feature Engineering - Lagged & Rolling)...")
    df_temp = df.copy() # Gunakan salinan untuk menghindari perubahan data asli
    for col in relevant_numeric_cols:
        for i in range(1, 4):
            df_temp[f'{col}_Lag_{i}'] = df_temp[col].shift(i).astype(float)
        df_temp[f'{col}_RollingMean_7d'] = df_temp[col].rolling(window=7, min_periods=1).mean().astype(float)
        df_temp[f'{col}_RollingSum_30d'] = df_temp[col].rolling(window=30, min_periods=1).sum().astype(float)
    df = df_temp
    print("     Fitur lagged dan agregat telah ditambahkan.")
    
    # Hapus baris yang memiliki NaN setelah rekayasa fitur
    initial_rows_after_features = df.shape[0]
    newly_engineered_cols = [c for c in df.columns if '_Lag_' in c or '_Rolling' in c]
    df.dropna(subset=newly_engineered_cols, inplace=True, how='any')
    if initial_rows_after_features - df.shape[0] > 0:
        print(f"     Dihapus {initial_rows_after_features - df.shape[0]} baris karena NaN dari fitur lagged/agregat.")
    
    print("\n--- Pra-pemrosesan Data Selesai ---")
    print(f"Ukuran data setelah pra-pemrosesan: {df.shape[0]} baris, {df.shape[1]} kolom.")
    
    # Simpan hasil pra-pemrosesan
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        df.to_excel(output_path, index=False, sheet_name='Data Processed')
        print(f"\nData yang telah diproses berhasil disimpan ke: '{output_path}'")
    except Exception as e:
        print(f"Error saat menyimpan data yang telah diproses ke Excel: {e}")

    return df, relevant_numeric_cols

def create_future_data(start_date, num_days, historical_df, ct_transformer, feature_cols, target_cols):
    """
    Membuat DataFrame untuk peramalan masa depan.
    """
    print(f"\n--- Membuat Data Masa Depan untuk {num_days} Hari ---")
    future_dates = pd.date_range(start=start_date + pd.Timedelta(days=1), periods=num_days, freq='D')
    future_df = pd.DataFrame({'Tanggal': future_dates})

    # Buat fitur waktu untuk data masa depan
    future_df['Bulan'] = future_df['Tanggal'].dt.month.astype(int)
    future_df['Hari_dalam_Minggu'] = future_df['Tanggal'].dt.dayofweek.astype(int)
    future_df['Hari_dalam_Bulan'] = future_df['Tanggal'].dt.day.astype(int)
    future_df['Hari_dalam_Tahun'] = future_df['Tanggal'].dt.dayofyear.astype(int)
    future_df['Minggu_dalam_Tahun'] = future_df['Tanggal'].dt.isocalendar().week.astype(int)
    future_df['Kuartal'] = future_df['Tanggal'].dt.quarter.astype(int)
    future_df['Tahun'] = future_df['Tanggal'].dt.year.astype(int)
    
    # Gabungkan dengan historical data untuk fitur lagged/rolling
    combined_df = pd.concat([historical_df, future_df], ignore_index=True)
    combined_df = combined_df.sort_values(by='Tanggal').reset_index(drop=True)
    
    # Isi kolom yang hilang dengan 0 untuk mencegah NaN saat komputasi fitur rekayasa
    combined_df[target_cols] = combined_df[target_cols].fillna(0)

    # Re-apply feature engineering pada data gabungan
    for col in target_columns:
        for i in range(1, 4):
            combined_df[f'{col}_Lag_{i}'] = combined_df[col].shift(i).astype(float)
        combined_df[f'{col}_RollingMean_7d'] = combined_df[col].rolling(window=7, min_periods=1).mean().astype(float)
        combined_df[f'{col}_RollingSum_30d'] = combined_df[col].rolling(window=30, min_periods=1).sum().astype(float)

    # Ambil data masa depan saja
    future_data_final = combined_df.iloc[len(historical_df):].copy()

    # Pastikan kolom fitur ada di data masa depan
    missing_cols_in_future = [col for col in feature_cols if col not in future_data_final.columns]
    for col in missing_cols_in_future:
        future_data_final[col] = 0

    # Urutkan kolom agar sesuai dengan urutan yang digunakan saat melatih model
    future_data_final = future_data_final[feature_cols]

    # Transformasi data masa depan menggunakan transformer yang sudah dilatih
    transformed_future_data = ct_transformer.transform(future_data_final)

    return transformed_future_data, future_dates

def plot_results(combined_test_results_df, final_forecast_df, plot_output_dir):
    """
    Membuat plot hasil prediksi vs aktual dan peramalan masa depan.
    """
    print("\n--- Membuat Grafik Hasil Prediksi dan Peramalan ---")
    os.makedirs(plot_output_dir, exist_ok=True)

    for target_col in target_columns:
        fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=False)
        fig.suptitle(f'Hasil Prediksi dan Peramalan untuk {target_col}', fontsize=16)

        # Plot 1: Test Set Predictions vs Actuals
        ax1 = axes[0]
        if f'Aktual_{target_col}' in combined_test_results_df.columns:
            sns.lineplot(x='Tanggal', y=f'Aktual_{target_col}', data=combined_test_results_df, label='Aktual', ax=ax1, color='blue')
            sns.lineplot(x='Tanggal', y=f'Prediksi_{target_col}', data=combined_test_results_df, label='Prediksi', ax=ax1, color='red', linestyle='--')
            ax1.set_title(f'Prediksi vs Aktual pada Data Test Set ({target_col})')
            ax1.set_xlabel('Tanggal')
            ax1.set_ylabel(target_col)
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
        else:
            ax1.text(0.5, 0.5, 'Data Prediksi/Aktual Tidak Tersedia', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12, color='gray')
            ax1.set_title(f'Prediksi vs Aktual pada Data Test Set ({target_col})')

        # Plot 2: Future Forecasts
        ax2 = axes[1]
        if f'Prediksi_{target_col}' in final_forecast_df.columns:
            sns.lineplot(x='Tanggal', y=f'Prediksi_{target_col}', data=final_forecast_df, label='Ramalan 2025', ax=ax2, color='green')
            ax2.set_title(f'Ramalan untuk Tahun 2025 ({target_col})')
            ax2.set_xlabel('Tanggal')
            ax2.set_ylabel(target_col)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, 'Data Ramalan Tidak Tersedia', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=12, color='gray')
            ax2.set_title(f'Ramalan untuk Tahun 2025 ({target_col})')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plot_filename = os.path.join(plot_output_dir, f'Prediksi_Ramalan_{target_col}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   Grafik untuk '{target_col}' disimpan ke: {plot_filename}")
    print("--- Pembuatan Grafik Selesai ---")

def mean_absolute_percentage_error(y_true, y_pred):
    """Menghitung Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Menghindari pembagian dengan nol
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_predict_and_forecast_xgboost(df, prediction_output_path, plot_output_dir):
    """
    Melatih dan mengevaluasi model XGBoost untuk setiap kolom target, dan melakukan prediksi forecast.
    """
    print("\n--- Memulai Proses XGBoost: Pelatihan, Prediksi, dan Peramalan ---")

    if df is None or df.empty:
        print("Error: DataFrame yang diproses kosong atau tidak valid. Tidak dapat melanjutkan pelatihan.")
        return

    # Pembagian Data Berdasarkan Waktu
    print("\n--- Pembagian Data (Training, Validation, Test Set) berdasarkan Tanggal ---")
    train_end_date = pd.to_datetime('2023-10-31')
    val_end_date = pd.to_datetime('2024-05-31')
    
    df_train = df[df['Tanggal'] <= train_end_date]
    df_val = df[(df['Tanggal'] > train_end_date) & (df['Tanggal'] <= val_end_date)]
    df_test = df[df['Tanggal'] > val_end_date]
    
    if df_train.empty or df_val.empty or df_test.empty:
        print("Peringatan: Satu atau lebih set data (train/val/test) kosong. Periksa rentang tanggal atau data input.")
        return

    print(f"Training Set: Dari {df_train['Tanggal'].min().strftime('%Y-%m-%d')} sampai {df_train['Tanggal'].max().strftime('%Y-%m-%d')} ({len(df_train)} baris)")
    print(f"Validation Set: Dari {df_val['Tanggal'].min().strftime('%Y-%m-%d')} sampai {df_val['Tanggal'].max().strftime('%Y-%m-%d')} ({len(df_val)} baris)")
    print(f"Test Set: Dari {df_test['Tanggal'].min().strftime('%Y-%m-%d')} sampai {df_test['Tanggal'].max().strftime('%Y-%m-%d')} ({len(df_test)} baris)")
    
    # Pisahkan fitur dan target
    features = [col for col in df.columns if col not in ['Tanggal'] + target_columns]
    X_train = df_train[features]
    y_train = df_train[target_columns]
    X_val = df_val[features]
    y_val = df_val[target_columns]
    X_test = df_test[features]
    y_test = df_test[target_columns]
    
    # Buat dan latih transformer di luar loop
    categorical_features_for_ohe = ['Bulan', 'Hari_dalam_Minggu', 'Kuartal']
    categorical_features_to_encode = [col for col in categorical_features_for_ohe if col in X_train.columns]
    
    features_to_scale = [col for col in X_train.columns if col not in categorical_features_to_encode]
    
    # Gunakan ColumnTransformer untuk preprocessing yang lebih aman
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features_to_encode),
            ('scaler', StandardScaler(), features_to_scale)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    print("Melatih preprocessor pada data training...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Ambil nama fitur yang telah diubah
    feature_names = preprocessor.get_feature_names_out()

    combined_test_results_df = pd.DataFrame()
    all_feature_importances = pd.DataFrame()
    evaluation_metrics = pd.DataFrame(columns=['Kolom Target', 'MAE', 'MSE', 'RMSE', 'R2', 'MAPE'])
    
    y_test_all_global = []
    y_pred_all_global = []
    
    for col in target_columns:
        print(f"\nMelatih model untuk kolom target: {col}")
        
        # Hyperparameter Tuning
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                     eval_metric='mae', 
                                     random_state=42, 
                                     n_jobs=-1,
                                     tree_method='hist')
        
        param_grid = {
            'n_estimators': [100, 200, 500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
            'reg_lambda': [1, 1.5, 2, 2.5]
        }
        
        random_search = RandomizedSearchCV(estimator=xgb_model,
                                            param_distributions=param_grid,
                                            n_iter=20, # Mengurangi iterasi untuk efisiensi
                                            scoring='neg_mean_absolute_error',
                                            cv=3,
                                            verbose=1,
                                            random_state=42,
                                            n_jobs=-1)
        
        random_search.fit(X_train_processed, y_train[col])
        best_params = random_search.best_params_
        print("Hyperparameter terbaik ditemukan:")
        print(best_params)

        # Latih model final dengan parameter terbaik
        final_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                        eval_metric='mae', 
                                        random_state=42, 
                                        n_jobs=-1,
                                        tree_method='hist',
                                        **best_params)
        
        final_model.fit(X_train_processed, y_train[col], 
                        eval_set=[(X_val_processed, y_val[col])], 
                        verbose=False)
        
        # Prediksi pada test set
        y_pred = final_model.predict(X_test_processed)
        
        y_test_all_global.append(y_test[col])
        y_pred_all_global.append(y_pred)

        # --- Bagian untuk menyimpan prediksi ke file .npy ---
        npy_output_dir = os.path.join(plot_output_dir, 'npy_predictions')
        os.makedirs(npy_output_dir, exist_ok=True)
        npy_filename = os.path.join(npy_output_dir, f'y_pred_xgb_{col}.npy')
        np.save(npy_filename, y_pred)
        print(f"   Prediksi untuk '{col}' berhasil disimpan ke: '{npy_filename}'")
        # --- Akhir bagian yang ditambahkan ---
        
        # Hitung metrik evaluasi
        mae = mean_absolute_error(y_test[col], y_pred)
        mse = mean_squared_error(y_test[col], y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test[col], y_pred)
        mape = mean_absolute_percentage_error(y_test[col], y_pred)
        
        new_row = pd.DataFrame([{'Kolom Target': col, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}])
        evaluation_metrics = pd.concat([evaluation_metrics, new_row], ignore_index=True)
        print(f"Metrik untuk {col}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}, MAPE={mape:.2f}%")

        # Simpan hasil prediksi test
        test_results = pd.DataFrame({
            'Tanggal': df_test['Tanggal'],
            f'Aktual_{col}': y_test[col],
            f'Prediksi_{col}': y_pred
        })
        if combined_test_results_df.empty:
            combined_test_results_df = test_results
        else:
            combined_test_results_df = pd.merge(combined_test_results_df, test_results, on='Tanggal', how='outer')

        # Simpan feature importances
        importances = final_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        feature_importance_df.insert(0, 'Kolom Target', col)
        all_feature_importances = pd.concat([all_feature_importances, feature_importance_df], ignore_index=True)
    
    # Lakukan peramalan untuk masa depan
    last_historical_date = df['Tanggal'].max()
    future_forecast_df = pd.DataFrame()
    for col in target_columns:
        print(f"\nMelakukan peramalan untuk {col}...")
        
        # Latih model final pada seluruh data historis
        X_all_processed = preprocessor.transform(df[features])
        final_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                        eval_metric='mae', 
                                        random_state=42, 
                                        n_jobs=-1,
                                        tree_method='hist',
                                        **best_params)
        final_model.fit(X_all_processed, df[col])
        
        # Buat data masa depan
        future_data, future_dates = create_future_data(last_historical_date, 365, df, preprocessor, features, target_columns)
        
        # Prediksi masa depan
        future_predictions = final_model.predict(future_data)
        
        forecast_results = pd.DataFrame({
            'Tanggal': future_dates,
            f'Prediksi_{col}': future_predictions
        })
        if future_forecast_df.empty:
            future_forecast_df = forecast_results
        else:
            future_forecast_df = pd.merge(future_forecast_df, forecast_results, on='Tanggal', how='outer')

    # Hitung metrik keseluruhan
    y_test_all_global_flat = np.concatenate(y_test_all_global)
    y_pred_all_global_flat = np.concatenate(y_pred_all_global)
    mae_global = mean_absolute_error(y_test_all_global_flat, y_pred_all_global_flat)
    mse_global = mean_squared_error(y_test_all_global_flat, y_pred_all_global_flat)
    rmse_global = np.sqrt(mse_global)
    r2_global = r2_score(y_test_all_global_flat, y_pred_all_global_flat)
    mape_global = mean_absolute_percentage_error(y_test_all_global_flat, y_pred_all_global_flat)
    metrics = {
        'MAE_Keseluruhan': mae_global,
        'MSE_Keseluruhan': mse_global,
        'RMSE_Keseluruhan': rmse_global,
        'R2_Keseluruhan': r2_global,
        'MAPE_Keseluruhan': mape_global
    }
    overall_metrics_df = pd.DataFrame([metrics])
    
    # Pastikan combined_test_results_df diurutkan berdasarkan tanggal
    combined_test_results_df['Tanggal'] = pd.to_datetime(combined_test_results_df['Tanggal'])
    combined_test_results_df = combined_test_results_df.sort_values(by='Tanggal').reset_index(drop=True)
    
    # Simpan plot
    plot_results(combined_test_results_df, future_forecast_df, plot_output_dir)

    # Simpan semua hasil ke dalam satu file Excel
    try:
        output_excel_path = os.path.join(prediction_output_path, 'hasil_prediksi_dan_metrik_xgboost.xlsx')
        print(f"Menyimpan hasil prediksi dan metrik ke {output_excel_path}")
        with pd.ExcelWriter(output_excel_path) as writer:
            overall_metrics_df.to_excel(writer, sheet_name='Evaluasi_Metrik_Keseluruhan', index=False)
            evaluation_metrics.to_excel(writer, sheet_name='Evaluasi_Metrik_Per_Kolom', index=False)
            combined_test_results_df.to_excel(writer, sheet_name='Hasil_Prediksi_Test', index=False)
            future_forecast_df.to_excel(writer, sheet_name='Hasil_Prediksi_Forecast', index=False)
            all_feature_importances.to_excel(writer, sheet_name='Feature_Importances', index=False)
        print("Penyimpanan data ke Excel selesai!")
    except Exception as e:
        print(f"Terjadi error saat mencoba menyimpan ke Excel: {e}")
        traceback.print_exc()

def main():
    """Fungsi utama untuk menjalankan aplikasi."""
    source_data_file_path = r'/content/drive/MyDrive/prediksi/AiTesis/0.Data/data_time_series_2019_2024.xlsx'
    xgboost_data_folder = r'/content/drive/MyDrive/prediksi/AiTesis/2.XGBoost'
    processed_data_file_path = r'/content/drive/MyDrive/prediksi/AiTesis/0.Data/data_lstm_2019_2024.xlsx'
    prediction_output_folder = r'/content/drive/MyDrive/prediksi/AiTesis/2.XGBoost' # Diubah menjadi folder
    plot_output_folder = os.path.join(xgboost_data_folder, 'grafik_prediksi_ramalan')

    while True:
        tampilkan_menu()
        pilihan = input("Masukkan pilihan Anda (1/2/3): ")

        if pilihan == '1':
            print("\n--- Memulai Proses Menyalin Data ---")
            if copy_data_file(source_data_file_path, processed_data_file_path):
                print("+ INFO : Penyalinan selesai. Anda dapat melanjutkan ke langkah 2.")
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == '2':
            print("\n--- Memulai Proses Pelatihan dan Prediksi XGBoost ---")
            try:
                # Perbaikan di sini: Menggunakan sheet_name='Data Processed'
                df_raw = pd.read_excel(processed_data_file_path, sheet_name='Data Harian')
                df_processed, relevant_numeric_cols = preprocess_data(df_raw, processed_data_file_path)

                if df_processed is not None and not df_processed.empty:
                    train_predict_and_forecast_xgboost(df_processed, prediction_output_folder, plot_output_folder)
                else:
                    print("Pra-pemrosesan data gagal (DataFrame kosong atau tidak valid). Model tidak dapat dilatih.")

            except FileNotFoundError:
                print(f"Error: File '{processed_data_file_path}' tidak ditemukan. Pastikan Anda sudah menyalin data di langkah 1.")
            except Exception as e:
                print(f"Terjadi error tidak terduga selama proses: {e}")
                traceback.print_exc()

            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == '3':
            print("+ INFO : Anda memilih keluar dari aplikasi. Sampai jumpa!")
            break

        else:
            print("+ INFO : Pilihan tidak valid, silahkan masukkan 1, 2, atau 3. ")
            input("\nTekan Enter untuk melanjutkan...")

if __name__ == "__main__":
    main()