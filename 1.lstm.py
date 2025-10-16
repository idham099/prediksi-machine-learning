import pandas as pd
import shutil
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt 

y_true_all_global = []
y_pred_all_global = []

# Fungsi untuk membuat urutan (sequences) untuk LSTM
def create_sequences(data, n_steps, target_column_indices):
    X, y = [], []
    for i in range(len(data) - n_steps):
        # X: n_steps data historis untuk SEMUA fitur
        X.append(data[i:(i + n_steps), :])

        # y: Nilai dari SEMUA KOLOM TARGET pada langkah waktu berikutnya (i + n_steps)
        # memilih kolom-kolom target menggunakan target_column_indices
        y.append(data[i + n_steps, target_column_indices])
    return np.array(X), np.array(y)

def build_and_train_model_lstm(X_train, y_train, X_val, y_val,
                                 n_steps, n_features, n_outputs,
                                 lstm_units, dropout_rate, learning_rate, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(units=int(lstm_units), activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(dropout_rate)) # Tambahkan dropout
    model.add(LSTM(units=int(lstm_units), activation='tanh', return_sequences=False))
    model.add(Dropout(dropout_rate)) # Tambahkan dropout
    model.add(Dense(units=25, activation='tanh'))
    model.add(Dense(units=int(n_outputs), activation='linear'))


    # 2. Kompilasi Model
    # Perhatikan: untuk menyesuaikan learning_rate, Anda perlu membuat instance optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

     # 3. Pelatihan Model
    history = model.fit(
        X_train, y_train,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_data=(X_val, y_val),
        verbose=0 # Atur verbose ke 0 atau 1 untuk mengurangi output saat tuning
    )
    
    return model, history

# Fungsi untuk menghitung MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask): 
        return 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def tampilkan_menu():
    print(" Prediksi LSTM : ")
    print("+=========================================+")
    print("| 1. Copy ke data LSTM                    |")
    print("| 2. Long Short-Term Memory (LSTM)        |")
    print("| 3. Keluar Aplikasi                      |")
    print("+=========================================+")

# Plotting predictions
def plot_predictions(y_true, y_pred, target_names, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Visualisasi Prediksi ---")
    for i, col_name in enumerate(target_names):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[:, i], label='Actual', color='blue')
        plt.plot(y_pred[:, i], label='Predicted', color='red', linestyle='--')
        plt.title(f'Actual vs Predicted for {col_name}')
        plt.xlabel('Time Step (Index)')
        plt.ylabel(col_name)
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(output_dir, f'{col_name}_prediction.png')
        plt.savefig(plot_path)
        plt.close() 
        print(f"Plot untuk '{col_name}' berhasil disimpan ke '{plot_path}'")
    print("-" * 30)


def main():
    while True:
        tampilkan_menu()
        pilihan = input("Silahkan pilih [1-3] : ")
        print(" ")

        if pilihan == '1':
            print("+ INFO : Sedang mengkopi ke data LSTM, Mohon menunggu..")
            print("")

            source_file_path = r'/content/drive/MyDrive/prediksi/AiTesis/0.Data/data_time_series_2019_2024.xlsx'
            destination_file_path = r'/content/drive/MyDrive/prediksi/AiTesis/0.Data/data_lstm_2019_2024.xlsx'

            try:
                df = pd.read_excel(source_file_path, sheet_name='Data Harian')
                print("File asli berhasil dibaca. Melanjutkan proses penyalinan.")
                shutil.copy(source_file_path, destination_file_path)
                print(f"\nBerhasil menyalin file.")
            except FileNotFoundError:
                print(f"Error: File '{source_file_path}' tidak ditemukan. Pastikan path sumber sudah benar.")
            except ValueError as e:
                print(f"Peringatan: Sheet 'Data Harian' mungkin tidak ditemukan atau ada masalah lain saat membaca file asli. Pesan: {e}")
                print("Meskipun demikian, proses penyalinan file akan tetap dicoba jika file utama ada.")
                try:
                    shutil.copy(source_file_path, destination_file_path)
                    print(f"\nBerhasil menyalin file meskipun ada peringatan.")
                except FileNotFoundError:
                    print(f"Error: File '{source_file_path}' tidak ditemukan setelah peringatan awal.")
                except Exception as e_copy:
                    print(f"Terjadi error saat menyalin file: {e_copy}")
            except Exception as e:
                print(f"Terjadi error lain: {e}")

            input("\nTekan Enter untuk kembali ke menu...")
#=========================================================================================================================================
        elif pilihan == '2':
            print("+ INFO : Sedang Melakukan Prediksi dengan algoritma LSTM , Mohon menunggu..")

            try:
                # Simpan df_lstm_original sebelum preprocessing untuk inverse transform
                df_lstm_original_for_scaling = pd.read_excel(r'/content/drive/MyDrive/prediksi/AiTesis/0.Data/data_lstm_2019_2024.xlsx', sheet_name='Data Harian')


                df_lstm = pd.read_excel(r'/content/drive/MyDrive/prediksi/AiTesis/0.Data/data_lstm_2019_2024.xlsx', sheet_name='Data Harian')
                print("Data LSTM berhasil dimuat.")
                print(df_lstm)
                print("")

                # Perbaiki Error Data Input
                print("\n--- Perbaikan Error Data Input ---")

                # Daftar kolom yang akan menjadi fitur/target
                numerical_cols_for_check = [
                    'Pesawat_DTG', 'Pesawat_BRK', 'Penumpang_DTG', 'Penumpang_BRK',
                    'Bagasi_DTG', 'Bagasi_BRK', 'Cargo_DTG', 'Cargo_BRK', 'Pos_DTG', 'Pos_BRK'
                ]

                errors_fixed = False
                for col in numerical_cols_for_check:
                    if col in df_lstm.columns:
                        if not pd.api.types.is_numeric_dtype(df_lstm[col]):
                            initial_non_numeric_count = df_lstm[col].apply(lambda x: pd.to_numeric(x, errors='coerce')).isnull().sum() - df_lstm[col].isnull().sum()
                            df_lstm[col] = pd.to_numeric(df_lstm[col], errors='coerce')
                            if initial_non_numeric_count > 0:
                                print(f"   - Kolom '{col}': {initial_non_numeric_count} nilai non-numerik diubah menjadi NaN.")
                                errors_fixed = True

                        negative_count = (df_lstm[col] < 0).sum()
                        if negative_count > 0:
                            df_lstm[col] = np.where(df_lstm[col] < 0, np.nan, df_lstm[col])
                            print(f"   - Kolom '{col}': {negative_count} nilai negatif diubah menjadi NaN.")
                            errors_fixed = True
                    else:
                        print(f"   - Kolom '{col}' tidak ditemukan di DataFrame.")

                if not errors_fixed:
                    print("Tidak ada error input data yang terdeteksi atau diperbaiki.")
                print("-" * 30)


                print("\n--- Cek Missing Values ---")
                print(df_lstm.isnull().sum())
                print("-" * 30)

                print("\n--- Memeriksa dan Menghapus Data Duplikat ---")
                initial_rows_dup = df_lstm.shape[0]
                df_lstm.drop_duplicates(inplace=True)
                rows_after_dropping_dup = df_lstm.shape[0]

                if initial_rows_dup > rows_after_dropping_dup:
                    print(f"Jumlah baris sebelum menghapus duplikat: {initial_rows_dup}")
                    print(f"Jumlah baris setelah menghapus duplikat: {rows_after_dropping_dup}")
                    print(f"Total {initial_rows_dup - rows_after_dropping_dup} baris duplikat dihapus.")
                else:
                    print("Tidak ada baris duplikat yang ditemukan.")
                print("-" * 30)

                print("\n--- Deteksi dan Penanganan Outlier (Metode IQR) ---")

                actual_numeric_cols = [
                    col for col in numerical_cols_for_check
                    if col in df_lstm.columns and pd.api.types.is_numeric_dtype(df_lstm[col])
                ]

                outliers_handled = False
                if not actual_numeric_cols:
                    print("Tidak ada kolom numerik yang relevan ditemukan untuk deteksi outlier.")
                else:
                    for col in actual_numeric_cols:
                        Q1 = df_lstm[col].quantile(0.25)
                        Q3 = df_lstm[col].quantile(0.75)
                        IQR = Q3 - Q1

                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        col_outliers = df_lstm[(df_lstm[col] < lower_bound) | (df_lstm[col] > upper_bound)]

                        if not col_outliers.empty:
                            outliers_handled = True
                            print(f"\nOutlier ditemukan pada kolom '{col}' ({len(col_outliers)} data):")
                            print(col_outliers[[col]])

                            df_lstm.loc[(df_lstm[col] < lower_bound) | (df_lstm[col] > upper_bound), col] = np.nan
                            print(f"Outlier di kolom '{col}' telah diganti dengan NaN.")

                    if not outliers_handled:
                        print("Tidak ada outlier ditemukan pada kolom numerik yang relevan.")
                print("-" * 30)

                print("\n--- Imputasi Missing Values (Interpolasi Linear) ---")

                if 'Tanggal' in df_lstm.columns:
                    df_lstm['Tanggal'] = pd.to_datetime(df_lstm['Tanggal'])
                    df_lstm.set_index('Tanggal', inplace=True)
                    print("Kolom 'Tanggal' telah diubah menjadi indeks DataFrame.")
                else:
                    print("Kolom 'Tanggal' tidak ditemukan. Pastikan data Anda memiliki kolom tanggal.")

                initial_nans = df_lstm.isnull().sum().sum()
                df_lstm.interpolate(method='linear', limit_direction='both', inplace=True)
                nans_after_interpolation = df_lstm.isnull().sum().sum()

                if initial_nans > 0:
                    if nans_after_interpolation == 0:
                        print("Semua missing values (termasuk outlier yang diganti NaN) telah diimputasi.")
                    else:
                        print(f"Beberapa missing values mungkin tidak dapat diimputasi (sisa NaN: {nans_after_interpolation}).")
                        print(df_lstm.isnull().sum())
                else:
                    print("Tidak ada missing values awal yang ditemukan untuk diimputasi.")
                print("-" * 30)

                print("\n--- Normalisasi Data (Min-Max Scaling) ---")

                # scaler ini akan digunakan untuk inverse transform 
                scaler = MinMaxScaler()
                if actual_numeric_cols:
                
                    #  df_lstm_original_for_scaling juga hanya berisi kolom numerik 
                    if all(col in df_lstm_original_for_scaling.columns for col in actual_numeric_cols):
                        scaler.fit(df_lstm_original_for_scaling[actual_numeric_cols].values)
                        df_lstm[actual_numeric_cols] = scaler.transform(df_lstm[actual_numeric_cols])
                        print("Data numerik telah dinormalisasi (Min-Max Scaling).")
                        print(df_lstm.head)
                    else:
                        print("Peringatan: Kolom untuk scaling tidak lengkap di data original. Tidak dapat melakukan scaling.")
                        scaler.fit(df_lstm[actual_numeric_cols].values)
                        df_lstm[actual_numeric_cols] = scaler.transform(df_lstm[actual_numeric_cols])
                        print("Data numerik telah dinormalisasi (Min-Max Scaling) menggunakan data yang sudah diimputasi.")

                else:
                    print("Tidak ada kolom numerik yang ditemukan untuk normalisasi.")
                print("-" * 30)

                # Menyimpan Data Normalisasi ---
                print("\n--- Menyimpan Data Normalisasi ---")
                output_dir = r'/content/drive/MyDrive/prediksi/AiTesis/1.LSTM/1.prepocessingLSTM/normalisasi'
                output_filename = 'data_lstm_normalized_2019_2024.xlsx'
                output_path = os.path.join(output_dir, output_filename)

                os.makedirs(output_dir, exist_ok=True) 

                try:
                    df_lstm.to_excel(output_path, sheet_name='Data Normalisasi', index=True)
                    print(f"Data normalisasi berhasil disimpan")
                except Exception as e:
                    print(f"Error saat menyimpan data normalisasi: {e}")
                print("-" * 30)

                # Pembentukan Urutan (Sequence/Lagged Features) untuk MULTI-OUTPUT
                print("\n--- Pembentukan Urutan (Sequence/Lagged Features) (Multi-Output) ---")

                n_steps = 7  # JUMLAH LANGKAH WAKTU/LOOKBACK (7 hari sebelumnya)

                # DAFTAR KOLOM TARGET
                multi_target_column_names = [
                    'Pesawat_DTG', 'Pesawat_BRK', 'Penumpang_DTG', 'Penumpang_BRK',
                    'Bagasi_DTG', 'Bagasi_BRK', 'Cargo_DTG', 'Cargo_BRK', 'Pos_DTG', 'Pos_BRK'
                ]

                # memastikan semua kolom target ada di 'actual_numeric_cols'
                missing_targets = [col for col in multi_target_column_names if col not in actual_numeric_cols]
                if missing_targets:
                    print(f"Error: Kolom target berikut tidak ditemukan dalam data yang sudah diproses: {missing_targets}")
                    print("Pastikan nama kolom sudah benar dan kolom-kolom tersebut numerik.")
                    input("\nTekan Enter untuk kembali ke menu...")
                    return

                # Mendapatkan indeks dari SEMUA KOLOM TARGET dalam array fitur numerik yang dinormalisasi
                target_column_indices_in_features = [actual_numeric_cols.index(col_name) for col_name in multi_target_column_names]

                # menyiapkan data untuk pembentukan urutan (menggunakan data numpy dari kolom numerik yang dinormalisasi)
                data_for_sequence = df_lstm[actual_numeric_cols].values

                # Membuat urutan (X_sequences akan berisi semua fitur, y_target akan berisi semua target)
                X_sequences, y_target = create_sequences(data_for_sequence, n_steps, target_column_indices_in_features)

                n_features = X_sequences.shape[2]
                n_outputs = y_target.shape[1]

                print(f"Menggunakan {n_steps} langkah waktu (timesteps) untuk prediksi.")
                print(f"Kolom target prediksi (multi-output): {multi_target_column_names}")
                print(f"Bentuk data input (X_sequences) untuk LSTM: {X_sequences.shape} (sampel, timesteps, fitur)")
                print(f"Bentuk data target (y_target) untuk LSTM: {y_target.shape} (sampel, jumlah_target)")
                print("-" * 30)


                # Pembagian Data (Data Splitting): Training 80%, Validation 10%, Testing 10%
                print("\n--- Pembagian Data (Data Splitting): Training 80%, Validation 10%, Testing 10% ---")
            
                #total_samples = len(X_sequences)
                #80%
                train_end_date = pd.to_datetime('2023-10-31')
                val_end_date = pd.to_datetime('2024-05-31')
            
                try:
                    idx_pos_train = df_lstm.index.searchsorted(train_end_date)
                    if idx_pos_train == len(df_lstm.index): # Jika tanggal setelah data terakhir
                        train_idx_in_df_lstm = len(df_lstm.index) - 1
                    elif idx_pos_train == 0: # Jika tanggal sebelum data pertama
                        train_idx_in_df_lstm = 0
                    else:
                        # Bandingkan jarak ke tanggal di idx_pos dan idx_pos-1
                        diff_after = abs(df_lstm.index[idx_pos_train] - train_end_date)
                        diff_before = abs(df_lstm.index[idx_pos_train - 1] - train_end_date)
                        if diff_before <= diff_after:
                            train_idx_in_df_lstm = idx_pos_train - 1
                        else:
                            train_idx_in_df_lstm = idx_pos_train
                    
                    # Indeks akhir untuk X_train / y_train dalam X_sequences
                    train_end_sequence_idx = train_idx_in_df_lstm - n_steps + 1
                    if train_end_sequence_idx < 0:
                        train_end_sequence_idx = 0

                    # --- Mencari val_idx_in_df_lstm (pengganti get_loc(method='nearest')) ---
                    idx_pos_val = df_lstm.index.searchsorted(val_end_date)
                    if idx_pos_val == len(df_lstm.index): # Jika tanggal setelah data terakhir
                        val_idx_in_df_lstm = len(df_lstm.index) - 1
                    elif idx_pos_val == 0: # Jika tanggal sebelum data pertama
                        val_idx_in_df_lstm = 0
                    else:
                        # Bandingkan jarak ke tanggal di idx_pos dan idx_pos-1
                        diff_after = abs(df_lstm.index[idx_pos_val] - val_end_date)
                        diff_before = abs(df_lstm.index[idx_pos_val - 1] - val_end_date)
                        if diff_before <= diff_after:
                            val_idx_in_df_lstm = idx_pos_val - 1
                        else:
                            val_idx_in_df_lstm = idx_pos_val

                    # Indeks akhir untuk X_val / y_val dalam X_sequences
                    val_end_sequence_idx = val_idx_in_df_lstm - n_steps + 1
                    if val_end_sequence_idx < train_end_sequence_idx:
                        val_end_sequence_idx = train_end_sequence_idx
                    
                    # Batasi indeks agar tidak melebihi panjang total X_sequences
                    train_end_sequence_idx = min(train_end_sequence_idx, len(X_sequences))
                    val_end_sequence_idx = min(val_end_sequence_idx, len(X_sequences))

                    

                    # Membagi data secara sekuensial (berdasarkan waktu)
                    X_train = X_sequences[:train_end_sequence_idx]
                    y_train = y_target[:train_end_sequence_idx]

                    X_val = X_sequences[train_end_sequence_idx:val_end_sequence_idx]
                    y_val = y_target[train_end_sequence_idx:val_end_sequence_idx]

                    X_test = X_sequences[val_end_sequence_idx:]
                    y_test = y_target[val_end_sequence_idx:]

                    print(f"Total sampel urutan: {len(X_sequences)}")
                    print(f"Tanggal akhir training set: {df_lstm.index[train_idx_in_df_lstm].strftime('%Y-%m-%d')}")
                    print(f"Ukuran Training Set: {len(X_train)} sampel")
                    print(f"Bentuk X_train: {X_train.shape}")
                    print(f"Bentuk y_train: {y_train.shape}")
                    print("---")
                    print(f"Tanggal akhir validation set: {df_lstm.index[val_idx_in_df_lstm].strftime('%Y-%m-%d')}")
                    print(f"Ukuran Validation Set: {len(X_val)} sampel")
                    print(f"Bentuk X_val: {X_val.shape}")
                    print(f"Bentuk y_val: {y_val.shape}")
                    print("---")
                    
                    if len(X_test) > 0:
                        test_start_date_actual = df_lstm.index[val_idx_in_df_lstm + 1]
                        test_end_date_actual = df_lstm.index[-1]
                        print(f"Tanggal awal testing set: {test_start_date_actual.strftime('%Y-%m-%d')}")
                        print(f"Tanggal akhir testing set: {test_end_date_actual.strftime('%Y-%m-%d')}")
                    else:
                        print("Tidak ada sampel untuk Testing Set (data mungkin berakhir sebelum tanggal validasi selesai).")
                    
                    print(f"Ukuran Testing Set: {len(X_test)} sampel")
                    print(f"Bentuk X_test: {X_test.shape}")
                    print(f"Bentuk y_test: {y_test.shape}")
                    print("-" * 30)

                    

                except Exception as e:
                    # Menangkap semua jenis error yang mungkin terjadi selama pencarian indeks
                    print(f"Terjadi error saat pembagian data berdasarkan tanggal: {e}")
                    print("Pastikan 'df_lstm' memiliki DatetimeIndex dan mencakup rentang tanggal yang diinginkan.")
                    print(f"Tanggal yang dicari: Training hingga {train_end_date.strftime('%Y-%m-%d')}, Validation hingga {val_end_date.strftime('%Y-%m-%d')}")
                    import traceback
                    traceback.print_exc()
                    input("\nTekan Enter untuk kembali ke menu...")
                    return

                # menghitung indeks untuk pembagian
                #train_end_index = int(total_samples * 0.80)
                #val_end_index = int(total_samples * 0.90) # 80% + 10% = 90% dari total

                # Membagi data secara sekuensial (berdasarkan waktu)
                #X_train = X_sequences[:train_end_sequence_idx]
                #y_train = y_target[:train_end_sequence_idx]

                #X_val = X_sequences[train_end_sequence_idx:val_end_sequence_idx]
                #y_val = y_target[train_end_sequence_idx:val_end_sequence_idx]

                #X_test = X_sequences[val_end_sequence_idx:]
                #y_test = y_target[val_end_sequence_idx:]

                #print(f"Total sampel urutan: {len(X_sequences)}")
                #print(f"Tanggal akhir training set: {df_lstm.index[train_idx_in_df_lstm].strftime('%Y-%m-%d')}")
                #print(f"Ukuran Training Set: {len(X_train)} sampel")
                #print(f"Bentuk X_train: {X_train.shape}")
                #print(f"Bentuk y_train: {y_train.shape}")
                #print("---")
                #print(f"Tanggal akhir validation set: {df_lstm.index[val_idx_in_df_lstm].strftime('%Y-%m-%d')}")
                #print(f"Ukuran Validation Set: {len(X_val)} sampel")
                #print(f"Bentuk X_val: {X_val.shape}")
                #print(f"Bentuk y_val: {y_val.shape}")
                #print("---")
                #print(f"Ukuran Testing Set: {len(X_test)} sampel")
                #print(f"Bentuk X_test: {X_test.shape}")
                #print(f"Bentuk y_test: {y_test.shape}")
                #print("-" * 30)

                 # --- 3. Implementasi Random Search Loop ---
                print("\n--- Memulai Random Search untuk Optimasi Hyperparameter ---")

                best_loss = float('inf')
                best_hyperparameters = {}
                best_model = None

                num_trials = 20 # Misalnya, lakukan 10 kali percobaan random search

                # Definisi ruang pencarian 
                param_grid = {
                    'lstm_units': [32, 50, 64, 100],
                    'dropout_rate': [0.0, 0.1, 0.2, 0.3],
                    'learning_rate': [0.01, 0.001, 0.0001],
                    'epochs': [50, 75, 100],
                    'batch_size': [16, 32, 64]
                }

                for trial in range(num_trials):
                    print(f"\n--- Percobaan Random Search ke-{trial+1}/{num_trials} ---")

                    # Memilih hyperparameter secara acak
                    current_hyperparameters = {
                        'lstm_units': np.random.choice(param_grid['lstm_units']),
                        'dropout_rate': np.random.choice(param_grid['dropout_rate']),
                        'learning_rate': np.random.choice(param_grid['learning_rate']),
                        'epochs': np.random.choice(param_grid['epochs']),
                        'batch_size': np.random.choice(param_grid['batch_size'])
                    }
                    print(f"Hyperparameter yang dicoba: {current_hyperparameters}")

                    # Membangun, melatih, dan mengevaluasi model
                    # (Anda akan perlu membuat fungsi build_and_train_model_lstm ini)
                    model, history = build_and_train_model_lstm(
                        X_train, y_train, X_val, y_val,
                        n_steps, n_features, n_outputs,
                        current_hyperparameters['lstm_units'],
                        current_hyperparameters['dropout_rate'],
                        current_hyperparameters['learning_rate'],
                        current_hyperparameters['epochs'],
                        current_hyperparameters['batch_size']
                    )

                    val_loss = history.history['val_loss'][-1]
                    print(f"Validation Loss untuk percobaan ini: {val_loss:.4f}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_hyperparameters = current_hyperparameters
                        best_model = model # Simpan model terbaik

                print("\n--- Random Search Selesai ---")
                print(f"Best Validation Loss: {best_loss:.4f}")
                print(f"Best Hyperparameters: {best_hyperparameters}")
                print("-" * 30)

                # Setelah random search selesai, gunakan best_model untuk evaluasi dan prediksi
                # Pastikan variabel 'model' yang digunakan di tahap selanjutnya adalah best_model
                model = best_model # Mengganti model yang ada dengan model terbaik

                #n_steps = X_train.shape[1]
                #n_features = X_train.shape[2]
                #n_outputs = y_train.shape[1] # kolom target yang akan diprediksi
                print("\n--- Desain Arsitektur Model (Model Architecture Design) ---")
                # Ringkasan Arsitektur Model
                model.summary()
                print("-" * 30)

                print("\nArsitektur Model LSTM telah didesain sesuai spesifikasi Anda:")
                print(f"   - Jumlah Lapisan LSTM:")
                print(f"   - Jumlah Unit per Lapisan LSTM:")
                print(f"   - Lapisan Tambahan (Dense): 1 lapisan dengan 25 unit")
                print(f"   - Lapisan Output (Dense): 1 lapisan dengan {n_outputs} unit (sesuai jumlah target)")
                print(f"   - Fungsi Aktivasi: 'tanh' untuk lapisan tersembunyi, 'linear' untuk output.")
                print("\nKode di atas mendefinisikan arsitektur model dan menampilkan ringkasannya.")


                # --- KOMPILASI MODEL ---
                print("\n--- Kompilasi Model ---")
                #model.compile(optimizer='adam', loss='mse')
                print("Model berhasil dikompilasi.")
                print("-" * 30)

                # --- PELATIHAN MODEL ---
                print("\n--- Pelatihan Model (Model Training) ---")
                #epochs = 50
                #batch_size = 32

                #history = model.fit(
                #    X_train, y_train,
                #    epochs=epochs,
                #    batch_size=batch_size,
                #    validation_data=(X_val, y_val),
                #    verbose=1
                #)

                #print(f"\nModel selesai dilatih selama {epochs} epoch dengan batch_size {batch_size}.")
                #print("-" * 30)

                # Optional: Tampilkan ringkasan history pelatihan
                print("\n--- Ringkasan Pelatihan ---")
                print(f"Loss terakhir pada Training Set: {history.history['loss'][-1]:.4f}")
                print(f"Loss terakhir pada Validation Set: {history.history['val_loss'][-1]:.4f}")
                print("Untuk analisis lebih lanjut, Anda bisa memplot history['loss'] dan history['val_loss'].")
                print("-" * 30)

                # --- Evaluasi Model ---
                print("\n--- Evaluasi Model (Model Evaluation) ---")

                # prediksi pada data test
                print("Melakukan prediksi pada data test...")
                y_pred_scaled = model.predict(X_test)
                print(f"Bentuk prediksi (scaled): {y_pred_scaled.shape}")

                # total jumlah fitur dari scaler (jumlah kolom aktual_numeric_cols)
                total_features_in_scaler = scaler.n_features_in_

                # Inisialisasi array dummy untuk y_test_inverted
                y_test_full_features = np.zeros((len(y_test), total_features_in_scaler))

                # Memasukkan y_test ke kolom yang sesuai
                # memastikan target_column_indices_in_features sesuai dengan indeks kolom target saat scaling
                for i, col_idx in enumerate(target_column_indices_in_features):
                    y_test_full_features[:, col_idx] = y_test[:, i]

                # melakukan inverse_transform
                y_test_inverted = scaler.inverse_transform(y_test_full_features)
                # mengambil kembali hanya kolom target
                y_test_actual = y_test_inverted[:, target_column_indices_in_features]

                

                print(f"y_test.npy disimpan dengan shape: {y_test_actual.shape}")
                np.save(r'/content/drive/MyDrive/prediksi/AiTesis/y_test.npy', y_test_actual)
                print("Data y_test.npy berhasil disimpan.")
                

                # Inverse transform untuk y_pred_scaled (nilai prediksi)
                # melakukan hal yang sama seperti y_test_full_features
                y_pred_full_features = np.zeros((len(y_pred_scaled), total_features_in_scaler))
                for i, col_idx in enumerate(target_column_indices_in_features):
                    y_pred_full_features[:, col_idx] = y_pred_scaled[:, i]

                y_pred_actual = scaler.inverse_transform(y_pred_full_features)
                y_pred_actual = y_pred_actual[:, target_column_indices_in_features]

                y_true_all_global.extend(y_test_actual.ravel())
                y_pred_all_global.extend(y_pred_actual.ravel())

                print(f"y_pred_lstm.npy disimpan dengan shape: {y_pred_actual.shape}")
                np.save(f'/content/drive/MyDrive/prediksi/AiTesis/y_pred_lstm.npy', y_pred_actual)
                print(f"y_pred_lstm.npy berhasil disimpan.")

                # Memastikan tidak ada nilai negatif di y_pred_actual setelah inverse transform (jika data aslinya non-negatif)
                y_pred_actual[y_pred_actual < 0] = 0 # Misal: ubah nilai negatif jadi 0 jika tidak masuk akal

                # Menghitung Metrik Kinerja untuk setiap kolom target
                evaluation_results = pd.DataFrame(index=multi_target_column_names, columns=['RMSE', 'MAE', 'MAPE', 'R2']) 

                for i, col_name in enumerate(multi_target_column_names):
                    rmse = np.sqrt(mean_squared_error(y_test_actual[:, i], y_pred_actual[:, i]))
                    mae = mean_absolute_error(y_test_actual[:, i], y_pred_actual[:, i])
                    mape = mean_absolute_percentage_error(y_test_actual[:, i], y_pred_actual[:, i])
                    r2 = r2_score(y_test_actual[:, i], y_pred_actual[:, i]) 

                    evaluation_results.loc[col_name, 'RMSE'] = rmse
                    evaluation_results.loc[col_name, 'MAE'] = mae
                    evaluation_results.loc[col_name, 'MAPE'] = mape
                    evaluation_results.loc[col_name, 'R2'] = r2 
                
        
                print("\n--- Hasil Evaluasi Model pada Test Set ---")
                print(evaluation_results)
                print("\nInterpretasi Hasil:")
                print(" - RMSE (Root Mean Squared Error): Mengukur rata-rata besarnya kesalahan prediksi. Nilai yang lebih rendah lebih baik.")
                print(" - MAE (Mean Absolute Error): Mengukur rata-rata besar kesalahan absolut.  Nilai yang lebih rendah lebih baik.")
                print(" - MAPE (Mean Absolute Percentage Error): Mengukur rata-rata kesalahan dalam persentase.  Nilai yang lebih rendah lebih baik.")
                print(" - R^2 (R-squared): Mengukur proporsi varians dalam variabel dependen yang dapat diprediksi dari variabel independen. Nilai mendekati 1 lebih baik (model menjelaskan lebih banyak varians). Nilai 0 berarti model tidak menjelaskan varians sama sekali. Nilai negatif berarti model lebih buruk daripada hanya memprediksi rata-rata.")
                print("-" * 30)

                print("\n" + "="*50)
                print("--- METRIK EVALUASI MODEL KESELURUHAN ---")
                print("="*50)

                # Gunakan data yang sudah dikumpulkan di wadah global
                y_true_total = np.array(y_true_all_global)
                y_pred_total = np.array(y_pred_all_global)

                # Pastikan y_true_all_global dan y_pred_all_global tidak kosong
                if y_true_total.size > 0 and y_pred_total.size > 0:
                    # Hitung metrik
                    rmse_total = np.sqrt(mean_squared_error(y_true_total, y_pred_total))
                    mae_total = mean_absolute_error(y_true_total, y_pred_total)
                    r2_total = r2_score(y_true_total, y_pred_total)
                    mape_total = mean_absolute_percentage_error(y_true_total, y_pred_total)

                    print(f"RMSE (Keseluruhan): {rmse_total:.4f}")
                    print(f"MAE (Keseluruhan): {mae_total:.4f}")
                    print(f"R2 (Keseluruhan): {r2_total:.4f}")
                    print(f"MAPE (Keseluruhan): {mape_total:.4f}%")
                else:
                    print("Tidak ada data untuk dihitung. Pastikan data 'y_true_all_global' dan 'y_pred_all_global' telah terisi.")
                
                print("="*50)

                # --- Opsional: Menyimpan Hasil Evaluasi ke Excel dengan Multi-sheet ---
                print("\n--- Menyimpan Hasil Evaluasi ---")
                eval_output_dir = r'/content/drive/MyDrive/prediksi/AiTesis/1.LSTM/2.evaluasiLSTM'
                eval_output_filename = 'evaluasi_model_lstm.xlsx'
                eval_output_path = os.path.join(eval_output_dir, eval_output_filename)

                # Membuat DataFrame untuk metrik keseluruhan
                overall_results = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'MAPE', 'R2'],
                    'Value': [rmse_total, mae_total, mape_total, r2_total]
                }).set_index('Metric')

                os.makedirs(eval_output_dir, exist_ok=True)
                try:
                    with pd.ExcelWriter(eval_output_path) as writer:
                        # Simpan hasil per kolom ke sheet pertama
                        evaluation_results.to_excel(writer, sheet_name='Evaluasi per Kolom')
                        
                        # Simpan hasil keseluruhan ke sheet kedua
                        overall_results.to_excel(writer, sheet_name='Evaluasi Keseluruhan')
                    
                    print(f"\nHasil evaluasi model berhasil disimpan ke '{eval_output_path}'")
                    print("Tersedia di dua sheet: 'Evaluasi per Kolom' dan 'Evaluasi Keseluruhan'.")
                except Exception as e:
                    print(f"Error saat menyimpan hasil evaluasi: {e}")
                print("-" * 30)

                # --- Memanggil fungsi plotting ---
                plot_output_dir = r'/content/drive/MyDrive/prediksi/AiTesis/1.LSTM/3.grafik_prediksi'
                plot_predictions(y_test_actual, y_pred_actual, multi_target_column_names, plot_output_dir)


                # --- Prediksi untuk Tahun 2025 ---
                print("\n--- Melakukan Prediksi untuk Tahun 2025 ---")

                # 1. Mendapatkan data historis terakhir yang cukup untuk membentuk sequence
                # Mengambil 7 hari terakhir dari df_lstm (yang sudah dinormalisasi dan diimputasi)
                last_n_steps_data = df_lstm[actual_numeric_cols].tail(n_steps).values

                if last_n_steps_data.shape[0] < n_steps:
                    print(f"Peringatan: Tidak cukup data historis ({last_n_steps_data.shape[0]} baris) untuk membentuk {n_steps} langkah waktu.")
                    print("Prediksi 2025 tidak dapat dilakukan. Pastikan data input mencukupi.")
                else:
                    # Membuat rentang tanggal untuk tahun 2025
                    start_date_2025 = pd.to_datetime('2025-01-01')
                    end_date_2025 = pd.to_datetime('2025-12-31')
                    future_dates = pd.date_range(start=start_date_2025, end=end_date_2025, freq='D')

                    predictions_2025_scaled = []
                    current_input = last_n_steps_data.copy() # Start with the last known sequence

                    for _ in range(len(future_dates)):
                        input_for_prediction = current_input.reshape(1, n_steps, n_features)
                        next_day_prediction_scaled = model.predict(input_for_prediction, verbose=0)[0]
                        predictions_2025_scaled.append(next_day_prediction_scaled)

                        # membuat baris dummy dengan shape dari total_features_in_scaler
                        next_full_feature_row_scaled = np.zeros(total_features_in_scaler)

                        # Mengisi kolom target dengan nilai prediksi
                        for i, col_idx in enumerate(target_column_indices_in_features):
                            next_full_feature_row_scaled[col_idx] = next_day_prediction_scaled[i]

                        last_known_full_features = df_lstm[actual_numeric_cols].iloc[-1].values
                        non_target_indices = [idx for idx in range(total_features_in_scaler) if idx not in target_column_indices_in_features]
                        for non_target_idx in non_target_indices:
                             next_full_feature_row_scaled[non_target_idx] = last_known_full_features[non_target_idx]

                        current_input = np.vstack([current_input[1:], next_full_feature_row_scaled])


                    # konversi skala prediksi ke nilai actual 
                    predictions_2025_scaled = np.array(predictions_2025_scaled)
                    predictions_2025_actual_full_features = np.zeros((len(predictions_2025_scaled), total_features_in_scaler))

                    for i, col_idx in enumerate(target_column_indices_in_features):
                        predictions_2025_actual_full_features[:, col_idx] = predictions_2025_scaled[:, i]

                    predictions_2025_actual = scaler.inverse_transform(predictions_2025_actual_full_features)
                    predictions_2025_actual = predictions_2025_actual[:, target_column_indices_in_features]
                    predictions_2025_actual[predictions_2025_actual < 0] = 0

                    # Membuat DataFrame untuk prediksi tahun 2025 
                    df_predictions_2025 = pd.DataFrame(predictions_2025_actual, columns=multi_target_column_names, index=future_dates)
                    df_predictions_2025.index.name = 'Tanggal'
                    df_predictions_2025.index = df_predictions_2025.index.strftime('%d-%m-%Y')
                    print("\n--- Tabel Prediksi untuk Tahun 2025 ---")
                    print(df_predictions_2025.head()) 
                    print(f"Total {len(df_predictions_2025)} hari diprediksi untuk tahun 2025.")

                    
                    # Menyimpan Prediksi 2025
                    predictions_output_dir = r'/content/drive/MyDrive/prediksi/AiTesis/1.LSTM/4.prediksi_2025'
                    predictions_output_filename = 'prediksi_lstm_2025.xlsx'
                    predictions_output_path = os.path.join(predictions_output_dir, predictions_output_filename)

                    os.makedirs(predictions_output_dir, exist_ok=True)
                    try:
                        df_predictions_2025.to_excel(predictions_output_path, sheet_name='Prediksi 2025')
                        print(f"\nPrediksi tahun 2025 berhasil disimpan ke '{predictions_output_path}'")
                    except Exception as e:
                        print(f"Error saat menyimpan prediksi 2025: {e}")
                print("-" * 30)

            except FileNotFoundError:
                print("Error: File 'data_lstm_2019_2024.xlsx' belum ditemukan. Harap jalankan Opsi 1 terlebih dahulu.")
            except Exception as e:
                print(f"Terjadi error saat menjalankan LSTM: {e}")
                import traceback
                traceback.print_exc() # Untuk melihat detail error

            input("\nTekan Enter untuk kembali ke menu...")
#===============================================================================================================================
        elif pilihan == '3':
            print("+ INFO : Anda memilih keluar dari aplikasi. Sampai jumpa!")
            break

        else :
            print("+ INFO : Pilihan tidak valid, silahkan masukkan 1, 2, atau 3. ")
            input("\nTekan Enter untuk melanjutkan...")

if __name__ == "__main__":
    main()