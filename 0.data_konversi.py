# \myenv\Scripts\Activate

import pandas as pd
import numpy as np
from datetime import datetime

excel_file_path = r"/content/drive/MyDrive/prediksi/AiTesis/0.Data/datalatih.xlsx"

# Daftar tahun yang ingin diproses
years_to_process = [2019, 2020, 2021, 2022, 2023, 2024]

# Inisialisasi DataFrame kosong 
all_years_daily_data = pd.DataFrame()

# nama kolom 
new_column_names = [
    'Bulan',
    'Pesawat_DTG',
    'Pesawat_BRK',
    'Penumpang_DTG',
    'Penumpang_BRK',
    'Bagasi_DTG',
    'Bagasi_BRK',
    'Cargo_DTG',
    'Cargo_BRK',
    'Pos_DTG',
    'Pos_BRK',
]

# bobot hari
day_weights = {
    'Monday': 0.9,
    'Tuesday': 0.95,
    'Wednesday': 1.0,
    'Thursday': 1.05,
    'Friday': 1.1,
    'Saturday': 1.5,
    'Sunday': 1.5,
}

# Pemetaan nama bulan ke angka 
month_mapping = {
    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
    'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
}

# Loop melalui setiap tahun
for year in years_to_process:
    sheet_name = str(year) # Nama sheet yang sesuai dengan tahun

    try:
        # Baca data dari sheet 
        df_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name, skiprows=3, header=None)
        
        if len(df_raw.columns) == len(new_column_names):
            df_raw.columns = new_column_names
        else:
            continue # Lanjutkan ke tahun berikutnya jika ada masalah kolom

        df_raw.dropna(how='all', inplace=True)
        df_raw['Bulan'] = df_raw['Bulan'].astype(str).str.capitalize() 
        df_raw['Month_Num'] = df_raw['Bulan'].map(month_mapping)

        initial_rows = df_raw.shape[0]
        df_raw.dropna(subset=['Month_Num'], inplace=True)
        rows_after_month_dropna = df_raw.shape[0]
        if df_raw.empty:
            continue # Lanjutkan ke tahun berikutnya jika tidak ada data yang tersisa

        df_raw['Month_Num'] = df_raw['Month_Num'].astype(int)

        numerical_columns_to_process = [col for col in new_column_names if col not in ['Bulan']]

        for col in numerical_columns_to_process:
            if col in df_raw.columns: 
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        df_raw = df_raw.infer_objects(copy=False)

        current_year_daily_data = pd.DataFrame()
        total_months_processed_in_year = 0

        for index, monthly_row in df_raw.iterrows():
            current_month_num = int(monthly_row['Month_Num'])
            current_month_name = monthly_row['Bulan']

            start_date = pd.to_datetime(f'{year}-{current_month_num}-01')
            end_date = start_date + pd.offsets.MonthEnd(0)
            daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')

            current_month_daily_df = pd.DataFrame({'Tanggal': daily_dates})
            current_month_daily_df['Hari'] = current_month_daily_df['Tanggal'].dt.day_name()
            
            current_month_daily_df['Weight'] = current_month_daily_df['Hari'].map(day_weights)
            total_monthly_weight = current_month_daily_df['Weight'].sum()

            for col in numerical_columns_to_process:
                monthly_total = monthly_row[col]
                if pd.isna(monthly_total) or monthly_total == 0 or total_monthly_weight == 0:
                    current_month_daily_df[col] = 0
                else:
                    current_month_daily_df[col] = (monthly_total / total_monthly_weight) * current_month_daily_df['Weight']
            
            current_month_daily_df['Tahun'] = year
            current_month_daily_df['Bulan'] = current_month_name
            current_month_daily_df['Month_Num'] = current_month_num

            current_month_daily_df.drop(columns=['Weight', 'Hari'], inplace=True)

            current_year_daily_data = pd.concat([current_year_daily_data, current_month_daily_df], ignore_index=True)
            total_months_processed_in_year += 1
        
        all_years_daily_data = pd.concat([all_years_daily_data, current_year_daily_data], ignore_index=True)

    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {excel_file_path}")
        break
    except KeyError as ke:
        print(f"  [ERROR] Sheet '{sheet_name}' tidak ditemukan di file Excel. Melewati tahun {year}. Detail: {ke}")
        continue
    except Exception as e:
        print(f"  [ERROR] Terjadi kesalahan saat memproses tahun {year}: {e}")
        continue


## Memastikan Kolom 'Tanggal' Berformat Tanggal di Excel
# --- Tampilkan ringkasan data akhir ---
print("\n--- Ringkasan Data Harian Gabungan untuk Semua Tahun (2019-2024) ---")
print(all_years_daily_data)

# --- Menghilangkan Kolom Sebelum Disimpan ---
columns_to_drop = ['Tahun', 'Bulan', 'Month_Num']
final_daily_data_for_export = all_years_daily_data.drop(columns=columns_to_drop, errors='ignore').copy()

# Simpan DataFrame ke file Excel dengan format tanggal
try:
    output_excel_path = r"/content/drive/MyDrive/prediksi/AiTesis/0.Data/data_time_series_2019_2024.xlsx"
    
    # Membuat objek ExcelWriter
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter', datetime_format='dd-mm-yyyy') as writer:
        # Menulis DataFrame ke sheet pertama
        final_daily_data_for_export.to_excel(writer, sheet_name='Data Harian', index=False)
    print(f"\nData harian gabungan berhasil disimpan.")
except Exception as e:
    print(f"\n[ERROR] Gagal menyimpan data ke file Excel: {e}")