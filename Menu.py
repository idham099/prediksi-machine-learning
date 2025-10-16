import os
import sys
import subprocess
import pandas as pd 

def tampilkan_menu():
    print("+=========================================+")
    print("| Aplikasi prediksi Bandara berbasis Ai   |")
    print("|     Karya : Ainul idham (202330001)     |")
    print("+=========================================+")
    print("| 0. Konversi data menjadi time series    |")
    print("|=========================================|")
    print("| Pilih Algoritma :                       |")
    print("| 1. Long Short-Term Memory (LSTM)        |")
    print("| 2. Extreme Gradient Boosting (XGBoost)  |")
    print("| 3. Hipotesis                            |")
    print("+=========================================+")
    print("| 4. Keluar                               |")
    print("+=========================================+")

def jalankan_konversi_data():
    print("+ INFO : Sedang Melakukan konversi data menjadi time series, Mohon menunggu..")
    conversion_script_path = "0.data_konversi.py"
    if os.path.exists(conversion_script_path):
        try:
            subprocess.run([sys.executable, conversion_script_path], check=True, capture_output=False)
            print("\n+ INFO : Konversi data Time Series selesai.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Gagal menjalankan {conversion_script_path}: Proses berakhir dengan kode {e.returncode}")
        except FileNotFoundError:
            print(f"\n[ERROR] Interpreter Python tidak ditemukan untuk menjalankan {conversion_script_path}.")
        except Exception as e:
            print(f"\n[ERROR] Terjadi kesalahan tidak terduga saat menjalankan {conversion_script_path}: {e}")
    else:
        print(f"\n[ERROR] File '{conversion_script_path}' tidak ditemukan di direktori yang sama.")
        print("Pastikan file konversi Anda berada di lokasi yang benar.")

def jalankan_lstm():
    print("+ INFO : Sedang Menghubungkan algoritma LSTM , Mohon menunggu..")
    lstm_script_path = "1.lstm.py"
    if os.path.exists(lstm_script_path):
        try:
            subprocess.run([sys.executable, lstm_script_path], check=True, capture_output=False)
            print("\n+ INFO : Prediksi dengan algoritma LSTM selesai.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Gagal menjalankan {lstm_script_path}: Proses berakhir dengan kode {e.returncode}")
        except FileNotFoundError:
            print(f"\n[ERROR] Interpreter Python tidak ditemukan untuk menjalankan {lstm_script_path}.")
        except Exception as e:
            print(f"\n[ERROR] Terjadi kesalahan tidak terduga saat menjalankan {lstm_script_path}: {e}")
    else:
        print(f"\n[ERROR] File '{lstm_script_path}' tidak ditemukan di direktori yang sama.")
        print("Pastikan file Anda berada di lokasi yang benar.")

def jalankan_xgboost():
    print("+ INFO : Sedang Menghubungkan algoritma XGBoost , Mohon menunggu..")
    xgboost_script_path = "2.xgboost.py"
    if os.path.exists(xgboost_script_path):
        try:
            subprocess.run([sys.executable, xgboost_script_path], check=True, capture_output=False)
            print("\n+ INFO : Prediksi dengan algoritma XGBoost selesai.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Gagal menjalankan {xgboost_script_path}: Proses berakhir dengan kode {e.returncode}")
        except FileNotFoundError:
            print(f"\n[ERROR] Interpreter Python tidak ditemukan untuk menjalankan {xgboost_script_path}.")
        except Exception as e:
            print(f"\n[ERROR] Terjadi kesalahan tidak terduga saat menjalankan {xgboost_script_path}: {e}")
    else:
        print(f"\n[ERROR] File '{xgboost_script_path}' tidak ditemukan di direktori yang sama.")
        print("Pastikan file Anda berada di lokasi yang benar.")

def jalankan_hybrid():
    print("+ INFO : Sedang Melakukan Perhitungan Uji Diebold-Mariano per Kolom Target , Mohon menunggu..")
    hybrid_script_path = "hipotesis.py"
    if os.path.exists(hybrid_script_path):
        try:
            subprocess.run([sys.executable, hybrid_script_path], check=True, capture_output=False)
            print("\n+ INFO : Perhitungan dengan Uji Diebold-Mariano per Kolom Target selesai.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Gagal menjalankan {hybrid_script_path}: Proses berakhir dengan kode {e.returncode}")
        except FileNotFoundError:
            print(f"\n[ERROR] Interpreter Python tidak ditemukan untuk menjalankan {hybrid_script_path}.")
        except Exception as e:
            print(f"\n[ERROR] Terjadi kesalahan tidak terduga saat menjalankan {hybrid_script_path}: {e}")
    else:
        print(f"\n[ERROR] File '{hybrid_script_path}' tidak ditemukan di direktori yang sama.")
        print("Pastikan file Anda berada di lokasi yang benar.")

def main():
    while True: 
        tampilkan_menu()
        pilihan = input("Silahkan pilih [0-4] : ") 
        print(" ")

        if pilihan == '0':
            jalankan_konversi_data()
        elif pilihan == '1':
            jalankan_lstm()
        elif pilihan == '2':
            jalankan_xgboost()
        elif pilihan == '3':
            jalankan_hybrid()
        elif pilihan == '4':
            print("+ INFO : Terima kasih telah menggunakan program ini. Sampai jumpa!!")
            break 
        else:
            print("+ INFO : Pilihan tidak valid, silahkan masukkan 0, 1, 2, 3, atau 4. ")

        if pilihan != '4': 
            input("\nTekan Enter untuk kembali ke menu...")
            
if __name__ == "__main__":
    main()