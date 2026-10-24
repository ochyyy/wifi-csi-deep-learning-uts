# Hasil dari notebook asli

File di folder ini **disalin dari output notebook** (`notebooks/Project_UTS_PML_original.ipynb`),
bukan hasil eksekusi ulang script repository.

- `grafik_training_validasi.png` – kurva training loss dan Macro-F1 validasi
- `perbandingan_model.png` – Macro-F1 validasi seluruh eksperimen
- `perbandingan_validasi.csv` / `.md` – tabel 13 konfigurasi (`.md` dibuat dari `.csv` lewat `python -m src.visualize compare --from-csv`)
- `confusion_matrix_test_final.png` – confusion matrix 27 aktivitas pada S09–S10
- `perbandingan_test.png`, `perbandingan_test_awal_tuning.csv` – CNN-LSTM awal vs CNN-LSTM + noise di test
- `metrik_test_final.csv` – metrik test model terpilih
- `contoh_sinyal.png` – contoh sinyal CSI hasil preprocessing

Untuk mereproduksi angka ini dibutuhkan dataset MM-Fi dan checkpoint asli, yang tidak disertakan.
