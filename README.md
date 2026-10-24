# WiFi CSI Human Activity Recognition

Repository project UTS **Pembelajaran Mesin Lanjut** — Kelompok 4.  
Topik: **Cross-Subject Human Activity Recognition** menggunakan dataset **MM-Fi (E01)**.

## Tentang project

Project ini menggunakan sinyal WiFi CSI untuk mengenali 27 aktivitas manusia. Model yang digunakan adalah **1D-CNN** dan **CNN-LSTM** dengan PyTorch.

Pengujian dilakukan secara *cross-subject*: data S01–S06 untuk training, S07–S08 untuk validasi, dan S09–S10 untuk testing. Kami juga mencoba beberapa pengaturan preprocessing dan training, seperti normalisasi, dropout, dan penambahan Gaussian noise.

Kode eksperimen awal ada di [`notebooks/Project_UTS_PML_Kel4.ipynb`](notebooks/Project_UTS_PML_Kel4.ipynb). Versi Python yang lebih terpisah ada di folder `src/`.

## Isi repository

```text
notebooks/        Notebook Google Colab asli
src/              Preprocessing, model, training, evaluasi, dan prediksi
results/          Grafik dan hasil eksperimen dari notebook
train.py          Perintah untuk menjalankan training
requirements.txt  Library Python yang dibutuhkan
README.md         Panduan repository
```

## Cara menjalankan

Gunakan Python 3.10 atau lebih baru. Jalankan perintah dari folder utama repository.

**1. Instal library**

```bash
python -m pip install -r requirements.txt
```

**2. Siapkan dataset**

Letakkan file `E01.zip` dan `MMFi_action_segments.csv` dari dataset MM-Fi di folder `data/input/`:

```text
data/input/
├── E01.zip
└── MMFi_action_segments.csv
```

Dataset tidak disertakan di GitHub. Jika sudah memiliki file hasil preprocessing `processed_E01_v1.npz` dari Colab, simpan di `data/processed/` dan lewati perintah preprocessing berikut.

```bash
python -m src.preprocessing --data-dir data/input --raw-dir data/raw_tmp --out-dir data/processed
```

**3. Training model**

```bash
python train.py --group tuning --experiment CNN_ampnorm_fair
python train.py --group main --experiment CNN_LSTM_amp_sample_norm
python train.py --group seed --experiment LSTM_noise003_fixedseed2026
```

Checkpoint hasil training akan disimpan di `models/checkpoints/`.

**4. Evaluasi dan prediksi**

Contoh menggunakan model CNN-LSTM dengan Gaussian noise:

```bash
python -m src.evaluate --checkpoint models/checkpoints/tuning/LSTM_noise003_fixedseed2026_best.pth --split test
python -m src.predict --checkpoint models/checkpoints/tuning/LSTM_noise003_fixedseed2026_best.pth --index 0
```

Hasil evaluasi baru akan disimpan di `results/generated/`.

### Mencoba kode tanpa dataset asli

Untuk mengecek apakah alur program berjalan, tersedia generator data sintetis:

```bash
python -m src.make_demo_data --out-dir data/demo/input
python -m src.preprocessing --data-dir data/demo/input --raw-dir data/demo/raw_tmp --out-dir data/demo/processed
python train.py --data data/demo/processed/processed_E01_v1.npz --out-dir data/demo/ckpt --experiment CNN_LSTM_amp_sample_norm --epochs 5 --patience 3
python -m src.evaluate --data data/demo/processed/processed_E01_v1.npz --checkpoint data/demo/ckpt/main/CNN_LSTM_amp_sample_norm_best.pth --split test --out-dir data/demo/results
```

Data sintetis hanya untuk mencoba program, **bukan** untuk mengukur performa model pada MM-Fi.

## Hasil singkat

Dari output notebook Colab, **CNN-LSTM** memperoleh Macro-F1 validasi **20,23%**, sedangkan **1D-CNN** memperoleh **17,93%** pada pengaturan perbandingan yang sama. Konfigurasi CNN-LSTM dengan Gaussian noise 0,03 (seed 2026) mencapai **Macro-F1 testing 21,11%** dan **accuracy testing 23,30%**.

Hasil tersebut menunjukkan bahwa model masih kesulitan mengenali aktivitas pada subjek yang berbeda. Grafik, confusion matrix, dan tabel eksperimen tersedia di folder [`results/`](results/), termasuk [perbandingan validasi](results/perbandingan_validasi.md).

Angka di atas diambil dari **hasil notebook Colab asli**. Dataset dan checkpoint model tidak disertakan dalam repository, sehingga hasil tersebut belum dijalankan ulang melalui script modular dengan data asli.

## Anggota Kelompok 4

| Nama | NIM |
|---|---|
| Ochi Diza Inanta | 2311511006 |
| Mifta Riri Amela | 2311511012 |
| Andi Tri Akira | 2311513022 |
| Rintan Amorade | 2311513044 |

## Referensi

- [Awesome-WiFi-CSI-Sensing](https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing)
- Dataset MM-Fi
