"""Demo inferensi satu sampel sesuai cell 40 notebook."""
import argparse
from pathlib import Path
from .config import N_CLASS, DEFAULT_DATA, get_device
from .data_loader import CSIData
from .evaluate import predict_model


def main():
    ap = argparse.ArgumentParser(description='Demo prediksi aktivitas dari satu sampel test')
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--index', type=int, default=0)
    args = ap.parse_args()
    data = CSIData(args.data)
    true, pred, ids, confidence, _ = predict_model(args.checkpoint, args.split, data, get_device())
    i = args.index
    if not 0 <= i < len(true):
        ap.error(f'Index harus 0 sampai {len(true)-1}')
    names = [f'A{k:02d}' for k in range(1, N_CLASS+1)]
    print('Sampel:', ids[i])
    print('Aktivitas sebenarnya:', names[int(true[i])])
    print('Prediksi model:', names[int(pred[i])])
    print('Skor probabilitas tertinggi:', f'{confidence[i]:.2%}')
    print('Prediksi benar:', bool(true[i] == pred[i]))

if __name__ == '__main__':
    main()
