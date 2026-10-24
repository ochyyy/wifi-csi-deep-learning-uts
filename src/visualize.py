"""Visualisasi dan tabel perbandingan (sel 12, 26, 28 notebook asli).

Subperintah:
  signal   plot contoh sinyal CSI hasil preprocessing  (notebook sel 12)
  compare  tabel + grafik batang perbandingan validasi (notebook sel 26, 28)
  curves   kurva training loss dan Macro-F1 validasi   (notebook sel 28)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import SEED, DEFAULT_DATA, DEFAULT_CHECKPOINTS, DEFAULT_RESULTS
from .experiments import MAIN_EXPERIMENTS, TUNING_EXPERIMENTS, SEED_EXPERIMENTS

CURVE_MODELS = [('1D_CNN', 'main'), ('CNN_LSTM', 'main'),
                ('CNN_LSTM_amp_sample_norm', 'main'),
                ('LSTM_noise003', 'tuning'),
                ('LSTM_noise003_fixedseed2026', 'tuning')]


def plot_signal(data_path, out_path, index=0, channels=(0, 30, 90, 150, 240, 330)):
    from .data_loader import CSIData
    data = CSIData(data_path)
    sample = data.X_all[index]
    fig, ax = plt.subplots(figsize=(9, 3))
    for ch in channels:
        ax.plot(sample[ch], label=f'Kanal {ch + 1}', alpha=.8)
    ax.set(xlabel='Langkah waktu', ylabel='Amplitude ternormalisasi',
           title=f'Contoh input {data.sample_ids[index]}')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def collect_validation(ckpt_dir):
    """Ringkas epoch terbaik (Macro-F1 validasi) tiap eksperimen dari *_history.csv."""
    ckpt_dir = Path(ckpt_dir)
    plan = [(c, 'main', 'awal') for c in MAIN_EXPERIMENTS]
    plan += [(c, 'tuning', 'tuning') for c in TUNING_EXPERIMENTS + SEED_EXPERIMENTS]
    rows = []
    for cfg, sub, stage in plan:
        path = ckpt_dir / sub / f"{cfg['name']}_history.csv"
        if not path.is_file():
            continue
        h = pd.read_csv(path)
        best = h.loc[h.val_f1.idxmax()]
        rows.append({'model': cfg['name'], 'tahap': stage,
                     'seed': cfg.get('seed', SEED), 'epoch': int(best.epoch),
                     'val_accuracy': float(best.val_acc),
                     'val_macro_f1': float(best.val_f1)})
    if not rows:
        raise FileNotFoundError(f'Tidak ada *_history.csv di {ckpt_dir}/main atau /tuning')
    return pd.DataFrame(rows).sort_values('val_macro_f1', ascending=False)


def compare(ckpt_dir, out_dir, from_csv=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = (pd.read_csv(from_csv).sort_values('val_macro_f1', ascending=False)
             if from_csv else collect_validation(ckpt_dir))
    table.to_csv(out_dir / 'perbandingan_validasi.csv', index=False)
    fmt = table.copy()
    fmt['val_accuracy'] = fmt.val_accuracy.map('{:.2%}'.format)
    fmt['val_macro_f1'] = fmt.val_macro_f1.map('{:.2%}'.format)
    header = '| ' + ' | '.join(fmt.columns) + ' |\n|' + '---|' * len(fmt.columns) + '\n'
    body = ''.join('| ' + ' | '.join(map(str, r)) + ' |\n' for r in fmt.itertuples(index=False))
    (out_dir / 'perbandingan_validasi.md').write_text(header + body, encoding='utf-8')
    ax = table.sort_values('val_macro_f1').plot.barh(
        x='model', y='val_macro_f1', figsize=(9, 6), legend=False)
    ax.set(xlabel='Macro-F1 validasi', ylabel='', title='Perbandingan eksperimen')
    plt.tight_layout()
    plt.savefig(out_dir / 'perbandingan_model.png', dpi=160)
    plt.close()
    print(header + body)
    return table


def curves(ckpt_dir, out_path):
    ckpt_dir = Path(ckpt_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    drawn = 0
    for name, sub in CURVE_MODELS:
        path = ckpt_dir / sub / f'{name}_history.csv'
        if not path.is_file():
            continue
        h = pd.read_csv(path)
        axes[0].plot(h.epoch, h.train_loss, label=name)
        axes[1].plot(h.epoch, h.val_f1, label=name)
        drawn += 1
    if not drawn:
        raise FileNotFoundError(f'Tidak ada history yang cocok di {ckpt_dir}')
    axes[0].set(xlabel='Epoch', ylabel='Training loss', title='Training loss')
    axes[1].set(xlabel='Epoch', ylabel='Macro-F1', title='Validasi S07–S08')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('signal')
    s.add_argument('--data', type=Path, default=DEFAULT_DATA)
    s.add_argument('--index', type=int, default=0)
    s.add_argument('--out', type=Path, default=DEFAULT_RESULTS / 'contoh_sinyal.png')
    c = sub.add_parser('compare')
    c.add_argument('--ckpt-dir', type=Path, default=DEFAULT_CHECKPOINTS)
    c.add_argument('--from-csv', type=Path,
                   help='Pakai tabel yang sudah ada, contoh results/perbandingan_validasi.csv')
    c.add_argument('--out-dir', type=Path, default=DEFAULT_RESULTS)
    k = sub.add_parser('curves')
    k.add_argument('--ckpt-dir', type=Path, default=DEFAULT_CHECKPOINTS)
    k.add_argument('--out', type=Path, default=DEFAULT_RESULTS / 'grafik_training_validasi.png')
    a = ap.parse_args()
    if a.cmd == 'signal':
        a.out.parent.mkdir(parents=True, exist_ok=True)
        print('Disimpan:', plot_signal(a.data, a.out, a.index))
    elif a.cmd == 'compare':
        compare(a.ckpt_dir, a.out_dir, a.from_csv)
    else:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        print('Disimpan:', curves(a.ckpt_dir, a.out))

if __name__ == '__main__':
    main()
