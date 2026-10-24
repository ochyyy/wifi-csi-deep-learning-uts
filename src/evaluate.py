"""Evaluasi checkpoint model WiFi CSI: metrik, prediksi, confusion matrix."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, ConfusionMatrixDisplay)
from .config import N_CLASS, BATCH_SIZE, DEFAULT_DATA, DEFAULT_RESULTS, get_device
from .data_loader import CSIData
from .models import CSIModel


def load_checkpoint(path, data, device):
    path = Path(path)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if 'config' in ckpt:
        cfg = dict(ckpt['config'])
        cfg.setdefault('mode', 'amplitude_norm')
        mode = cfg['mode']
        use_lstm = cfg['use_lstm']
        dropout = cfg.get('dropout', .3)
    else:
        name = path.name.replace('_best.pth', '')
        mode = ckpt.get('mode') or ('amplitude_norm' if 'amp_sample_norm' in name
                                   else ('amplitude' if 'amp_only' in name else 'full'))
        use_lstm = ckpt['use_lstm']
        dropout = .3
        cfg = {'mode': mode, 'use_lstm': use_lstm, 'dropout': dropout}
    source = ckpt.get('source_npz', ckpt.get('preprocessing_file', data.path.name))
    assert source == data.path.name, 'Checkpoint menggunakan file cache berbeda'
    model = CSIModel(use_lstm=use_lstm, dropout=dropout).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, cfg, ckpt


def predict_model(path, split, data, device, batch_size=BATCH_SIZE):
    model, cfg, ckpt = load_checkpoint(path, data, device)
    dataset, ids = data.prepare_split(split, cfg['mode'])
    preds, probs = [], []
    with torch.inference_mode():
        for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits = model(x.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            probs.extend(logits.softmax(1).max(1).values.cpu().tolist())
    return dataset.tensors[1].numpy(), np.asarray(preds), ids, np.asarray(probs), ckpt


def model_name(path):
    return Path(path).stem.removesuffix('_best')


def evaluate(ckpt_path, data_path, out_dir, split='test', compare_checkpoint=None,
             label=None, compare_label=None):
    data = CSIData(data_path)
    device = get_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    true, pred, ids, confidence, ckpt = predict_model(ckpt_path, split, data, device)
    labels = list(range(N_CLASS))
    names = [f'A{i:02d}' for i in range(1, N_CLASS+1)]
    metrics = {
        'model': label or model_name(ckpt_path),
        'epoch': int(ckpt['epoch']),
        'jumlah_sampel': int(len(true)),
        'accuracy': float(accuracy_score(true, pred)),
        'macro_precision': float(precision_score(true, pred, labels=labels, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(true, pred, labels=labels, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(true, pred, labels=labels, average='macro', zero_division=0)),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if split == 'val' and 'val_macro_f1' in ckpt:
        # Setara sel 32 notebook: skor validasi ulang harus sama dengan skor tersimpan di checkpoint.
        gap = abs(metrics['macro_f1'] - float(ckpt['val_macro_f1']))
        print('Cek konsistensi Macro-F1 validasi vs checkpoint:',
              'OK' if gap < 1e-5 else f'PERINGATAN selisih {gap:.2e} (beda perangkat/seed?)')
    report = classification_report(true, pred, labels=labels, target_names=names, zero_division=0)
    result = pd.DataFrame({'sample_id': ids, 'true': true, 'pred': pred, 'confidence': confidence})
    result['subject'] = result.sample_id.str.split('_').str[1]
    result.to_csv(out_dir / f'prediksi_{split}.csv', index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / f'metrik_{split}.csv', index=False)
    (out_dir / f'classification_report_{split}.txt').write_text(report,encoding='utf-8')
    for subject, group in result.groupby('subject'):
        acc = accuracy_score(group.true, group.pred)
        f1 = f1_score(group.true, group.pred, labels=labels, average='macro', zero_division=0)
        print(f'{subject}: accuracy={acc:.2%} | macro-F1={f1:.2%}')
    fig, ax = plt.subplots(figsize=(12, 11))
    ConfusionMatrixDisplay.from_predictions(true, pred, labels=labels,
        display_labels=names, ax=ax, xticks_rotation=90, colorbar=False,
        include_values=False)
    ax.set_title(f'Confusion matrix {split} — {metrics["model"]}')
    fig.tight_layout()
    fig.savefig(out_dir / f'confusion_matrix_{split}.png', dpi=180)
    plt.close(fig)

    if compare_checkpoint:
        old_true, old_pred, _, _, _ = predict_model(compare_checkpoint, split, data, device)
        if not np.array_equal(old_true, true):
            raise ValueError('Urutan label test dua model tidak sama')
        comp = pd.DataFrame([
            {'model': compare_label or model_name(compare_checkpoint),
             'accuracy': accuracy_score(old_true, old_pred),
             'macro_f1': f1_score(old_true, old_pred, labels=labels, average='macro', zero_division=0)},
            {'model': metrics['model'], 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1']},
        ])
        comp.to_csv(out_dir / f'perbandingan_{split}.csv', index=False)
        ax = comp.set_index('model')[['accuracy', 'macro_f1']].plot.bar(figsize=(9, 4))
        ax.set(ylabel='Skor (0–1)', title=f'Perbandingan {split}')
        plt.xticks(rotation=12)
        plt.tight_layout()
        plt.savefig(out_dir / f'perbandingan_{split}.png', dpi=160)
        plt.close()
    return metrics


def main():
    ap = argparse.ArgumentParser(description='Evaluate model UTS on val/test data')
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_RESULTS)
    ap.add_argument('--compare-checkpoint', type=Path)
    ap.add_argument('--label', help='Nama model pada tabel/judul, contoh "CNN-LSTM + noise"')
    ap.add_argument('--compare-label', help='Nama model pembanding, contoh "CNN-LSTM awal"')
    args = ap.parse_args()
    evaluate(args.checkpoint, args.data, args.out_dir, args.split, args.compare_checkpoint,
             args.label, args.compare_label)

if __name__ == '__main__':
    main()
