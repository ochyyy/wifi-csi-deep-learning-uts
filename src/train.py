"""Training model dari notebook (cell 16, 18, 20, 22, 24), via CLI."""
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from .config import SEED, N_CLASS, BATCH_SIZE, EPOCHS, PATIENCE, DEFAULT_DATA, DEFAULT_CHECKPOINTS, get_device
from .data_loader import CSIData
from .models import CSIModel
from .experiments import MAIN_EXPERIMENTS, TUNING_EXPERIMENTS, SEED_EXPERIMENTS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_experiment(cfg, folder, data, device, batch_size=BATCH_SIZE,
                     epochs=EPOCHS, patience=PATIENCE, overwrite=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = cfg['name']
    ckpt_path = folder / f'{name}_best.pth'
    hist_path = folder / f'{name}_history.csv'
    if ckpt_path.is_file() and hist_path.is_file() and not overwrite:
        print(name, 'sudah ada, dilewati')
        return pd.read_csv(hist_path)

    seed = cfg.get('seed', SEED)
    set_seed(seed)
    train_ds, _ = data.prepare_split('train', cfg['mode'])
    val_ds, _ = data.prepare_split('val', cfg['mode'])
    rng = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=rng, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=device.type == 'cuda')
    model = CSIModel(cfg['use_lstm'], cfg['dropout']).to(device)
    counts = torch.bincount(train_ds.tensors[1], minlength=N_CLASS).float()
    assert (counts > 0).all()
    weights = counts.sum() / (N_CLASS * counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=cfg['wd'])
    history, best_f1, wait = [], -1.0, 0

    for epoch in range(1, epochs + 1):
        row = {'epoch': epoch}
        for split, loader in (('train', train_loader), ('val', val_loader)):
            training = split == 'train'
            model.train(training)
            total, denom, actual, predicted = 0.0, 0.0, [], []
            with torch.set_grad_enabled(training):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    if training and cfg['noise']:
                        x = x.clone()
                        x[:, :342] += torch.randn_like(x[:, :342]) * cfg['noise']
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits, y)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Loss tidak valid: {name}')
                    if training:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    w = weights[y.cpu()].sum().item()
                    total += loss.item() * w
                    denom += w
                    actual.extend(y.cpu().tolist())
                    predicted.extend(logits.argmax(1).detach().cpu().tolist())
            row[f'{split}_loss'] = total / denom
            row[f'{split}_acc'] = accuracy_score(actual, predicted)
            row[f'{split}_f1'] = f1_score(actual, predicted, labels=range(N_CLASS),
                                          average='macro', zero_division=0)
        history.append(row)
        print(f"{name} {epoch:02d}/{epochs} | loss {row['train_loss']:.3f} | "
              f"val acc {row['val_acc']:.3f} | val F1 {row['val_f1']:.3f}")
        if row['val_f1'] > best_f1:
            best_f1, wait = row['val_f1'], 0
            torch.save({'model_state': {k: v.detach().cpu().clone()
                                        for k, v in model.state_dict().items()},
                        'config': dict(cfg), 'epoch': epoch,
                        'val_macro_f1': best_f1, 'seed': seed,
                        'source_npz': data.path.name}, ckpt_path)
        else:
            wait += 1
        pd.DataFrame(history).to_csv(hist_path, index=False)
        if wait >= patience:
            print('Early stopping:', name)
            break
    return pd.DataFrame(history)


def main():
    ap = argparse.ArgumentParser(description='Train 1D-CNN dan CNN-LSTM dari notebook UTS')
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_CHECKPOINTS)
    ap.add_argument('--group', choices=['main', 'tuning', 'seed', 'all'], default='main')
    ap.add_argument('--experiment', help='Nama konfigurasi spesifik, contoh LSTM_noise003_fixedseed2026')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--patience', type=int, default=PATIENCE)
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    data = CSIData(args.data)
    device = get_device()
    print('Device:', device, '| data:', args.data)
    items = []
    for group, exps in [('main', MAIN_EXPERIMENTS), ('tuning', TUNING_EXPERIMENTS), ('seed', SEED_EXPERIMENTS)]:
        if args.group in ('all', group) or args.experiment:
            items.extend((cfg, 'main' if group == 'main' else 'tuning') for cfg in exps)
    if args.experiment:
        items = [(cfg, folder) for cfg, folder in items if cfg['name'] == args.experiment]
        if not items:
            ap.error(f'Nama eksperimen tidak ditemukan: {args.experiment}')
    for cfg, subfolder in items:
        train_experiment(cfg, args.out_dir / subfolder, data, device,
                         args.batch_size, args.epochs, args.patience, args.overwrite)

if __name__ == '__main__':
    main()
