"""Membuat data SINTETIS berformat MM-Fi E01 untuk uji pipeline (bukan data asli).

Hasil: <out-dir>/E01.zip dan <out-dir>/MMFi_action_segments.csv dengan struktur
E01/S01..S10/A01..A27/wifi-csi/frameNNN.mat (CSIamp, CSIphase berukuran 3x114x10).
Tiap kelas diberi pola sinyal berbeda sehingga model bisa belajar; sebagian kecil
nilai sengaja dibuat NaN agar langkah perbaikan nilai tidak valid ikut teruji.

Skor model pada data ini TIDAK mencerminkan hasil pada dataset MM-Fi asli.
"""
import argparse
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat


def make_frame(rng, action, t, n_frames):
    sub = np.arange(114)[None, :, None]
    pk = np.arange(10)[None, None, :]
    ant = np.arange(3)[:, None, None]
    phase_t = 2 * np.pi * t / n_frames
    amp = (20 + 2 * ant + 0.05 * sub
           + (1 + action % 5) * np.sin(phase_t * (1 + action // 5) + 0.1 * sub)
           + rng.normal(0, .5, (3, 114, 10)))
    phase = (0.3 * sub * (1 + 0.02 * action) + np.cos(phase_t * (1 + action % 7)) * (1 + action / 10)
             + 0.05 * pk + rng.normal(0, .1, (3, 114, 10)))
    if rng.random() < .01:  # simulasikan nilai tidak valid
        amp[rng.integers(3), rng.integers(10, 100), rng.integers(10)] = np.nan
    return amp.astype(np.float32), phase.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=Path('data/demo_input'))
    ap.add_argument('--frames', type=int, default=8, help='frame per rekaman (genap, >=4)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    half = a.frames // 2
    rows = []
    with zipfile.ZipFile(a.out_dir / 'E01.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for s in range(1, 11):
            for act in range(1, 28):
                for f in range(1, a.frames + 1):
                    amp, ph = make_frame(rng, act, f, a.frames)
                    buf = io.BytesIO()
                    savemat(buf, {'CSIamp': amp, 'CSIphase': ph})
                    z.writestr(f'E01/S{s:02d}/A{act:02d}/wifi-csi/frame{f:03d}.mat', buf.getvalue())
                segs = f'1-{half};{half + 1}-{a.frames}'
                rows.append(('E01', f'S{s:02d}', f'A{act:02d}', segs))
    pd.DataFrame(rows, columns=['Environment', 'Student', 'Action', 'Segments']).to_csv(
        a.out_dir / 'MMFi_action_segments.csv', index=False)
    print('Data demo dibuat di', a.out_dir, '|', len(rows) * a.frames, 'frame')

if __name__ == '__main__':
    main()
