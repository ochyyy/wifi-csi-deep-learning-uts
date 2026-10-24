"""Preprocessing CSI MM-Fi E01 (diturunkan dari cell 4, 6, 8 notebook asli).

Input: E01.zip serta MMFi_action_segments.csv dari dataset MM-Fi.
Output: file .npz, manifest, dan metadata segmentasi.

Peringatan: raw_dir adalah lokasi ekstraksi sementara dan akan dihapus pada awal proses,
sesuai perilaku notebook sumber. Gunakan folder khusus yang tidak berisi data penting.
"""
import argparse
from pathlib import Path
import json
import re
import shutil
import zipfile
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, detrend
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
from .config import SPLITS, TIME_STEPS


def run_preprocessing(data_dir: Path, raw_dir: Path, drive_dir: Path):
    data_dir = Path(data_dir).resolve()
    raw_dir = Path(raw_dir).resolve()
    drive_dir = Path(drive_dir).resolve()
    RAW_ZIP = data_dir / 'E01.zip'
    RAW_DIR = raw_dir
    DATA_DIR = data_dir
    DRIVE_DIR = drive_dir
    NPZ = drive_dir / 'processed_E01_v1.npz'
    T = TIME_STEPS
    # Hindari penghapusan folder input/output karena RAW_DIR di-reset pada setiap run.
    if RAW_DIR in (DATA_DIR, DRIVE_DIR) or DATA_DIR in RAW_DIR.parents or DRIVE_DIR in RAW_DIR.parents:
        raise ValueError('raw_dir tidak boleh sama dengan atau berada di dalam folder dataset/output')
    if not RAW_ZIP.is_file():
        raise FileNotFoundError(f'Dataset tidak ada: {RAW_ZIP}')
    if not (DATA_DIR / 'MMFi_action_segments.csv').is_file():
        raise FileNotFoundError(f'Anotasi tidak ada: {DATA_DIR / "MMFi_action_segments.csv"}')
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Ekstraksi dan manifest — sesuai isi notebook
    if RAW_DIR.exists():
        print('Menghapus ekstraksi sementara sebelumnya dari runtime Colab...')
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ZIP) as z:
        csi = [f for f in z.infolist() if not f.is_dir()
               and 'wifi-csi' in f.filename.replace('\\', '/').lower().split('/')
               and f.filename.lower().endswith('.mat')]
        assert csi, 'Tidak ditemukan file WiFi CSI (.mat) pada E01.zip'
        needed = sum(f.file_size for f in csi)
        assert shutil.disk_usage(RAW_DIR).free > needed + 1024**3, 'Ruang Colab tidak cukup untuk ekstraksi'
        print('Mulai ekstraksi:', len(csi), 'file CSI')
        for f in tqdm(csi, desc='Ekstraksi CSI'):
            dest = (RAW_DIR / f.filename).resolve()
            assert dest.is_relative_to(RAW_DIR.resolve()), 'Nama file ZIP tidak aman'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.is_file() or dest.stat().st_size != f.file_size:
                part = dest.with_suffix('.mat.part')
                with z.open(f) as src, part.open('wb') as out:
                    shutil.copyfileobj(src, out)
                part.replace(dest)
    print('File CSI diekstrak:', len(csi))

    root = RAW_DIR / 'E01'
    rows = []
    split_lookup = {s: name for name, subjects in SPLITS.items() for s in subjects}
    for f in root.glob('S*/A*/wifi-csi/frame*.mat'):
        subject, action, _, filename = f.relative_to(root).parts
        rows.append((f.relative_to(RAW_DIR).as_posix(), 'E01', subject, action,
                     int(f.stem.replace('frame', '')), split_lookup[subject]))
    manifest = pd.DataFrame(rows, columns=['relative_path','environment','subject',
                                            'activity','frame','split'])
    manifest = manifest.sort_values(['subject','activity','frame']).reset_index(drop=True)
    assert not manifest.empty, 'Tidak ditemukan CSI hasil ekstraksi'
    assert set(manifest.subject) == set(split_lookup), 'Ada subjek yang belum lengkap'
    for subject, group in manifest.groupby('subject'):
        assert set(group.activity) == {f'A{i:02d}' for i in range(1, 28)}, subject
    manifest.to_csv(DRIVE_DIR / 'manifest_E01_cross_subject_v1.csv', index=False)
    print('Manifest:', len(manifest), 'frame\n', manifest.groupby('split').size())

    # Segmentasi — sesuai isi notebook
    segments = pd.read_csv(DATA_DIR / 'MMFi_action_segments.csv')
    for col in ['Environment', 'Student', 'Action', 'Segments']:
        segments[col] = segments[col].astype(str).str.strip()
    segments = segments.query("Environment == 'E01'")
    assert not segments.duplicated(['Environment','Student','Action']).any(), 'Rekaman ganda'
    frame_map = {key: set(g.frame) for key, g in
                 manifest.groupby(['environment', 'subject', 'activity'])}
    rows, errors = [], []
    for r in tqdm(segments.itertuples(index=False), total=len(segments), desc='Segmentasi aktivitas'):
        key = (r.Environment, r.Student, r.Action)
        available, used = frame_map.get(key, set()), set()
        if not available: errors.append(f'{key}: rekaman tidak ada'); continue
        for num, token in enumerate(r.Segments.split(';'), 1):
            token = token.strip()
            if not token: continue
            match = re.fullmatch(r'(\d+)\s*-\s*(\d+)', token)
            if not match: errors.append(f'{key}: format salah {token}'); continue
            start, end = map(int, match.groups())
            if start > end: errors.append(f'{key}: rentang terbalik {token}'); continue
            indices = set(range(start, end + 1))
            if indices - available: errors.append(f'{key}: frame tidak ada {token}')
            if indices & used: errors.append(f'{key}: overlap {token}')
            used.update(indices)
            split = next(name for name, subjects in SPLITS.items() if r.Student in subjects)
            rows.append((r.Student, r.Action, num, start, end, end-start+1, split))
    if set(frame_map) != set(zip(segments.Environment, segments.Student, segments.Action)):
        errors.append('Daftar rekaman CSV dan manifest tidak sesuai')
    if errors: raise ValueError('Masalah segmentasi: ' + '; '.join(errors[:5]))

    seg = pd.DataFrame(rows, columns=['subject','activity','segment_id','start','end','n_frames','split'])
    excluded = seg[seg.n_frames < 2].copy()
    excluded['reason'] = 'Kurang dari 2 frame untuk pemodelan antarframe'
    samples = seg[seg.n_frames >= 2].copy()
    samples['environment'] = 'E01'
    samples['label'] = samples.activity.str[1:].astype(int) - 1
    samples['sample_id'] = samples.apply(lambda r:
        f"E01_{r.subject}_{r.activity}_seg{r.segment_id:03d}", axis=1)
    samples = samples[['sample_id','environment','subject','activity','label',
                       'segment_id','start','end','n_frames','split']].sort_values(
                       ['subject','activity','segment_id']).reset_index(drop=True)
    assert samples.sample_id.is_unique and samples.groupby('split').label.nunique().eq(27).all()
    assert samples.groupby('subject').split.nunique().eq(1).all()
    samples.to_csv(DRIVE_DIR / 'samples_E01_cross_subject_v1.csv', index=False)
    excluded.to_csv(DRIVE_DIR / 'excluded_segments_E01_v1.csv', index=False)
    print('Jumlah segmen:', len(seg), '| Dipakai:', len(samples), '| Dikeluarkan:', len(excluded))
    print(samples.groupby('split').size())

    # Preprocessing dan simpan cache — sesuai isi notebook
    print('Mulai preprocessing:', len(samples), 'segmen')
    sos = butter(2, 0.4, output='sos')
    X = np.empty((len(samples), 684, T), np.float32)
    repaired = [0]

    def fix_invalid(a):
        a = np.asarray(a, np.float64).copy()
        assert a.shape == (3, 114, 10), f'Shape CSI berbeda: {a.shape}'
        invalid = ~np.isfinite(a)
        repaired[0] += int(invalid.sum())
        for antenna, packet in np.argwhere(invalid.any(axis=1)):
            v = a[antenna, :, packet]
            good = np.flatnonzero(np.isfinite(v))
            if len(good) < 2: raise ValueError('Subcarrier valid terlalu sedikit')
            a[antenna, :, packet] = np.interp(np.arange(114), good, v[good])
        return a

    for (env, subject, action), group in tqdm(
        samples.groupby(['environment','subject','activity']), desc='Preprocessing CSI'):
        files = manifest[(manifest.environment == env) &
                         (manifest.subject == subject) &
                         (manifest.activity == action)]
        needed = {f for r in group.itertuples() for f in range(r.start, r.end + 1)}
        features = {}
        for r in files.itertuples():
            if r.frame not in needed: continue
            d = loadmat(RAW_DIR / r.relative_path, variable_names=['CSIamp','CSIphase'])
            amp = fix_invalid(d['CSIamp'])
            phase = detrend(np.unwrap(fix_invalid(d['CSIphase']), axis=1), axis=1)
            features[r.frame] = np.r_[np.median(amp, axis=-1).ravel(),
                                      np.median(phase, axis=-1).ravel()]
        for idx, r in group.iterrows():
            seq = np.stack([features[f] for f in range(r.start, r.end + 1)], axis=1)
            filtered = sosfiltfilt(sos, seq, axis=1, padlen=min(9, len(seq[0]) - 1))
            X[idx] = interp1d(np.linspace(0, 1, seq.shape[1]), filtered, axis=1)(
                np.linspace(0, 1, T)).astype(np.float32)

    print('Fitur selesai dihitung. Menghitung statistik training...')
    train_X = X[samples.split.eq('train').to_numpy()].astype(np.float64)
    mean = train_X.mean(axis=(0,2)).astype(np.float32)[None, :, None]
    std = train_X.std(axis=(0,2)).astype(np.float32)[None, :, None]
    del train_X
    std = np.maximum(std, 1e-6)
    X = (X - mean) / std
    if not np.isfinite(X).all(): raise ValueError('Preprocessing menghasilkan NaN/Inf')
    config = dict(time_steps=T, features='amplitude_median_then_detrended_phase_median',
                  phase_axis='subcarrier_index',
                  filter='Butterworth_order2_normalized_cutoff0.4',
                  normalization='training_only_per_feature', invalid_values_repaired=repaired[0])
    temp = DRIVE_DIR / 'processed_E01_v1.partial.npz'
    np.savez_compressed(temp, X=X.astype(np.float32),
        y=samples.label.to_numpy(np.int64), split=samples.split.to_numpy(str),
        sample_id=samples.sample_id.to_numpy(str), mean=mean, std=std,
        config=np.array(json.dumps(config)))
    temp.replace(NPZ)
    del X
    print('Preprocessing selesai. Nilai diperbaiki:', repaired[0], '| Simpan:', NPZ)
    return NPZ


def main():
    parser = argparse.ArgumentParser(description='Preprocess MM-Fi E01 WiFi CSI')
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Folder yang berisi E01.zip dan MMFi_action_segments.csv')
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw_tmp'),
                        help='Folder sementara ekstraksi; isinya akan dihapus saat dijalankan')
    parser.add_argument('--out-dir', type=Path, default=Path('data/processed'))
    args = parser.parse_args()
    print('Output:', run_preprocessing(args.data_dir, args.raw_dir, args.out_dir))

if __name__ == '__main__':
    main()
