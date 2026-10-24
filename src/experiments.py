"""13 eksperimen notebook asli (cell 20, 22, 24)."""
MAIN_EXPERIMENTS = [
    dict(name='1D_CNN', use_lstm=False, mode='full', dropout=.3, wd=1e-4, noise=0.),
    dict(name='CNN_LSTM', use_lstm=True, mode='full', dropout=.3, wd=1e-4, noise=0.),
    dict(name='1D_CNN_amp_only', use_lstm=False, mode='amplitude', dropout=.3, wd=1e-4, noise=0.),
    dict(name='CNN_LSTM_amp_only', use_lstm=True, mode='amplitude', dropout=.3, wd=1e-4, noise=0.),
    dict(name='CNN_LSTM_amp_sample_norm', use_lstm=True, mode='amplitude_norm', dropout=.3, wd=1e-4, noise=0.),
]
TUNING_EXPERIMENTS = [
    dict(name='CNN_ampnorm_fair', use_lstm=False, mode='amplitude_norm', dropout=.3, wd=1e-4, noise=0.),
    dict(name='LSTM_dropout050', use_lstm=True, mode='amplitude_norm', dropout=.5, wd=1e-4, noise=0.),
    dict(name='LSTM_wd001', use_lstm=True, mode='amplitude_norm', dropout=.3, wd=1e-3, noise=0.),
    dict(name='LSTM_noise003', use_lstm=True, mode='amplitude_norm', dropout=.3, wd=1e-4, noise=.03),
]
SEED_EXPERIMENTS = [
    dict(name=f'LSTM_{variant}_fixedseed{seed}', use_lstm=True,
         mode='amplitude_norm', dropout=.3, wd=1e-4, noise=noise, seed=seed)
    for seed in (123, 2026)
    for variant, noise in (('baseline', 0.0), ('noise003', 0.03))
]
