"""Titik masuk training dari root repository.

Contoh:  python train.py --group main --experiment 1D_CNN
Setara dengan:  python -m src.train ...
"""
from src.train import main

if __name__ == '__main__':
    main()
