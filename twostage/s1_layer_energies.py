# twostage/s1_layer_energies.py - train/test's layer energies (100000, 45) are saved as MeV numpy arrays.

from pathlib import Path
import numpy as np
from common import HERE, OUT, DATA

N_LAYER, N_VOX = 45, 144
CHUNK = 10000

def layer_energies(split):
    Y = np.load(DATA / f'{split}_voxel.npy', mmap_mode='r')     # Y ~ (100000, 6480) >> len(Y) = 100000 (default: dim=0)
    E_inc = np.load(DATA / f'{split}_energies.npy') 
    out = np.zeros((len(Y), N_LAYER), dtype=np.float32)         # zeroes ~ (100000, 45)
    for start in range(0, len(Y), CHUNK):
        end = min(start + CHUNK, len(Y))
        y = np.array(Y[start:end])                              # i.e. start=0 > y = Y[0:10000] ~ (10000, 6480)
        E = (np.expm1(y) / 1e4) * E_inc[start:end]
        E = E.reshape(-1, N_LAYER, N_VOX)
        out[start:end] = np.sum(E, axis=-1)
    return out, E_inc

for split in ['train', 'test']:
    El, E_inc = layer_energies(split)
    np.save(HERE / f'{split}_Elayer.npy', El)
    print(split, El.shape, "| E_tot/E_inc mean = %.3f" % (El.sum(1) / E_inc[:, 0]).mean(),
          "| zero-layer fraction = %.3f" % (El == 0).mean())