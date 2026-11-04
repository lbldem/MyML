# twostage / s4_flow1_rescale.py

import h5py
import numpy as np
from common import HERE, DATA

THRESHOLD = 0.01515
N_LAYER, N_VOXEL = 45, 144
CHUNK = 5000

src = h5py.File(HERE.parent / 'review_nfcvae' / 'gen_std.hdf5', 'r')
E_inc = src['incident_energies'][:]                             # (N, 1)
El_flow = np.load(HERE / 'gen_Elayer.npy')

E_test = np.load(DATA / 'test_energies.npy')
assert len(E_inc) == len(El_flow) and np.allclose(E_inc[:, 0], E_test[:, 0]), "E_inc order misaligned"
N = len(E_inc)

n_lost, E_lost = 0, 0.0
with h5py.File(HERE / 'gen_rescaled.hdf5', 'w') as dst:
    dst.create_dataset('incident_energies', data=E_inc)
    out = dst.create_dataset('showers', shape=(N, N_LAYER * N_VOXEL), dtype='float32')
    for s in range(0, N, CHUNK):
        V = src['showers'][s:s+CHUNK].astype(np.float32).reshape(-1, N_LAYER, N_VOXEL)
        Ec = V.sum(axis=-1)         # CVAE layer energy ~ (n, 45)
        Ef = El_flow[s:s+CHUNK]     # Flow1 layer energy ~ (n, 45)

        w = np.divide(Ef, Ec, out=np.zeros_like(Ec), where=(Ec !=0))    # ~ (n, 45)
        V = V * w[:, :, None]       # (n, 45, 144) * (n, 45, 1) = (n, 45, 144)
        V[V < THRESHOLD] = 0.0

        lost = (Ec == 0) & (Ef > 0)     # Array of T/F given the condition.
        n_lost += lost.sum(); E_lost += Ef[lost].sum()
        out[s:s+CHUNK] = V.reshape(-1, N_LAYER * N_VOXEL)
        print(f"{min(s+CHUNK, N)}/{N}", end='\r')

print(f"\nlost layers: {n_lost} ({n_lost / (N * N_LAYER):.4%}),"
      f"    lost energy fraction: {E_lost / El_flow.sum():.2e}")
