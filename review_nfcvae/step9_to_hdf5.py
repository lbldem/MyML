# step9_to_hdf5.py - Making our gen.result(gen_voxel_std.npy) to a compressed hdf5 file.

import numpy as np
import h5py
from common import DATA, OUT

THRESHOLD_MEV = 0.01515
G = np.load(OUT / 'gen_voxel_std.npy', mmap_mode = 'r')

with h5py.File(DATA / 'dataset_2_2.hdf5', 'r') as f:
    E_inc = f['incident_energies'][:]

CHUNK = 10000

with h5py.File(OUT / 'gen_std.hdf5', 'w') as f:
    f.create_dataset('incident_energies', data=E_inc)
    S = f.create_dataset('showers', shape=G.shape, dtype=np.float32)
    for start in range(0, len(G), CHUNK):
        end = min(start + CHUNK, len(G))
        y = np.array(G[start:end])
        E = (np.expm1(y) / 1e4) * E_inc[start:end]
        E[E < THRESHOLD_MEV] = 0.0
        S[start:end] = E

print("saved gen_std.hdf5")