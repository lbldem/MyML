import h5py
import numpy as np
from numpy.lib.format import open_memmap

THRESHOLD_MEV = 0.01515
CHUNK = 10000

def preprocess(showers, energies):
    showers = showers.copy()
    showers[showers < THRESHOLD_MEV] = 0.0
    return np.log1p(showers / energies * 1e4).astype(np.float32)

def convert(src, dst_x, dst_e):
    with h5py.File(src, 'r') as f:
        n, d = f['showers'].shape      # (100000, 6480)

        X = open_memmap(dst_x, mode='w+', dtype=np.float32, shape=(n, d))
        E = open_memmap(dst_e, mode='w+', dtype=np.float32, shape=(n, 1))

        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            e = f['incident_energies'][start:end]
            s = f['showers'][start:end]
            X[start:end] = preprocess(s, e)
            E[start:end] = e
            print(f"{src}: {end}/{n}")
        
        X.flush(); E.flush()

convert('dataset_2_1.hdf5', 'train_voxel.npy', 'train_energies.npy')
convert('dataset_2_2.hdf5', 'test_voxel.npy', 'test_energies.npy')
print("Saved.")