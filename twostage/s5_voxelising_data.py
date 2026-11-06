# twostage / s5_voxel_data.py - reshaping the layer energy 

import numpy as np
from numpy.lib.format import open_memmap
from common import HERE, DATA

THRESHOLD = 0.01515
LOG_THR = np.log10(THRESHOLD)
N_LAYER, N_VOXEL = 45, 144
CHUNK = 5000
rng = np.random.default_rng(0)

# 1. Preprocessed energies (N, 6480) into voxel configurations (N, 45, 144) ──────────────────────────
for split in ['train', 'test']:
    y = open_memmap(DATA / f'{split}_voxel.npy', mode='r')
    E_inc = np.load(DATA / f'{split}_energies.npy')
    N = len(E_inc)
    out = open_memmap(HERE / f'{split}_voxelised.npy', mode='w+', dtype=np.float32,
                      shape=(N, N_LAYER, N_VOXEL))
    for s in range(0, N, CHUNK):
        E = (np.expm1(y[s:s+CHUNK]) / 1e4) * E_inc[s:s+CHUNK]       # Now true energy it is.
        out[s:s+CHUNK] = E.reshape(-1, N_LAYER, N_VOXEL).astype(np.float32)
    out.flush()
    print(split, "Saved", out.shape)

# 2. Standardised statistics (from the voxelised train shower *above*, nonempty) ─────────────────────────
Ev = np.load(HERE / 'train_voxelised.npy', mmap_mode='r')
S = np.zeros((N_LAYER, N_VOXEL))
S2 = np.zeros((N_LAYER, N_VOXEL))
cnt = np.zeros((N_LAYER, 1))
for s in range(0, len(Ev), CHUNK):
    E = np.asarray(Ev[s:s+CHUNK], dtype=np.float64)     # E ~ (n, 45, 144)
    El = E.sum(axis=-1)                                 # El ~ (n, 45) >> energy sum over each layer
    x = np.where(E > 0, np.log10(np.maximum(E, THRESHOLD)), 
                 LOG_THR - rng.random(size=E.shape))    # x ~ (n, 45, 144) 
    m = (El > 0)[:, :, None].astype(np.float64)         # m ~ (n, 45, 1) boolean array >> Putting "None" opens an axis.
    S += (x * m).sum(axis=0)                            # x * m ~ (n, 45, 144) broadcasted.
    S2 += (x**2 * m).sum(axis=0)                        ## S was defined to bring us statistics.
    cnt += m.sum(axis=0)

v_mean = S / cnt
v_std = np.sqrt(np.maximum(S2/cnt - v_mean**2, 1e-12))
np.savez(HERE / 'flow2_norm.npz', v_mean=v_mean.astype(np.float32), v_std=v_std.astype(np.float32),
         THRESHOLD=THRESHOLD)

# 3. Sanity check ──────────────────────────────────────────────────────────────────────────────────────
El_check = np.load(HERE / 'train_Elayer.npy')
print("max(|sum(Evox) - Elayer|):", np.abs(np.asarray(Ev[:CHUNK]).sum(-1) - El_check[:CHUNK]).max())
print("non-empty layer fraction:", (cnt[:, 0] / len(Ev)).round(3)[[0, 10, 20, 30, 44]])
print("v_std range:", v_std.min().round(3), "~", v_std.max().round(3))