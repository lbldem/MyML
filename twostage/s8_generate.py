# twostage / s8_generate.py - Flow1 layer energies + Flow2 layer-wise autoregressive shapes to hdf5.

import numpy as np
import torch
import h5py
from common import HERE, DATA, device
from s6_flow2_model import build_flow2, N_LAYER, N_VOXEL

torch.manual_seed(2)

# 1. Preparations ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
norm = np.load(HERE / 'flow2_norm.npz')
vm = torch.tensor(norm['v_mean'], device=device); vs = torch.tensor(norm['v_std'], device=device)
                                                                            # v are from voxelised shower Energies. (45, 144)
THRESHOLD = float(norm['THRESHOLD']); LOG_THR = float(np.log10(THRESHOLD))  
ctxn = np.load(HERE / 'flow2_ctx.npz')                                      # layer energy from training pairs as a context, but ratio picked up normalised and saved as a dictionary of single numbers.
r_mean, r_std = float(ctxn['r_mean']), float(ctxn['r_std'])                 
f1 = torch.load(HERE / 'flow1_norm.pt', weights_only=False)                 # Incident energy normalisation 
E_inc = np.load(DATA / 'test_energies.npy')[:, 0]                           # E_inc ~ (N, )
El_gen = np.load(HERE / 'gen_Elayer.npy')                                   # El_gen ~ (N, 45) from RQS N1 flow.
assert len(E_inc) == len(El_gen)
c_inc = ((np.log(E_inc) - f1['logE_min']) / (f1['logE_max'] - f1['logE_min'])).astype(np.float32)

# 2. Model loading ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
flow = build_flow2().to(device)
sd = torch.load(HERE / 'flow2.pt', map_location=device)
flow.load_state_dict(sd['state'] if 'state' in sd else sd)
flow.eval()
ONEHOT = torch.eye(N_LAYER, device=device)

def to_x_t(E):
    return torch.where(E > 0, torch.log10(E.clamp_min(THRESHOLD)), LOG_THR - torch.rand_like(E))

# 3. Generating ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
N, CHUNK = len(E_inc), 10000
n_lost = 0
with h5py.File(HERE / 'gen_flow2.hdf5', 'w') as f:
    f.create_dataset('incident_energies', data=E_inc[:, None].astype(np.float32))
    out = f.create_dataset('showers', shape=(N, N_LAYER * N_VOXEL), dtype='float32')

    for s in range(0, N, CHUNK):
        Ei = torch.tensor(E_inc[s:s+CHUNK], device=device, dtype=torch.float32)
        ci = torch.tensor(c_inc[s:s+CHUNK], device=device)
        El = torch.tensor(El_gen[s:s+CHUNK], device=device)         # (n, 45) >> This was from Flow1.
        n = len(Ei)
        shower = torch.zeros(n, N_LAYER, N_VOXEL, device=device)
        prev = torch.zeros(n, N_VOXEL, device=device)

        with torch.no_grad():
            for l in range(N_LAYER):
                m = El[:, l] > 0        # (n, ) >> Boolean tensor of layer energy emptiness checking. We can later use this as an ordinary array index as usual.
                k = int(m.sum())        # k = nonempty event number k = m.sum()
                if k > 0:
                    c_el = (torch.log10(El[m, l] / Ei[m]) - r_mean) / r_std                                         # c_el already chosen in event m.
                    ctx = torch.cat([ci[m][:, None], c_el[:, None], ONEHOT[l].expand(k, -1), prev[m]], dim=-1)      # torch.cat([(k, 1), (k, 1), (k, 45), (k, 144)], dim=-1)
                    z = flow.sample(1, context=ctx).squeeze(1)  # (k, 144)
                    x = (z * vs[l]) + vm[l]                     # (k, 144)
                    E = 10 ** x                                 # (k, 144)
                    E[E < THRESHOLD] = 0.0
                    tot = E.sum(-1)                             # (k, ) >> total energy in an event-k.
                    scale = torch.where(tot!=0, El[m,l]/tot.clamp_min(1e-12), 0.0)      # scale ~ (k, )
                    E = E * scale[:, None]                                              # E ~ (k, 144) * (k, 1) >> broadcasting 'scale'.
                    E[E < THRESHOLD] = 0.0
                    n_lost += int((tot==0).sum())
                    shower[m, l] = E
                prev = (to_x_t(shower[:, l]) - vm[l]) / vs[l]
        out[s:s+n] = shower.reshape(n, -1).cpu().numpy()
        print(f"{s + n}/{N}", flush=True)

print("Lost layers (Flow 1 > 0 but all voxels under THR):", n_lost, f"({n_lost/(N*N_LAYER):.4%})")