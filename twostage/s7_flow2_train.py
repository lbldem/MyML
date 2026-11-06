# twostage / s7_flow2_train.py - train Flow II on (event, layer) pairs with teacher forcing.

import copy
import numpy as np
import torch
from common import HERE, DATA, device
from s6_flow2_model import build_flow2, N_LAYER, N_VOXEL

torch.manual_seed(0)
rng = np.random.default_rng(0)

# 1. Loading data ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Ev = np.load(HERE / 'train_voxelised.npy')                      # real, voxelised shower (N, 45, 144) MeV
El = np.load(HERE / 'train_Elayer.npy')                         # real, **layerised shower** (N, 45) MeV >> total energy of each layers
E_inc = np.load(DATA / 'train_energies.npy')[:, 0]              # real, incident energy  (N, ) MeV
norm = np.load(HERE / 'flow2_norm.npz')                         # normalisation of standardised voxelised data.
v_mean, v_std = norm['v_mean'], norm['v_std']                   # (45, 144), (45, 144)
THRESHOLD = float(norm['THRESHOLD'])                            # 0.01515 MeV
LOG_THR = np.log10(THRESHOLD)                                   # log10(0.01515) = -1.82
f1 = torch.load(HERE / 'flow1_norm.pt', weights_only=False)
c_inc = ((np.log(E_inc) - f1['logE_min']) / (f1['logE_max'] - f1['logE_min'])).astype(np.float32)
                                                                # normalised incident energy (N, )

# 2. (Event, layer) pair + train/evaluation split ─────────────────────────────────────────────────────── # M = number of nonempty (event, layer) pairs. ──────────────────────────────────────────────────────────────────────────────
pairs = np.argwhere(El > 0)                                     # (M, 2) >> np.argwhere will give you (i, j) of matrix elements satisfying condition
ev_perm = rng.permutation(len(El))                              # (N, ) >> permutating integers of range [0, len(El))
val_ev = np.zeros(len(El), bool)                                # (N, ) bool array of zeros of len(El)
val_ev[ev_perm[:len(El)//10]] = True                            # First 10% of pairs are for evaluation.
is_val = val_ev[pairs[:, 0]]                                    # (M, ) bool >> pairs[:, 0] is an event number. val_ev[pairs[:, 0]] will give you True-array. (evaluation)
tr_pairs, va_pairs = pairs[~is_val], pairs[is_val]              # (0.9M, 2), (0.1M, 2) >> 0 for training, 1 for evaluation.
print("Pairs: train", len(tr_pairs), "| val", len(va_pairs))    

# 3. c_El standardised statistics for train pairs ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
r = np.log10(El[tr_pairs[:, 0], tr_pairs[:, 1]] / E_inc[tr_pairs[:, 0]])
                                                                # (number of train pairs, ) >> shower E ratio
r_mean, r_std = r.mean(), r.std()
np.savez(HERE / 'flow2_ctx.npz', r_mean=r_mean, r_std=r_std)
ONEHOT = np.eye(N_LAYER, dtype=np.float32)                      # np.eye gives you a square identity matrix.

def to_x(E):
    return np.where(E > 0, np.log10(np.maximum(E, THRESHOLD)),
                    LOG_THR - rng.random(size=E.shape)).astype(np.float32)

def make_batch(p):
    i, l = p[:, 0], p[:, 1]                                                             # (B, ), (B, ) >> i = #event, l = #layer
    x = (to_x(Ev[i, l]) - v_mean[l]) / v_std[l]                                         # (B, 144) >> standardised answer shower we need to learn 
    prev = np.zeros((len(p), N_VOXEL), np.float32)                                      # (B, 144) >> previous layer initialised. 
    has = l > 0                                                                         # (B, ) bool >> has a previous layer?
    lp = l[has] - 1                                                                     # (#Trues in has, ) >> indices of previous layers
    prev[has] = (to_x(Ev[i[has], lp]) - v_mean[lp]) / v_std[lp]                         # (#Trues in has, 144)>> standardised shower of previous layers
    c_el = ((np.log10(El[i, l] / E_inc[i]) - r_mean) / r_std).astype(np.float32)        # (B, ) >> context for a layer's energy El, standardised.
    ctx = np.concatenate((c_inc[i][:, None], c_el[:, None], ONEHOT[l], prev), axis=-1)  # None adds you an axis.
    return torch.from_numpy(x).to(device), torch.from_numpy(ctx).to(device)             # (B, 144), (B, 191) batched tensors

va_sub = va_pairs[rng.choice(len(va_pairs), 20000, replace=False)]                      # (20000, 2) >> random subset of evaluation pairs
xv, cv = make_batch(va_sub)                                                             # (20000, 144), (20000, 191) >> x, context for evaluation.

# 4. Training ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
flow = build_flow2().to(device)
opt = torch.optim.Adam(flow.parameters(), lr=5e-4)
EPOCHS, STEPS, BATCH = 60, 1000, 512
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
best_val, best_state = float('inf'), None

for epoch in range(EPOCHS):
    flow.train(); tot, n = 0.0, 0
    for step in range(STEPS):
        p = tr_pairs[rng.integers(0, len(tr_pairs), BATCH)]                             # (512, 2) >> training pair picked up at this step. (sampled with replacements)
        xb, cb = make_batch(p)                                                          # (512, 144), (512, 191) >> x, context for the batch
        loss = -flow.log_prob(xb, context=cb).mean()
        if not torch.isfinite(loss):
            print("infinite loss, skipped a batch"); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=10.0)
        opt.step()
        tot += loss.item(); n += 1
    sched.step()
    flow.eval()
    with torch.no_grad():
        val = torch.cat([-flow.log_prob(xv[s:s+2000], context=cv[s:s+2000])
                         for s in range(0, len(xv), 2000)]).mean().item()
    if val < best_val:
        best_val, best_state = val, copy.deepcopy(flow.state_dict())
        torch.save(best_state, HERE / 'flow2.pt')
    print(f"Epoch {epoch+1:3d} | train NLL {tot/max(n,1):.2f} | val NLL {val:.2f} | lr {sched.get_last_lr()[0]:.1e}")

print(f"best val NLL {best_val:.2f}  → flow2.pt")