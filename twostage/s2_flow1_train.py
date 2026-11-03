# twostage/s2_flow1_train.py - training NF model for layer energies.
import copy
# from pathlib import Path
from common import HERE, DATA, device
import numpy as np
import torch
from s2_flow1_model import build_flow1

THRESHOLD = 0.01515
torch.manual_seed(0)
rng = np.random.default_rng(0)

# Data and condition setting ──────────────────────────────────────────────────────────────────────────────────
El = np.load(HERE / 'train_Elayer.npy')             # (100000, 45)
E_inc = np.load(DATA / 'train_energies.npy')        # (100000, 1)
logE = np.log(E_inc)
logE_min, logE_max = logE.min(), logE.max()
c = ((logE - logE_min) / (logE_max - logE_min)).astype(np.float32)

# Preprocess: dequantization + log + normalise ──────────────────────────────────────────────────────────────────
def to_x(E):
    noise = THRESHOLD * (1.0 - rng.random(size=E.shape))
    return np.log10(E + noise).astype(np.float32)

x0 = to_x(El)
x_mean = np.mean(x0, axis=0, keepdims=True).astype(np.float32)            # mean computed over events >> (1, 45)
x_std = np.std(x0, axis=0, keepdims=True).astype(np.float32)              # std computed over events >> (1, 45)
torch.save({'logE_min': logE_min, 'logE_max': logE_max, 'x_mean': x_mean, 'x_std': x_std, 'THRESHOLD': THRESHOLD}, 
           HERE / 'flow1_norm.pt')

# Splitting datasets: 10% from train dataset to validation ────────────────────────────────────────────────────────
idx = rng.permutation(len(El))                          # len(El) will give 100000, first axis + randomly permutated
n_val = len(El) // 10                                   # = 10,000 = 100,000 / 10
iv, it = idx[:n_val], idx[n_val:]                       # = 10,000 + 90,000
El_t, C_t = torch.tensor(El[it]), torch.tensor(c[it])   # El_t = El[it] = El[idx[n_val:]]   ## 90% training data
xv = torch.tensor((to_x(El[iv]) - x_mean) / x_std).to(device)
cv = torch.tensor(c[iv]).to(device)
xm, xs = torch.tensor(x_mean).to(device), torch.tensor(x_std).to(device)

flow = build_flow1().to(device)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

EPOCHS, BATCH = 100, 512
best_val, best_state = float('inf'), None
for epoch in range(EPOCHS):
    # NF training loop
    flow.train()
    perm = torch.randperm(len(El_t))
    tot = 0.0
    for i in range(0, len(perm), BATCH):
        j = perm[i:i + BATCH]
        Eb, cb = El_t[j].to(device), C_t[j].to(device)                      # Eb is a torch tensor stored in GPU.
        xb = torch.log10(Eb + THRESHOLD * (1.0 - torch.rand_like(Eb)))      # So don't apply np functions on Eb, and we're only giving log10 to x.
        xb = (xb - xm) / (xs)
        loss = -flow.log_prob(xb, context=cb).mean()
        if not torch.isfinite(loss):
            print("infinite loss, batch skipped"); continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=10.0)
        optimizer.step()
        tot += loss.item() * len(j)
    
    # NF evaluating loop - to save best epoch of the model training.
    flow.eval()
    with torch.no_grad():
        val = -flow.log_prob(xv, context=cv).mean().item()
    if val < best_val:
        best_val, best_state = val, copy.deepcopy(flow.state_dict())
    print(f"Epoch {epoch+1:3d} | train NLL = {tot/len(El_t):.3f} | val NLL = {val:.3f}")

torch.save(best_state, HERE / 'flow1.pt')
print(f"Saved best (val NLL = {best_val:.3f})")