import numpy as np
import torch
from calo_cvae_hurdle_model import HurdleCVAE, Z_DIM
from z_prior_model import build_flow
from numpy.lib.format import open_memmap

TAG = 'b01'
MODE = 'nf_prior'   # 'generate', 'reconstruct', 'nf_prior'
THRESHOLD_MEV = 0.01515


# ── Loading model and parameters ───────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device: ", device)
model = HurdleCVAE()
model.load_state_dict(torch.load(f'calo_cvae_hurdle_{TAG}.pt', map_location='cpu'))
model.to(device)
model.eval()
norm = torch.load('norm_params_voxel.pt', weights_only=False)
test_E = np.load('test_energies.npy')
N = len(test_E)


# ── 1) Normalising condition(incident energies) ────────────────────────────
log_E = np.log(test_E)
c = (log_E - norm['log_E_min']) / (norm['log_E_max'] - norm['log_E_min'])
C = torch.tensor(c, dtype=torch.float32).to(device)
y_thr = np.log1p(THRESHOLD_MEV / test_E * 1e4)
Y_thr = torch.tensor(y_thr, dtype=torch.float32).to(device)

G = open_memmap(f'gen_voxel_{TAG}.npy', mode='w+', dtype=np.float32, shape=(N, 6480))


# ── 2) Generation ────────────────────────────────────────────────────────
BATCH = 2000
MAX_TRIES = 10
Y_MAX = np.log1p(1e4)

flow = build_flow()
flow.load_state_dict(torch.load(f'z_prior_nf_{TAG}.pt', map_location='cpu'))
flow.to(device)
flow.eval()

with torch.no_grad():
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        n = end - start
        Cb = C[start:end]
        Yb = Y_thr[start:end]
        z = flow.sample(1, context=Cb).squeeze(1)
        hit_logit, val, log_var_x = model.decode(z, Cb)
        log_var_x = log_var_x.clamp(-7, 3)
        p = torch.sigmoid(hit_logit)
        mask = torch.bernoulli(p)       # Returns 0 or 1 according to the sampling prob. p
        sigma = torch.exp(0.5 * log_var_x)
        eps = torch.randn_like(val)
        y = val + sigma * eps

        for _ in range(MAX_TRIES):
            bad = (mask > 0) & (y < Yb)
            if not bad.any():
                break
            y_new = val + sigma * torch.randn_like(val)
            y = torch.where(bad, y_new, y)

        y = torch.maximum(y, Yb * 1.001)
        y = torch.clamp(y, max=Y_MAX)
        G[start:end] = (y * mask).cpu().numpy()
        print(f" {end}/{N}")

G.flush()
print(f"Saved gen_voxel_{TAG}.npy")