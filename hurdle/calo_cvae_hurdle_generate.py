import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from calo_cvae_hurdle_model import HurdleCVAE, Z_DIM

MODE = 'generate'
X_test = np.load('test_voxel.npy', mmap_mode = 'r')
N = 20000
THRESHOLD_MEV = 0.01515


# ── Loading model and parameters ───────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Device: ", device)
model = HurdleCVAE()
model.load_state_dict(torch.load('calo_cvae_hurdle.pt', map_location='cpu'))
model.to(device)
model.eval()
norm = torch.load('norm_params_voxel.pt', weights_only=False)
test_E = np.load('test_energies.npy')[:N]   # (N, 1) MeV


# ── 1) Normalising condition(incident energies) ────────────────────────────
log_E = np.log(test_E)
c = (log_E - norm['log_E_min']) / (norm['log_E_max'] - norm['log_E_min'])
C = torch.tensor(c, dtype=torch.float32).to(device)
y_thr = np.log1p(THRESHOLD_MEV / test_E * 1e4)
Y_thr = torch.tensor(y_thr, dtype=torch.float32).to(device)

# ── 2) Now, we don't generate but reconstruct, using model: z ~ N(0, I) → decoder ────────────────────────────
BATCH = 2000
MAX_TRIES = 10
y_list = []
z_list = []
Y_MAX = np.log1p(1e4)
sig_sum, hit_sum = 0.0, 0.0
with torch.no_grad():
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        n = end - start
        Cb = C[start:end]
        Yb = Y_thr[start:end]
        if MODE == 'reconstruct':
            xb = torch.from_numpy(np.array(X_test[start:end])).to(device)
            h = model.encoder(torch.cat([xb, Cb], dim=1))
            mu_z = model.fc_mu(h)
            lv_z = model.fc_log_var(h)
            z = model.reparametrize(mu_z, lv_z)
        else:
            z = torch.randn(n, Z_DIM, device=device)
        z_list.append(z.cpu().numpy())
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
        over = ((y > Y_MAX) & (mask > 0)).sum().item()
        print(over)
        y = torch.clamp(y, max=Y_MAX)
        y_list.append((y * mask).cpu().numpy())
        sig_sum += (sigma * mask).sum().item()
        hit_sum += mask.sum().item()
        print("max sigma among hits: ", (sigma * mask).max().item())
    y_gen = np.concatenate(y_list)
    Z = np.concatenate(z_list)
    print("hits from coin: ", hit_sum / N)
    print("mean sigma: ", sig_sum / hit_sum)
    print("z mean per dim: min %.2f max %.2f" % (Z.mean(0).min(), Z.mean(0).max()))
    print("z std per dim: min %.2f max %.2f" % (Z.std(0).min(), Z.std(0).max()))


# ── log_to_fraction ───────────────────────────────────────────────────
def log_to_fraction(y, E):
    v = np.expm1(y) / 1e4
    v[v < THRESHOLD_MEV / E] = 0.0
    return v.astype(np.float32)


# ── Sets to compare ─────────────────────────────────────────────
with h5py.File('dataset_2_2.hdf5', 'r') as f:
    S = f['showers'][:N]
    E_h = f['incident_energies'][:N]
S[S < THRESHOLD_MEV] = 0.0
real = (S / E_h).astype(np.float32)               # Original             
gen = log_to_fraction(y_gen, test_E)              # Gen.data -> inverse preprocess
sets = {'Real': real, MODE: gen}
print("hits after cuts: ", (gen > 0).sum(1).mean())


# ── Checking hit voxels per event and energy fraction ────────────────────────────
for name, v in sets.items():
    print(f"{name:5s} hit voxels/event = {(v > 0).sum(1).mean():7.1f}   "
        f"energy fraction = {v.sum(1).mean():.3f}")


# ── Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 3, figsize=(16, 9))

for name, v in sets.items():
    # Energy fraction over events, sum over voxels
    ax[0, 0].hist(v.sum(1), bins=60, range=(0, 1.5), histtype='step', density=True, label=name)

    # Mean energy over voxels, sum over events
    ax[0, 1].plot(v.mean(0), lw=0.8, label=name)

    # Zero Voxel fractions over events, sum over voxels
    ax[0, 2].hist((v==0).mean(1), bins=40, range=(0, 1), histtype='step', density=True, label=name)

    # STD between events over voxels, sum over events
    ax[1, 1].plot(v.std(0), lw=0.8, label=name)

ax[0, 0].set_title('Total energy fraction')
ax[0, 1].set_title('Mean voxel profile')
ax[0, 2].set_title('Sparsity (zero fraction per event)')
ax[1, 1].set_title('Event-to-event STD per voxel')

edges = np.logspace(np.log10(test_E.min()), np.log10(test_E.max()), 11)
idx = np.clip(np.digitize(test_E[:, 0], edges) - 1, 0, 9)
centers = np.sqrt(edges[:-1] * edges[1:]) / 1000    # GeV
for name, v in sets.items():
    tot = v.sum(1)
    m = [tot[idx == k].mean() for k in range(10)]
    s = [tot[idx == k].std() for k in range(10)]
    ax[1, 0].errorbar(centers, m, yerr=s, marker='o', capsize=2.5, label=name)
ax[1, 0].set_xscale('log')
ax[1, 0].set_title('Response vs E_inc')

bins = np.logspace(-8, 0, 81)
for name, v in sets.items():
    vals = v[v > 0]
    ax[1, 2].hist(vals, bins=bins, histtype='step', label=name)
ax[1, 2].set_xscale('log')
ax[1, 2].set_yscale('log')
ax[1, 2].set_title('Voxel energy spectrum')

frac = r'$E_\mathrm{voxel}/E_\mathrm{inc}$'
tot  = r'$\sum E_\mathrm{voxel}/E_\mathrm{inc}$'

ax[0, 0].set_xlabel(f'Total energy fraction {tot}')
ax[0, 0].set_ylabel('Probability density')
ax[0, 1].set_xlabel('Voxel index')
ax[0, 1].set_ylabel(f'Mean {frac}')
ax[0, 2].set_xlabel('Fraction of zero voxels per event')
ax[0, 2].set_ylabel('Probability density')
ax[1, 0].set_xlabel(r'$E_\mathrm{inc}$ [GeV]')
ax[1, 0].set_ylabel(f'Mean {tot} (bars: STD)')
ax[1, 1].set_xlabel('Voxel index')
ax[1, 1].set_ylabel(f'Std of {frac} over events')
ax[1, 2].set_xlabel(f'Voxel energy fraction {frac}')
ax[1, 2].set_ylabel(f'Number of hits (N = {N} events)')


for a in ax.flat:
    a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'cvae_hurdle_comparison_new_{MODE}.png', dpi=150)
plt.show()