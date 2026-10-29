import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from calo_cvae_voxel_model import CVAE, Z_DIM


N = 20000
THRESHOLD_MEV = 0.01515


# ── Loading model and parameters ───────────────────────────────────────────────────
model = CVAE()
model.load_state_dict(torch.load('calo_cvae_voxel.pt', map_location='cpu'))
model.eval()
norm = torch.load('norm_params_voxel.pt', weights_only=False)
test_E = np.load('test_energies.npy')[:N]   # (N, 1) MeV


# ── 1) Normalising condition(incident energies) ────────────────────────────
log_E = np.log(test_E)
c = (log_E - norm['log_E_min']) / (norm['log_E_max'] - norm['log_E_min'])
C = torch.tensor(c, dtype=torch.float32)


# ── 2) Generate using model: z ~ N(0, I) → decoder ────────────────────────────
with torch.no_grad():
    z = torch.randn(N, Z_DIM)
    y_gen = model.decoder(torch.cat([z, C], dim=1)).numpy()


# ── log_to_fraction ───────────────────────────────────────────────────
def log_to_fraction(y, E):
    v = np.expm1(y) / 1e4
    v[v < THRESHOLD_MEV / E] = 0.0
    return v.astype(np.float32)


# ── Variables to compare ─────────────────────────────────────────────
with h5py.File('dataset_2_2.hdf5', 'r') as f:
    S = f['showers'][:N]
    E_h = f['incident_energies'][:N]
S[S < THRESHOLD_MEV] = 0.0
real = (S / E_h).astype(np.float32)               # Original             
gen = log_to_fraction(y_gen, test_E)              # Gen.data -> inverse preprocess
sets = {'Real': real, 'CVAE': gen}


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
plt.savefig('cvae_voxel_comparison.png', dpi=150)
plt.show()