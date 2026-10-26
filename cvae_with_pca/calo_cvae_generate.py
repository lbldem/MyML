import numpy as np
import torch
import pickle
import h5py
import matplotlib.pyplot as plt
from cvae_model import CVAE, Z_DIM


N = 20000
THRESHOLD_MEV = 0.01515


# ── Loading model and parameters ───────────────────────────────────────────────────
model = CVAE()
model.load_state_dict(torch.load('calo_cvae.pt', map_location='cpu'))
model.eval()
norm = torch.load('norm_params.pt', weights_only=False)
with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)
test_pca = np.load('test_pca.npy')[:N]      # (N, 64)
test_E = np.load('test_energies.npy')[:N]   # (N, 1) MeV


# ── 1) Normalising condition(incident energies) ────────────────────────────
log_E = np.log(test_E)
c = (log_E - norm['log_E_min']) / (norm['log_E_max'] - norm['log_E_min'])
C = torch.tensor(c, dtype=torch.float32)


# ── 2) Generate using model: z ~ N(0, I) → decoder ────────────────────────────
with torch.no_grad():
    z = torch.randn(N, Z_DIM)
    x_n = model.decoder(torch.cat([z, C], dim=1))


# Undoing normalisation → PCA ────────────────────────────
x_pca_gen = (x_n * norm['X_std'] + norm['X_mean']).numpy()


# ── PCA → voxel (inverse pca) ───────────────────────────
def pca_to_voxels(x_pca, E):
    y = pca.inverse_transform(x_pca)        # Undoing the PCA
    v = np.expm1(y) / 1e4                   # Undoing the preprocess
    v[v < THRESHOLD_MEV / E] = 0.0
    return v.astype(np.float32)


# ── Variables to compare ─────────────────────────────────────────────
with h5py.File('dataset_2_2.hdf5', 'r') as f:
    S = f['showers'][:N]
    E_h = f['incident_energies'][:N]
S[S < THRESHOLD_MEV] = 0.0
real = (S / E_h).astype(np.float32)         # Original
real_pca = pca_to_voxels(test_pca, test_E)  # Original -> inverse pca
gen = pca_to_voxels(x_pca_gen, test_E)      # Gen.data -> inverse pca
sets = {'Real': real, 'Real->PCA->back': real_pca, 'CVAE': gen}


# ── Checking pca coordinate of generated data ────────────────────────────
print("std of generated standardized corods:", x_n.std(dim=0).mean().item())


# ── Plot ──────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for name, v in sets.items():
    ax[0, 0].hist(v.sum(1), bins=60, range=(0, 1.5), histtype='step', density=True, label=name)
    ax[0, 1].plot(v.mean(0), lw=0.8, label=name)
    ax[0, 2].hist((v==0).mean(1), bins=40, range=(0, 1), histtype='step', density=True, label=name)
    ax[1, 1].plot(v.std(0), lw=0.8, label=name)
ax[0, 0].set_title('Total energy fraction')
ax[0, 1].set_title('Mean voxel profile')
ax[0, 2].set_title('Sparsity (zero fraction per event)')
ax[1, 1].set_title('Event-to-event std per voxel')          # See how matching axes tell about each other.

edges = np.logspace(np.log10(test_E.min()), np.log10(test_E.max()), 11)
idx = np.clip(np.digitize(test_E[:, 0], edges) - 1, 0, 9)
centers = np.sqrt(edges[:-1] * edges[1:]) / 1000    # GeV
for name, v in sets.items():
    tot = v.sum(1)
    m = [tot[idx == k].mean() for k in range(10)]
    s = [tot[idx == k].std() for k in range(10)]
    ax[1, 0].errorbar(centers, m, yerr=s, marker='o', capsize=3, label=name)
ax[1, 0].set_xscale('log')
ax[1, 0].set_xlabel('E_inc [GeV]')
ax[1, 0].set_title('Response vs E_inc')


x_real_n = (test_pca - norm['X_mean'].numpy()) / norm['X_std'].numpy()
ax[1, 2].hist(x_real_n[:, 0], bins=60, histtype='step', density=True, label='Real')
ax[1, 2].hist(x_n.numpy()[:, 0], bins=60, histtype='step', density=True, label='CVAE')
ax[1, 2].set_title('1st PCA coordinate (standardized)')

for a in ax.flat:
    a.legend(fontsize=8)

plt.tight_layout()
plt.savefig('cvae_comparison.png', dpi=150)
plt.show()