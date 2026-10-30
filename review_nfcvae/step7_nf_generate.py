# step6_generate.py — NF prior에서 z를 뽑아 decoder로 shower를 생성하고 실제와 비교
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from common import DATA, OUT, device
from step1_model import HurdleCVAE
from step4_flow_model import build_flow

MODEL_PATH = OUT / 'calo_cvae_hurdle.pt'
NORM_PATH  = OUT / 'norm_params_voxel.pt'
NF_PATH    = OUT / 'z_prior_nf_std.pt'
N, BATCH, MAX_TRIES = 20000, 2000, 10
THRESHOLD_MEV = 0.01515
Y_MAX = np.log1p(1e4)                                  # f = 1 일 때의 y


model = HurdleCVAE(); model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')); model.to(device).eval()
flow = build_flow();  flow.load_state_dict(torch.load(NF_PATH, map_location='cpu'));     flow.to(device).eval()
norm = torch.load(NORM_PATH, weights_only=False)
zn = torch.load(OUT / 'z_norm.pt')
z_mean, z_std = zn['z_mean'].to(device), zn['z_std'].to(device)

# ── condition과 threshold (test 에너지 사용) ───────────────────────
test_E = np.load(DATA / 'test_energies.npy')[:N]       # (N, 1) MeV
c = (np.log(test_E) - norm['log_E_min']) / (norm['log_E_max'] - norm['log_E_min'])
C = torch.tensor(c, dtype=torch.float32).to(device)
y_thr = np.log1p(THRESHOLD_MEV / test_E * 1e4)                                                          # [B32] threshold 15.15 keV를 y 공간으로: 전처리 식에 그대로 대입
Y_thr = torch.tensor(y_thr, dtype=torch.float32).to(device)                                             ## y_thr = (20000, 1) array >> Y_thr = (20000, 1) tensor

y_list = []
z_list = []
with torch.no_grad():
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        Cb, Yb = C[start:end], Y_thr[start:end]
        z = flow.sample(1, context=Cb).squeeze(1)
        z = z * z_std + z_mean                                                    # [B33] NF prior에서 이벤트당 z 하나. sample의 출력 shape (n, 1, 64)에 주의
        z_list.append(z.cpu().numpy())
        hit_logit, val, log_var_x = model.decode(z, Cb)
        log_var_x = log_var_x.clamp(-7, 3)
        
        mask  = torch.bernoulli(torch.sigmoid(hit_logit))                                               # [B34] 동전 던지기: hit 확률로 0/1 샘플링 (sigmoid → bernoulli)
        sigma = torch.exp(0.5 * log_var_x)                                                              # [B35] ln σ² → σ
        y = val + sigma * torch.randn_like(val)

        for _ in range(MAX_TRIES):                                                                      # threshold 아래로 떨어진 hit만 다시 뽑기
            bad = (mask > 0) & (y < Yb)                                                                 # [B36] "hit인데 y < Yb" 인 위치 (두 조건을 & 로, 괄호 주의)
            if not bad.any():
                break
            y_new = val + sigma * torch.randn_like(val)                                                 # 매번 새 ε
            y = torch.where(bad, y_new, y)                                                              # [B37] bad인 곳만 y_new로 교체

        y = torch.maximum(y, Yb * 1.001)               # 10번 안에 못 올라온 값은 threshold 바로 위로
        y = torch.clamp(y, max=Y_MAX)
        y_list.append((y * mask).cpu().numpy())        # hit가 아니면 0
y_gen = np.concatenate(y_list)
Zg = np.concatenate(z_list)
print("z std per dim: min %.2f max %.2f" % (Zg.std(0).min(), Zg.std(0).max()))

# ── y → 에너지 비율 f = E_voxel / E_inc ───────────────────────────
def log_to_fraction(y, E):
    v = np.expm1(y) / 10**4                                                                             # [B38] y = ln(1 + 10⁴ f) 의 역변환 (np.expm1 활용)
    v[v < THRESHOLD_MEV / E] = 0.0
    return v.astype(np.float32)

with h5py.File(DATA / 'dataset_2_2.hdf5', 'r') as f:
    S, E_h = f['showers'][:N], f['incident_energies'][:N]
S[S < THRESHOLD_MEV] = 0.0
sets = {'Real': (S / E_h).astype(np.float32), 'NF prior': log_to_fraction(y_gen, test_E)}

for name, v in sets.items():
    print(f"{name:8s} hit voxels/event = {(v > 0).sum(1).mean():7.1f}   energy fraction = {v.sum(1).mean():.3f}")


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

plt.show()