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
NF_PATH    = OUT / 'z_prior_nf.pt'
N, BATCH, MAX_TRIES = 20000, 2000, 10
THRESHOLD_MEV = 0.01515
Y_MAX = np.log1p(1e4)                                  # f = 1 일 때의 y

model = HurdleCVAE(); model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')); model.to(device).eval()
flow = build_flow();  flow.load_state_dict(torch.load(NF_PATH, map_location='cpu'));     flow.to(device).eval()
norm = torch.load(NORM_PATH, weights_only=False)

# ── condition과 threshold (test 에너지 사용) ───────────────────────
test_E = np.load(DATA / 'test_energies.npy')[:N]       # (N, 1) MeV
c = (np.log(test_E) - norm['log_E_min']) / (norm['log_E_max'] - norm['log_E_min'])
C = torch.tensor(c, dtype=torch.float32).to(device)
y_thr = ___                                            # [B32] threshold 15.15 keV를 y 공간으로: 전처리 식에 그대로 대입
Y_thr = torch.tensor(y_thr, dtype=torch.float32).to(device)

y_list = []
with torch.no_grad():
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        Cb, Yb = C[start:end], Y_thr[start:end]

        z = ___                                        # [B33] NF prior에서 이벤트당 z 하나. sample의 출력 shape (n, 1, 64)에 주의
        hit_logit, val, log_var_x = model.decode(z, Cb)
        log_var_x = log_var_x.clamp(-7, 3)

        mask  = ___                                    # [B34] 동전 던지기: hit 확률로 0/1 샘플링 (sigmoid → bernoulli)
        sigma = ___                                    # [B35] ln σ² → σ
        y = val + sigma * torch.randn_like(val)

        for _ in range(MAX_TRIES):                     # threshold 아래로 떨어진 hit만 다시 뽑기
            bad = ___                                  # [B36] "hit인데 y < Yb" 인 위치 (두 조건을 & 로, 괄호 주의)
            if not bad.any():
                break
            y_new = val + sigma * torch.randn_like(val)   # 매번 새 ε
            y = ___                                    # [B37] bad인 곳만 y_new로 교체

        y = torch.maximum(y, Yb * 1.001)               # 10번 안에 못 올라온 값은 threshold 바로 위로
        y = torch.clamp(y, max=Y_MAX)
        y_list.append((y * mask).cpu().numpy())        # hit가 아니면 0
y_gen = np.concatenate(y_list)

# ── y → 에너지 비율 f = E_voxel / E_inc ───────────────────────────
def log_to_fraction(y, E):
    v = ___                                            # [B38] y = ln(1 + 10⁴ f) 의 역변환 (np.expm1 활용)
    v[v < THRESHOLD_MEV / E] = 0.0
    return v.astype(np.float32)

with h5py.File(DATA / 'dataset_2_2.hdf5', 'r') as f:
    S, E_h = f['showers'][:N], f['incident_energies'][:N]
S[S < THRESHOLD_MEV] = 0.0
sets = {'Real': (S / E_h).astype(np.float32), 'NF prior': log_to_fraction(y_gen, test_E)}

for name, v in sets.items():
    print(f"{name:8s} hit voxels/event = {(v > 0).sum(1).mean():7.1f}   energy fraction = {v.sum(1).mean():.3f}")

# ── 간단한 비교 플롯 3개 ──────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
bins = np.logspace(-8, 0, 81)
for name, v in sets.items():
    ax[0].hist(v.sum(1), bins=60, range=(0, 1.5), histtype='step', density=True, label=name)
    ax[1].hist((v == 0).mean(1), bins=40, range=(0, 1), histtype='step', density=True, label=name)
    ax[2].hist(v[v > 0], bins=bins, histtype='step', label=name)
ax[0].set_xlabel(r'$\sum E_\mathrm{voxel}/E_\mathrm{inc}$'); ax[0].set_ylabel('Probability density')
ax[1].set_xlabel('Fraction of zero voxels per event');   ax[1].set_ylabel('Probability density')
ax[2].set_xlabel(r'$E_\mathrm{voxel}/E_\mathrm{inc}$');    ax[2].set_ylabel(f'Number of hits (N = {N})')
ax[2].set_xscale('log'); ax[2].set_yscale('log')
for a in ax: a.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / 'review_nf_prior.png', dpi=150)
plt.show()
