# step7_nf_generate.py — NF prior에서 z를 뽑아 decoder로 shower를 생성하고 실제와 비교
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from common import DATA, OUT, device
from step1_model import HurdleCVAE
from step4_flow_model import build_flow
from numpy.lib.format import open_memmap


MODEL_PATH = OUT / 'calo_cvae_hurdle.pt'
NORM_PATH  = OUT / 'norm_params_voxel.pt'
NF_PATH    = OUT / 'z_prior_nf_std.pt'
N, BATCH, MAX_TRIES = 100000, 2000, 10
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
G = open_memmap(OUT / 'gen_voxel_std.npy', mode='w+', dtype=np.float32, shape=(N, 6480))

z_list = []
with torch.no_grad():
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        Cb, Yb = C[start:end], Y_thr[start:end]
        z = flow.sample(1, context=Cb).squeeze(1)
        z = z * z_std + z_mean                                                                          # [B33] NF prior에서 이벤트당 z 하나. sample의 출력 shape (n, 1, 64)에 주의
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
        G[start:end] = (y * mask).cpu().numpy()        # hit가 아니면 0

Zg = np.concatenate(z_list)
print("z std per dim: min %.2f max %.2f" % (Zg.std(0).min(), Zg.std(0).max()))
G.flush()