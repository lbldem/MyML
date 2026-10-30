# step3_collect_z.py — 학습된 encoder로 aggregate posterior 샘플 z를 모은다
import numpy as np
import torch
from common import DATA, OUT, device
from step1_model import HurdleCVAE

MODEL_PATH = OUT / 'calo_cvae_hurdle.pt'                            # step2를 건너뛰었다면 DATA / 'calo_cvae_hurdle.pt'      >>> Model이 학습한 가중치들을 모아 둔 리스트.
NORM_PATH  = OUT / 'norm_params_voxel.pt'                           # 위와 같은 경우 DATA / 'norm_params_voxel.pt'           >>> Condition인 incident energy의 특징 값들을 적어 둔 리스트.

model = HurdleCVAE()
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.to(device)
model.eval()                                                        # [B19] 평가 모드로 전환

norm = torch.load(NORM_PATH, weights_only=False)
X_train = np.load(DATA / 'train_voxel.npy', mmap_mode='r')
log_E = np.log(np.load(DATA / 'train_energies.npy'))
c_train = (log_E - norm['log_E_min']) / (norm['log_E_max'] - norm['log_E_min'])
C_train = torch.tensor(c_train, dtype=torch.float32).to(device)

N, BATCH = len(X_train), 2000
z_list = []
with torch.no_grad():                                               # [B20] 기울기 계산을 끄는 문맥
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        xb = torch.from_numpy(np.array(X_train[start:end])).to(device)
        Cb = C_train[start:end]
        h = model.encoder(torch.cat([xb, Cb], dim=1))               # [B21] encoder 입력
        mu_z, lv_z = model.fc_mu(h), model.fc_log_var(h)
        z = model.reparametrize(mu_z, lv_z)                         # [B22] μ가 아니라 "샘플"을 모은다. 왜일까? (이론 문서 §9.1)
        z_list.append(z.cpu().numpy())

Z = np.concatenate(z_list)                                          # [B23] 배치 리스트를 (100000, 64) 하나로
np.save(OUT / 'train_z.npy', Z)
np.save(OUT / 'train_c.npy', c_train.astype(np.float32))
print(Z.shape)
print("z mean per dim: min %.2f max %.2f" % (Z.mean(0).min(), Z.mean(0).max()))
print("z std  per dim: min %.2f max %.2f" % (Z.std(0).min(),  Z.std(0).max()))
