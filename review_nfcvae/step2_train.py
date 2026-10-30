# step2_train.py — Hurdle CVAE 학습 (β = 1, N(0,I) prior)
# 시간이 오래 걸리면 EPOCHS를 줄여서 먼저 돌아가는지만 확인하자.
# 학습을 건너뛰고 싶으면 step3부터 MODEL_PATH를 원래 파일(DATA/'calo_cvae_hurdle.pt')로 바꿔도 된다.
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from common import DATA, OUT, device
from step1_model import HurdleCVAE

class VoxelDataset(Dataset):
    """train_voxel.npy를 memmap으로 열어 두고, 요청이 올 때 이벤트 하나씩만 읽는다."""
    def __init__(self, x_path, c):
        self.X = np.load(x_path, mmap_mode='r')                             # [B10] 읽기 전용 memmap 모드 문자
        self.C = torch.tensor(c, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.X[i]))                           # np.array(...)로 디스크에서 실제로 읽어 온다
        return x, self.C[i]

# ── condition 만들기 ─────────────────────────────────────────────
train_E = np.load(DATA / 'train_energies.npy')                              # (100000, 1) MeV
log_E = np.log(train_E)
log_E_min, log_E_max = log_E.min(), log_E.max()
c_train = (log_E - log_E_min) / (log_E_max - log_E_min)                                                             # [B11] ln E를 [0, 1]로 min-max 정규화
torch.save({'log_E_min': log_E_min, 'log_E_max': log_E_max}, OUT / 'norm_params_voxel.pt')

loader = DataLoader(VoxelDataset(DATA / 'train_voxel.npy', c_train), batch_size=512, shuffle=True)                  # 모델에게 train_voxel.npy를 학습시키고 그 결과를 train_voxel.npy와 비교함.

# ── loss ─────────────────────────────────────────────────────────
BETA = 1.0
def hurdle_loss(hit_logit, val, log_var_x, x, mu, log_var):
    log_var_x = log_var_x.clamp(-7, 3)
    hit = (x > 0).float()                                                                           # [B12] voxel이 hit이면 1.0, 아니면 0.0 (float 텐서)        >>> 정답 x를 0/1로 label 주기
    bce = F.binary_cross_entropy_with_logits(hit_logit, hit, reduction='sum')                       # [B13] logit을 직접 받는 BCE 함수 (F.xxx)                  >>> 예측값인 hit_logit과 정답인 hit 사이의 오차
    nll = 0.5 * (hit * ((x - val)**2 / log_var_x.exp() + log_var_x)).sum()                          # [B14] Gaussian NLL의 괄호 안: (x−μ)²/σ² + ln σ²          >>> 예측값인 val, log_var_x와 정답인 x 사이의 오차
    kl  = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())                                 # [B15] N(μ,σ²)와 N(0,1) 사이 KL, 64차원과 batch 전체 합    >>> 예측값인 mu, log_var와 N(0, I) 사이의 오차
    return bce + nll + BETA * kl, bce, nll, kl

model = HurdleCVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30
for epoch in range(EPOCHS):
    tot, tot_b, tot_n, tot_k = 0, 0, 0, 0
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        hit_logit, val, log_var_x, mu, log_var = model(x, c)
        loss, bce, nll, kl = hurdle_loss(hit_logit, val, log_var_x, x, mu, log_var)
        optimizer.zero_grad()                                                                       # [B16] 기울기 초기화
        loss.backward()                                                                             # [B17] 역전파
        optimizer.step()                                                                            # [B18] 가중치 갱신
        tot += loss.item(); tot_b += bce.item(); tot_n += nll.item(); tot_k += kl.item()
    n = len(loader.dataset)
    print(f"Epoch {epoch+1:2d}, loss={tot/n:.2f}, bce={tot_b/n:.2f}, nll={tot_n/n:.2f}, kl={tot_k/n:.2f}")

torch.save(model.state_dict(), OUT / 'calo_cvae_hurdle.pt')
