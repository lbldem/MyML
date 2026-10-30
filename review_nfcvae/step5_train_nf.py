# step5_train_nf.py — z ~ q(z|x,c) 샘플에 conditional NF를 최대우도로 맞춘다
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from common import OUT, device
from step4_flow_model import build_flow

Z = torch.tensor(np.load(OUT / 'train_z.npy')).to(device)   # (100000, 64)
C = torch.tensor(np.load(OUT / 'train_c.npy')).to(device)   # (100000, 1)
DIM = Z.shape[1]
z_mean = torch.mean(Z, dim=0, keepdim=True)
z_std = torch.std(Z, dim=0, keepdim=True)
Z = (Z - z_mean) / z_std
torch.save({'z_mean': z_mean.cpu(), 'z_std': z_std.cpu()}, OUT / 'z_norm.pt')
print("Standardised z std per dim: min %.2f max %.2f" % (Z.std(0).min(), Z.std(0).max()))
loader = DataLoader(TensorDataset(Z, C), batch_size=512, shuffle=True)

# ── 기준선: z가 그냥 N(0, I)라고 할 때의 평균 NLL ──────────────────
#    −ln N(z; 0, I) = ½|z|² + (DIM/2) ln 2π
base_nll = (0.5 * Z.pow(2).sum(1) + 0.5 * DIM * np.log(2 * np.pi)).mean().item()                    # [B30] 이벤트마다 위 식을 계산 (Z.pow(2).sum(1) 활용)
print(f"baseline NLL under N(0,I): {base_nll:.2f}")

flow = build_flow(DIM=DIM).to(device)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

EPOCHS = 50
for epoch in range(EPOCHS):
    tot = 0
    for zb, cb in loader:
        optimizer.zero_grad()
        loss = -flow.log_prob(zb, context=cb).mean()                                                 # [B31] 최대우도 = 평균 NLL 최소화. context를 넘기는 것 잊지 말기
        loss.backward()
        optimizer.step()
        tot += loss.item() * len(zb)                                                                 # batch 평균 × batch 크기 = batch 합
    nll = tot / len(Z)
    print(f"Epoch {epoch+1:2d}  NLL={nll:.2f}  (gain over N(0,I): {base_nll - nll:.2f})")

torch.save(flow.state_dict(), OUT / 'z_prior_nf_std.pt')
