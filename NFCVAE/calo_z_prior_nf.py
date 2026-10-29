# calo_z_prior_nf.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from nflows import flows, distributions, transforms

TAG = 'b01'

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

Z = torch.tensor(np.load(f'train_z_{TAG}.npy')).to(device)      # (100000, 64)
C = torch.tensor(np.load(f'train_c_{TAG}.npy')).to(device)      # (100000, 1)
DIM, C_DIM = Z.shape[1], C.shape[1]

loader = DataLoader(TensorDataset(Z, C), batch_size=512, shuffle=True)

# ── 기준선: z ~ N(0, I)라고 했을 때의 NLL ─────────────────────
base_nll = (0.5 * Z.pow(2).sum(1) + 0.5 * DIM * np.log(2 * np.pi)).mean().item()
print(f"baseline NLL under N(0,I): {base_nll:.2f}")

# ── context를 실제로 쓰는 subnet ──────────────────────────────
class ContextNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x, context=None):
        return self.net(torch.cat([x, context], dim=1))                            # (1) x와 context를 붙여서

def make_net(in_f, out_f):
    return ContextNet(nn.Sequential(
        nn.Linear(in_f + C_DIM, 256), nn.ReLU(),        # 입력이 c만큼 늘어남
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, out_f)
    ))

# ── 8-layer coupling, mask 번갈아 ─────────────────────────────
half   = DIM // 2
mask_A = torch.cat([torch.ones(half), torch.zeros(DIM - half)])
mask_B = 1 - mask_A

layers = []
for i in range(8):
    mask = mask_A if i % 2 == 0 else mask_B
    layers.append(transforms.AffineCouplingTransform(
        mask=mask.byte(), transform_net_create_fn=make_net))

flow = flows.Flow(transforms.CompositeTransform(layers),
                  distributions.StandardNormal(shape=[DIM])).float().to(device)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

# ── 학습 ──────────────────────────────────────────────────────
EPOCHS = 50
for epoch in range(EPOCHS):
    tot = 0
    for zb, cb in loader:
        optimizer.zero_grad()
        loss = -flow.log_prob(zb, context=cb).mean()                                      # (2) context를 넘겨서 NLL
        loss.backward()
        optimizer.step()
        tot += loss.item() * len(zb)
    nll = tot / len(Z)
    print(f"Epoch {epoch+1:2d}  NLL={nll:.2f}  (gain over N(0,I): {base_nll - nll:.2f})")

torch.save(flow.state_dict(), f'z_prior_nf_{TAG}.pt')