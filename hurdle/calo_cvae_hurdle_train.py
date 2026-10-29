import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from calo_cvae_hurdle_model import HurdleCVAE


class VoxelDataset(Dataset):
    def __init__(self, x_path, c):
        self.X = np.load(x_path, mmap_mode='r')
        self.C = torch.tensor(c, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.X[i]))
        return x, self.C[i]

train_E = np.load('train_energies.npy')
log_E = np.log(train_E)
log_E_min = np.min(log_E)
log_E_max = np.max(log_E)
c_train = (log_E - log_E_min) / (log_E_max - log_E_min)

torch.save({'log_E_min': log_E_min, 'log_E_max': log_E_max}, 'norm_params_voxel.pt')

dataset = VoxelDataset('train_voxel.npy', c_train)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

BETA = 1.0
def hurdle_loss(hit_logit, val, log_var_x, x, mu, log_var):
    log_var_x = log_var_x.clamp(-7, 3)
    hit = (x > 0).float()
    bce = F.binary_cross_entropy_with_logits(hit_logit, hit, reduction='sum')
    nll = 0.5 * (hit * (((x - val)**2)/(log_var_x.exp()) + log_var_x)).sum()
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + nll + BETA * kl, bce, nll, kl

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = HurdleCVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30
for epoch in range(EPOCHS):
    tot, tot_b, tot_n, tot_k = 0, 0, 0, 0
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        hit_logit, val, log_var_x, mu, log_var = model(x, c)
        loss, bce, nll, kl = hurdle_loss(hit_logit, val, log_var_x, x, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot += loss.item(); tot_b += bce.item(); tot_n += nll.item(); tot_k += kl.item()
    n = len(loader.dataset)
    print(f"Epoch {epoch+1:2d}, loss={tot/n:.2f}, bce={tot_b/n:.2f}, nll={tot_n/n:.2f}, kl={tot_k/n:.2f}")

torch.save(model.state_dict(), 'calo_cvae_hurdle.pt')