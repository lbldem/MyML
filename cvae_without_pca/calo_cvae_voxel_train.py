import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from calo_cvae_voxel_model import CVAE

TAG = 'b01'

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

BETA = 0.1
def cvae_loss(x_recon, x, mu, log_var):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + BETA * kl_loss, recon_loss, kl_loss

model = CVAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 30
for epoch in range(EPOCHS):
    tot, tot_r, tot_k = 0, 0, 0
    for x, c in loader:
        x_recon, mu, log_var = model(x, c)
        loss, recon_loss, kl_loss = cvae_loss(x_recon, x, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot += loss.item(); tot_r += recon_loss.item(); tot_k += kl_loss.item()
    n = len(loader.dataset)
    print(f"Epoch {epoch+1:2d}, loss={tot/n:.2f}, recon_loss={tot_r/n:.2f}, kl_loss={tot_k/n:.2f}")

torch.save(model.state_dict(), f'calo_cvae_voxel_{TAG}.pt')