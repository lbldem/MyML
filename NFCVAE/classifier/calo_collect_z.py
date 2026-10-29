import numpy as np
import torch
from calo_cvae_hurdle_model import HurdleCVAE

TAG = 'b01'

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = HurdleCVAE()
model.load_state_dict(torch.load(f'calo_cvae_hurdle_{TAG}.pt', map_location='cpu'))
model.to(device)
model.eval()
norm = torch.load('norm_params_voxel.pt', weights_only=False)
X_train = np.load('train_voxel.npy', mmap_mode='r')
E_train = np.load('train_energies.npy')

log_E = np.log(E_train)
c_train = (log_E - norm['log_E_min']) / (norm['log_E_max'] - norm['log_E_min'])
C_train = torch.tensor(c_train, dtype=torch.float32).to(device)

N = len(X_train)
BATCH = 2000
z_list = []

with torch.no_grad():
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        xb = torch.from_numpy(np.array(X_train[start:end])).to(device)
        Cb = C_train[start:end]
        h = model.encoder(torch.cat([xb, Cb], dim=1))
        mu_z = model.fc_mu(h)
        lv_z = model.fc_log_var(h)
        z = model.reparametrize(mu_z, lv_z)
        z_list.append(z.cpu().numpy())
Z = np.concatenate(z_list).astype(np.float32)
np.save(f'train_z_{TAG}.npy', Z)
np.save(f'train_c_{TAG}.npy', c_train.astype(np.float32))
print(Z.shape)
print("z mean per dim: min %.2f max %.2f" % (Z.mean(0).min(), Z.mean(0).max()))
print("z std  per dim: min %.2f max %.2f" % (Z.std(0).min(),  Z.std(0).max()))