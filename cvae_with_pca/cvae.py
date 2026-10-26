import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

X_DIM = 64  # PCA dimension (input)
C_DIM = 1   # Condition's dimension
Z_DIM = 16  # Latent dimension

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: [x, c] -> h
        self.encoder = nn.Sequential(
            nn.Linear(X_DIM + C_DIM, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU())

        self.fc_mu = nn.Linear(128, Z_DIM)
        self.fc_log_var = nn.Linear(128, Z_DIM)

        # Decoder: [z, c] -> x_reconstructed
        self.decoder = nn.Sequential(
            nn.Linear(Z_DIM + C_DIM, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, X_DIM))
    
    def reparametrize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    
    def forward(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z = self.reparametrize(mu, log_var)
        x_recon = self.decoder(torch.cat([z,c], dim=1))
        return x_recon, mu, log_var


train_pca = np.load('train_pca.npy')
train_E = np.load('train_energies.npy')
log_E = np.log(train_E)
log_E_min = np.min(log_E)
log_E_max = np.max(log_E)
c_train = (log_E - log_E_min) / (log_E_max - log_E_min)
print(c_train.min(), c_train.max())
X_train = torch.tensor(train_pca, dtype=torch.float32)
C_train = torch.tensor(c_train, dtype=torch.float32)
print(X_train.shape, C_train.shape)

X_mean = X_train.mean(dim=0, keepdim=True)
X_std = X_train.std(dim=0, keepdim=True)
X_train_n = (X_train - X_mean) / X_std      # Standard normalisation to balance off things between reco loss and KL loss.

torch.save({'X_mean': X_mean,
            'X_std': X_std,
            'log_E_min': log_E_min,
            'log_E_max': log_E_max}, 'norm_params.pt')      # Dictionary saved.

loader = DataLoader(TensorDataset(X_train_n, C_train), batch_size=512, shuffle=True)

BETA = 1.0

def cvae_loss(x_recon, x, mu, log_var):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + BETA * kl_loss, recon_loss, kl_loss

model = CVAE()      # Giving the defined CVAE structure to a variable model. 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 100
for epoch in range(EPOCHS + 1):
    tot, tot_r, tot_k = 0, 0, 0
    for x, c in loader:
        x_recon, mu, log_var = model(x, c)
        loss, recon_loss, kl_loss = cvae_loss(x_recon, x, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot += loss.item()
        tot_r += recon_loss.item()
        tot_k += kl_loss.item()
    n = len(loader.dataset)
    print(f"Epoch {epoch+1:2d}, loss={tot/n:.4f}, recon_loss={tot_r/n:.4f}, kl_loss={tot_k/n:.4f}")     # Dividing by n(event number) to give us a mean losses of every batch we pick up.

torch.save(model.state_dict(), 'calo_cvae.pt')