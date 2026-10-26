import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

with h5py.File('dataset_1_pions_1.hdf5', 'r') as f:
    energies = f['incident_energies'][:]
    showers = f['showers'][:]

n = len(energies)
n_train = int(n * 0.8)

x_train = torch.tensor(showers[:n_train], dtype=torch.float32)
x_test = torch.tensor(showers[n_train:], dtype=torch.float32)
e_train = torch.tensor(energies[:n_train], dtype=torch.float32)
e_test = torch.tensor(energies[n_train:], dtype=torch.float32)
e_train_norm = torch.log(e_train)
e_test_norm = torch.log(e_test)
x_sum_train = x_train.sum(dim=1, keepdim=True)
x_train_norm = x_train / (x_sum_train + 1e-8)
x_sum_test = x_test.sum(dim=1, keepdim=True)
x_test_norm = x_test / (x_sum_test + 1e-8)

train_dataset = TensorDataset(x_train_norm, e_train_norm)
test_dataset= TensorDataset(x_test_norm, e_test_norm)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

class VAE(nn.Module):
    def __init__(self, input_dim=368, latent_dim=16, condition_dim=1):
        super().__init__()

        # Encoder: shower + energy -> (mu, log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU())
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder: z(dist.) + energy -> shower
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid())    # This normalises the return of decoder, the shower energy.)

    def encode(self, x, energy):
        h = self.encoder(torch.cat([x, energy], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std   # This is the final encoded z.

    def decode(self, z, energy):
        return self.decoder(torch.cat([z, energy], dim=1))

    def forward(self, x, energy):
        mu, logvar = self.encode(x, energy)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, energy), mu, logvar

def vae_loss(x, x_pred, mu, logvar, kl_weight=0.001):
    recon_loss = F.mse_loss(x_pred, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

model = VAE(input_dim=533)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

for epoch in range(100):
    model.train()
    train_loss = 0
    for x_batch, e_batch in train_loader:
        x_pred, mu, logvar = model(x_batch, e_batch)
        loss, recon, k1 = vae_loss(x_batch, x_pred, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    if epoch % 10 == 0:
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_batch, e_batch in test_loader:
                x_pred, mu, logvar = model(x_batch, e_batch)
                loss, _, _ = vae_loss(x_batch, x_pred, mu, logvar, kl_weight=0.001)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f"epoch {epoch:3d} | train loss: {train_loss:.4e} | test loss: {test_loss:.4e}")


model.eval()
layer_sizes = [8, 100, 100, 5, 150, 160, 10]

with torch.no_grad():
    z = torch.randn(1, 16)
    E = e_test_norm[0:1]
    pred_shower = model.decode(z, E).numpy()[0]
    true_shower = x_test_norm[0].numpy()

    fig, axes = plt.subplots(2, 7, figsize=(20, 4))
    idx = 0
    for i, size in enumerate(layer_sizes):
        pred_layer = pred_shower[idx:idx+size]
        true_layer = true_shower[idx:idx+size]
        
        axes[0][i].bar(range(size), true_layer, color='blue', alpha=0.7)
        axes[0][i].set_title(f'Layer {i+1} ({size} cells)')
        if i == 0:
            axes[0][i].set_ylabel('Geant4')

        axes[1][i].bar(range(size), pred_layer, color='orange', alpha=0.7)
        if i == 0:
            axes[1][i].set_ylabel('VAE')
        
        idx += size

    plt.suptitle(f'Pion shower comparison (E = {e_test[0].item():.0f} MeV)')
    plt.tight_layout()
    plt.savefig('pion_shower_comparison.png')
    plt.show()