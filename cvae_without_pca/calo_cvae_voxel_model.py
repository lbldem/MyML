import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

X_DIM = 6480  # without-PCA dimension (input)
C_DIM = 1   # Condition's dimension
Z_DIM = 64  # Latent dimension

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: [x, c] -> h
        self.encoder = nn.Sequential(
            nn.Linear(X_DIM + C_DIM, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU())

        self.fc_mu = nn.Linear(256, Z_DIM)
        self.fc_log_var = nn.Linear(256, Z_DIM)

        # Decoder: [z, c] -> x_reconstructed
        self.decoder = nn.Sequential(
            nn.Linear(Z_DIM + C_DIM, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, X_DIM), nn.ReLU())
    
    def reparametrize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    
    def forward(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z = self.reparametrize(mu, log_var)
        x_recon = self.decoder(torch.cat([z, c], dim=1))
        return x_recon, mu, log_var