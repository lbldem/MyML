import torch
import torch.nn as nn

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
