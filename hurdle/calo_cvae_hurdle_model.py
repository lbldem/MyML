import torch
import torch.nn as nn

X_DIM = 6480
C_DIM = 1
Z_DIM = 64

class HurdleCVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: same as the non-PCA model
        self.encoder = nn.Sequential(
            nn.Linear(X_DIM + C_DIM, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, Z_DIM)
        self.fc_log_var = nn.Linear(256, Z_DIM)

        # Decoder body: same as the non-PCA model
        self.decoder_body = nn.Sequential(
            nn.Linear(Z_DIM + C_DIM, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU())

        # Decoder head: will pop out - hit(Bernoulli dist.), shower E(if hit exists, Gaussian dist.), log_var(variation of each voxel)
        self.head_hit = nn.Linear(1024, X_DIM)
        self.head_val = nn.Sequential(
            nn.Linear(1024, X_DIM), nn.Softplus())
        self.head_log_var = nn.Linear(1024, X_DIM)

    def reparametrize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def decode(self, z, c):
        h = self.decoder_body(torch.cat([z, c], dim=1))
        return self.head_hit(h), self.head_val(h), self.head_log_var(h)

    def forward(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparametrize(mu, log_var)
        hit_logit, val, log_var_x = self.decode(z, c)
        return hit_logit, val, log_var_x, mu, log_var