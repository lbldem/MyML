# twostage/s2_flow1_model.py - a NF model for layer energies. Upon E_inc as conditions, NF learns joint distribution of 45 layer energies ~ (100000, 45)

import torch
import torch.nn as nn
from nflows import flows, distributions, transforms

class ContextNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x, context=None):
        return self.net(torch.cat([x, context], dim=1))

def build_flow1(DIM=45, C_DIM=1, n_layers=8, hidden=256):
    def make_net(in_f, out_f):
        return ContextNet(nn.Sequential(
            nn.Linear(in_f + C_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_f)))
    mask = torch.zeros(DIM)
    mask[:DIM//2] = 1
    layers = []
    for _ in range(n_layers):
        layers.append(transforms.RandomPermutation(features=DIM))
        layers.append(transforms.AffineCouplingTransform(
            mask=mask.byte(), transform_net_create_fn=make_net))
    return flows.Flow(transforms.CompositeTransform(layers),
                      distributions.StandardNormal(shape=[DIM])).float()