import torch
import torch.nn as nn
from nflows import flows, distributions, transforms

class ContextNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x, context=None):
        return self.net(torch.cat([x, context], dim=1))   

def build_flow(DIM=64, C_DIM=1, n_layers=8):
    def make_net(in_f, out_f):
        return ContextNet(nn.Sequential(
            nn.Linear(in_f + C_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_f)))

    half = DIM // 2
    mask_A = torch.cat([torch.ones(half), torch.zeros(DIM - half)])
    mask_B = 1 - mask_A
    layers = []
    for i in range(n_layers):
        mask = mask_A if i % 2 == 0 else mask_B
        layers.append(transforms.AffineCouplingTransform(
            mask=mask.byte(), transform_net_create_fn=make_net))

    return flows.Flow(transforms.CompositeTransform(layers),
                      distributions.StandardNormal(shape=[DIM])).float()