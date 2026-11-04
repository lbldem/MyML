# twostage / s2_flow1_model_v3.py - we're using Rational-Quadratic Spline coupling instead of affine coupling.

import torch
import torch.nn as nn
from nflows import flows, transforms, distributions

class ContextNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x, context=None):
        return self.net(torch.cat([x, context], dim=1))

def build_flow1_v3(DIM=45, C_DIM=1, n_layers=8, hidden=256, num_bins=8, tail_bound=6.0):
    def make_net(in_f, out_f):
        net = ContextNet(nn.Sequential(
            nn.Linear(in_f + C_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_f)))
        net.hidden_features = hidden
        return net
        
    mask = torch.zeros(DIM); mask[:DIM//2] = 1
    layers = []
    for _ in range(n_layers):
        layers.append(transforms.ActNorm(features=DIM))         # Puts on y = exp(log_scale) * x + shift
        layers.append(transforms.RandomPermutation(features=DIM))
        layers.append(transforms.LULinear(features=DIM, identity_init=True))    # Does a LU decomposition on Jacobian matrix.
        layers.append(transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask.byte(), transform_net_create_fn=make_net,
            num_bins=num_bins, tails='linear', tail_bound=tail_bound))
    return flows.Flow(transforms.CompositeTransform(layers), 
                      distributions.StandardNormal(shape=[DIM])).float()