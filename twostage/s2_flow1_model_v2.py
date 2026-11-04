# twostage / s2_flow1_model_v2.py - fixing the scaling preset problem in nflows ...

import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import flows, transforms, distributions
from s2_flow1_model import ContextNet

class WideAffineCoupling(transforms.AffineCouplingTransform):
    """ Scale range is now widened up to 3, s.t we have (0.001, 3) scaling in NF net. """
    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features:, ...]
        shift = transform_params[:, :self.num_transform_features, ...]
        scale = (F.softplus(unconstrained_scale) + 1e-3).clamp(0, 3)
        return scale, shift

def build_flow1_v2(DIM=45, C_DIM=1, n_layers=8, hidden=256):
    def make_net(in_f, out_f):
        return ContextNet(nn.Sequential(
            nn.Linear(in_f + C_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_f)))
    mask = torch.zeros(DIM); mask[:DIM // 2] = 1
    layers = []
    for _ in range(n_layers):
        layers.append(transforms.ActNorm(features=DIM))
        layers.append(transforms.LULinear(features=DIM, identity_init=True))
        layers.append(WideAffineCoupling(mask=mask.byte(), transform_net_create_fn=make_net))
    return flows.Flow(transforms.CompositeTransform(layers),
                      distributions.StandardNormal(shape=[DIM])).float()