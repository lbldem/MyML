# twostage / s6_flow2_model.py - a NF model that takes some new contexts(dim=191) than before!

import torch
import torch.nn as nn
from nflows import flows, transforms, distributions
from s2_flow1_model_v3 import ContextNet

N_LAYER, N_VOXEL = 45, 144
CTX_RAW = 1 + 1 + N_LAYER + N_VOXEL       # = 191 = dim(E_inc + E(layer=l) + N_LAYER + N_VOXEL)

def build_flow2(DIM=N_VOXEL, CTX_EMB=64, n_layers=8, hidden=512, num_bins=8, tail_bound=6.0):
    embed = nn.Sequential(
        nn.Linear(CTX_RAW, 256), nn.ReLU(),
        nn.Linear(256, CTX_EMB))

    def make_net(in_f, out_f):
        net = ContextNet(nn.Sequential(
            nn.Linear(in_f + CTX_EMB, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_f)))
        net.hidden_features = hidden
        return net

    mask = torch.zeros(DIM); mask[:DIM//2] = 1
    layers = []
    for _ in range(n_layers):
        layers.append(transforms.ActNorm(features=DIM))
        layers.append(transforms.RandomPermutation(features=DIM))
        layers.append(transforms.LULinear(features=DIM, identity_init=True))
        layers.append(transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask.byte(), transform_net_create_fn=make_net,
            num_bins=num_bins, tails='linear', tail_bound=tail_bound))
    return flows.Flow(transforms.CompositeTransform(layers),
                      distributions.StandardNormal(shape=[DIM]),
                      embedding_net=embed).float()

if __name__ == '__main__':
    import time
    flow = build_flow2()
    flow.train()
    x = torch.randn(512, N_VOXEL); ctx = torch.randn(512, CTX_RAW)
    e = flow._embedding_net(ctx)
    u, _ = flow._transform(x, context=e)
    x_back, _ = flow._transform.inverse(u, context=e)
    print("emb", tuple(e.shape), "| max(|x - x_back|)", (x - x_back).abs().max().item())
    print("n params", sum(p.numel() for p in flow.parameters()))
    t = time.time()
    for _ in range(5):
        loss = -flow.log_prob(x, context=ctx).mean(); loss.backward()
    print(f"CPU fwd+bwd per batch took: {(time.time() - t) / 5:.3f}s")