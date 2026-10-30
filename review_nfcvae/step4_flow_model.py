# step4_flow_model.py — conditional affine-coupling flow 설계도 (step5, step6가 import)
import torch
import torch.nn as nn
from nflows import flows, distributions, transforms

class ContextNet(nn.Module):
    """coupling 층의 subnet. 안 바뀌는 절반 z_A 뒤에 condition c를 이어 붙여 net에 넣는다."""
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x, context=None):
        return self.net(torch.cat([x, context], dim=1))                                             # [B24] x와 context를 이어 붙이기

def build_flow(DIM=64, C_DIM=1, n_layers=8):
    def make_net(in_f, out_f):
        # nflows가 in_f(안 바뀌는 차원 수)와 out_f(바뀌는 차원 수 × 2: scale, shift)를 넣어 호출한다
        return ContextNet(nn.Sequential(
            nn.Linear(in_f + C_DIM, 256), nn.ReLU(),                                                # [B25] subnet의 실제 입력 차원
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_f)))

    half = DIM // 2
    mask_A = torch.cat([torch.ones(half), torch.zeros(DIM - half)])
    mask_B = 1 - mask_A                                                                             # [B26] mask_A의 0과 1을 뒤집은 것
    layers = []
    for i in range(n_layers):
        mask = mask_A if i % 2 == 0 else mask_B                                                     # [B27] 층마다 번갈아 쓰는 조건
        layers.append(transforms.AffineCouplingTransform(
            mask=mask.byte(), transform_net_create_fn=make_net))

    base = distributions.StandardNormal(shape=[DIM])                                                # [B28] base 분포: DIM차원 표준정규 (nflows의 distributions 안)
    return flows.Flow(transforms.CompositeTransform(layers), base).float()                          # [B29] MPS는 float64를 못 쓴다

if __name__ == '__main__':
    flow = build_flow()
    z = torch.randn(5, 64); c = torch.rand(5, 1)
    print("log_prob shape:", tuple(flow.log_prob(z, context=c).shape))      # (5,)
    print("sample shape  :", tuple(flow.sample(1, context=c).shape))        # (5, 1, 64)
    print("parameters    :", sum(p.numel() for p in flow.parameters()))     # 약 72.8만
