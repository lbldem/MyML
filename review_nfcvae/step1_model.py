# step1_model.py — Hurdle CVAE 모델 정의
# 이 파일은 실행하는 파일이 아니라 다른 스크립트가 import하는 "설계도"다.
# 확인용: python step1_model.py 로 실행하면 더미 입력으로 출력 shape를 찍어 준다.
import torch
import torch.nn as nn

X_DIM = 6480    # voxel 수
C_DIM = 1       # condition: min-max 정규화된 ln(E_inc)
Z_DIM = 64      # latent 차원

class HurdleCVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # ── Encoder: (x, c) → 256차원 특징 h ───────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(___, 1024), nn.ReLU(),            # [B1] 입력 차원은? (x와 c를 이어 붙인 것)
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU())
        self.fc_mu      = nn.Linear(256, ___)           # [B2] q(z|x,c)의 평균 μ의 차원
        self.fc_log_var = nn.Linear(256, Z_DIM)

        # ── Decoder 몸통: (z, c) → 1024차원 특징 ─────────────────────────
        self.decoder_body = nn.Sequential(
            nn.Linear(___, 256), nn.ReLU(),             # [B3] decoder 입력 차원
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU())

        # ── Decoder head 세 개 ────────────────────────────────────────
        self.head_hit = ___                             # [B4] hit logit ℓ. 범위가 (-∞, ∞)여야 한다. activation을 붙여야 할까?
        self.head_val = nn.Sequential(
            nn.Linear(1024, X_DIM), ___)                # [B5] hit일 때의 평균 μ > 0 을 보장하는 activation
        self.head_log_var = nn.Linear(1024, X_DIM)      # ln σ². activation 없음 (왜? exp를 취하면 자동으로 양수)

    def reparametrize(self, mu, log_var):
        sigma = ___                                     # [B6] log_var = ln σ² 에서 σ를 구하는 식
        eps   = ___                                     # [B7] sigma와 같은 shape의 N(0,1) 난수
        return mu + sigma * eps

    def decode(self, z, c):
        h = self.decoder_body(___)                      # [B8] z와 c를 어떻게 합쳐서 넣나? (dim에 주의)
        return self.head_hit(h), self.head_val(h), self.head_log_var(h)

    def forward(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparametrize(mu, log_var)
        hit_logit, val, log_var_x = self.decode(z, c)
        return ___                                      # [B9] train에서 loss에 필요한 다섯 개를 돌려준다 (순서: decoder 3개, encoder 2개)


if __name__ == '__main__':
    m = HurdleCVAE()
    x = torch.rand(4, X_DIM); c = torch.rand(4, C_DIM)
    for name, t in zip(['hit_logit', 'val', 'log_var_x', 'mu', 'log_var'], m(x, c)):
        print(f"{name:10s}", tuple(t.shape))
    print("val >= 0 ?", bool((m(x, c)[1] >= 0).all()))
    print("parameters:", sum(p.numel() for p in m.parameters()))   # 약 2,793만이 나와야 한다
