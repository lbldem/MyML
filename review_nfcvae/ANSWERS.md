# 정답과 한 줄 이유

## step1_model.py
- B1 `X_DIM + C_DIM` — encoder는 x(6480)와 c(1)를 이어 붙인 6481차원을 받는다.
- B2 `Z_DIM` — μ는 z와 같은 64차원.
- B3 `Z_DIM + C_DIM` — decoder도 condition을 받는다(65).
- B4 `nn.Linear(1024, X_DIM)` — activation 없음. ReLU를 붙이면 ℓ ≥ 0 → p ≥ 0.5로 묶여 BCE가 내려가지 않는다.
- B5 `nn.Softplus()` — 매끄럽게 양수를 보장.
- B6 `torch.exp(0.5 * log_var)` — σ = exp(½ ln σ²).
- B7 `torch.randn_like(sigma)` — 무작위성을 ε로 빼내야 μ, σ로 기울기가 흐른다.
- B8 `torch.cat([z, c], dim=1)` — dim=1: 이벤트 축(0)이 아니라 특징 축(1)으로 붙인다.
- B9 `hit_logit, val, log_var_x, mu, log_var`

## step2_train.py
- B10 `'r'`
- B11 `(log_E - log_E_min) / (log_E_max - log_E_min)`
- B12 `(x > 0).float()`
- B13 `F.binary_cross_entropy_with_logits` — sigmoid + log를 합쳐 수치적으로 안정.
- B14 `(x - val)**2 / log_var_x.exp() + log_var_x`
- B15 `-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())`
- B16–B18 `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` — 기울기는 누적되므로 매번 0으로.

## step3_collect_z.py
- B19 `model.eval()`
- B20 `torch.no_grad()`
- B21 `torch.cat([xb, Cb], dim=1)`
- B22 `model.reparametrize(mu_z, lv_z)` — aggregate posterior q(z) = 평균 q(z|x)는 μ들의 분포가 아니라 샘플들의 분포. μ만 모으면 실제보다 좁아진다.
- B23 `np.concatenate(z_list)`

## step4_flow_model.py
- B24 `torch.cat([x, context], dim=1)`
- B25 `in_f + C_DIM` — z_A(32) + c(1) = 33.
- B26 `1 - mask_A`
- B27 `i % 2 == 0`
- B28 `distributions.StandardNormal(shape=[DIM])`
- B29 `float` — 즉 `.float()`.

## step5_train_nf.py
- B30 `0.5 * Z.pow(2).sum(1) + 0.5 * DIM * np.log(2 * np.pi)`
- B31 `-flow.log_prob(zb, context=cb).mean()`

## step6_generate.py
- B32 `np.log1p(THRESHOLD_MEV / test_E * 1e4)` — 전처리 y = ln(1 + 10⁴·E/E_inc)에 E = threshold 대입.
- B33 `flow.sample(1, context=Cb).squeeze(1)` — context마다 1개씩 → (n, 1, 64) → (n, 64).
- B34 `torch.bernoulli(torch.sigmoid(hit_logit))` — 0.5로 자르지 않고 샘플링.
- B35 `torch.exp(0.5 * log_var_x)`
- B36 `(mask > 0) & (y < Yb)` — `&`가 비교보다 먼저 계산되므로 괄호 필수.
- B37 `torch.where(bad, y_new, y)`
- B38 `np.expm1(y) / 1e4`
