This repository is a training session for the CaloChallenge, which was held in 2022 (https://arxiv.org/pdf/2410.21611).

As a novice of ML in hep-ex, I aim to learn as much as one could, on my way of trying multiple models on quite oversimplified dataset, consisting of two branches - incident energies, and shower energies.

Two branches were wrapped in hdf5 format. 

Incident energy was in shape - (100000, 1) as a np array, and shower energies were in (100000, 6480) also as a np array. 6480 comes from the voxel dissection initially defined by CaloChallenge team on GeANT4 simulation.

Up to now, I have tried VAE(Variational AutoEncoder - once with PCA and once without.), CVAE(Conditional VAE), and NFCVAE(Normalised Flow + CVAE).

Introducing different models significantly improved the expectation quality of the model.

Still got works to be done!


# CaloChallenge DS2 생성 모델 진행 기록: NF부터 Hurdle CVAE + NF prior까지

Oct 29, 2026 · @Se Joon Yim

## 개요

지금까지 가장 좋은 모델은 **Hurdle CVAE(β = 1) + conditional NF prior**이고, 플롯 6종은 실제와 거의 겹치지만 classifier test에서는 AUC 0.944로 아직 쉽게 구분된다. 구분되는 이유는 voxel 사이의 상관관계를 못 만들기 때문이라는 데까지 확인했고, 그걸 고치려고 β를 0.1로 낮췄더니 이번엔 NF prior가 병목이 됐다. 지금은 NF prior를 강화하는 단계다.

목표는 CaloChallenge Dataset 2(전자 샤워, voxel 6,480개)를 머신러닝으로 빠르게 모사하는 것이고, 이 문서는 NF를 처음 배운 시점부터 현재까지 한 일과 각 단계에서 배운 것을 정리한다.

| 단계 | 바꾼 것 | 고친 문제 | 새로 드러난 문제 |
| --- | --- | --- | --- |
| PCA + NF | 6,480 → PCA 64차원, NF로 생성 | (첫 시도) | 총 에너지 분포 틀림, E\_inc 조건 없음 |
| PCA + CVAE | E\_inc를 condition으로 | 에너지 의존성 | PCA가 정보 손실, 저에너지 에너지 소실 |
| voxel CVAE | PCA 제거, 출력 ReLU | 정확한 0 표현 | decoder가 평균만 뱉음, sparsity와 다양성 틀림 |
| Hurdle CVAE | voxel을 동전 + 크기로 분리 | sparsity | 크기가 평균 하나뿐, 스펙트럼 꼬리 없음 |
| + σ 고정 샘플링 | 크기에 σ = 0.71 노이즈 | 스펙트럼 꼬리, 다양성 | 고에너지 과다, 저에너지 hit 지워짐 |
| + σ 학습 + truncated 재샘플링 | voxel마다 σ 학습, threshold 아래면 다시 뽑기 | 에너지별 과다/부족, sparsity 복구 | 생성 시 hit 17% 부족 |
| + NF prior | encoder z 분포를 NF로 학습 | prior hole, hit 개수 | classifier AUC 0.944 (상관관계 부족) |
| β = 0.1 | KL 비용 낮춤 | 최대 voxel AUC 0.787 → 0.625 | NF prior가 z 분포를 못 따라감, hit 45% 부족 |

## NF 기초

Normalizing Flow(NF)는 간단한 분포를 가역 변환해서 복잡한 분포를 만드는 모델이고, 밀도를 정확히 계산할 수 있어서 샘플만 가지고도 학습할 수 있다. 처음엔 2차원 Gaussian mixture(toy 데이터)에 4-layer NF를 학습시켜 두 군집을 재현하는 것으로 개념을 확인했다.

### 핵심 수식

```latex
u \sim \mathcal{N}(0, I), \quad x = f(u), \quad \log p(x) = \log \mathcal{N}(f^{-1}(x); 0, I) + \log \left| \det J_{f^{-1}}(x) \right|
```

- **학습**은 데이터 → base 방향이다. 실제 x를 f⁻¹로 보내고, 그 값이 N(0, I)에서 얼마나 그럴듯한지와 부피 변화(log |det J|)를 더한다.
- **생성**은 base → 데이터 방향이다. N(0, I)에서 u를 뽑아 f를 통과시킨다.
- nflows는 데이터 → base를 정방향(forward)으로 정의한다. 그래서 base → 데이터는 `flow._transform.inverse(u)`다.

### Coupling layer

임의의 신경망은 가역이 아니고 Jacobian도 계산하기 어렵다. 그래서 입력을 절반으로 나눠 한쪽만 변환한다.

```latex
z_A = x_A, \qquad z_B = x_B \odot \exp(s(x_A, c)) + t(x_A, c)
```

- x\_A는 그대로 통과하므로 거꾸로 풀 때 x\_A를 알고, 그걸로 s, t를 다시 계산해 x\_B를 복원한다. 그래서 가역이다.
- Jacobian이 삼각 행렬이라 log |det J| = Σ s로 바로 나온다.
- s, t를 만드는 **subnet**은 보통 신경망이고 가역일 필요가 없다. 출력은 s와 t를 이어 붙인 것이다.
- mask를 번갈아 쓰며 여러 층을 쌓아야 모든 차원이 변환된다. 다만 항상 같은 방식(앞 절반 ↔ 뒤 절반)으로만 나누면 같은 절반 안의 차원끼리는 서로를 직접 보지 못한다. 이게 나중에 β = 0.1 실험에서 한계로 드러났다.

### Base 분포와 Flow 객체

`flows.Flow(transform, base)`는 변환과 출발점을 묶어 `log_prob`과 `sample`을 제공한다. base는 밀도 계산과 샘플링이 둘 다 쉬운 `StandardNormal`을 쓴다. 이 안에는 학습되지 않는 상수(buffer)가 float64로 저장돼 있어서, MPS로 옮기려면 `.float()`가 필요했다.

### Context (조건부 NF)

subnet이 x\_A 옆에 조건 c를 붙여 받으면 변환 자체가 c에 따라 달라져서 p(x | c)를 배운다. nflows는 subnet을 `net(x, context)`로 부르므로, context를 쓰지 않을 땐 `ContextFreeNet`으로 받아서 버렸고, NF prior에서는 `ContextNet`으로 `torch.cat([x, context], dim=1)` 해서 넣는다.

### Loss의 의미

loss = −(1/N) Σ log p(xᵢ | cᵢ)는 두 분포를 직접 비교하는 게 아니라, 실제 샘플을 모델 분포로 **채점**하는 것이다. 수학적으로는 KL(실제 ‖ 모델) + 상수와 같아서, 이 loss를 줄이는 것은 두 분포의 KL을 줄이는 것과 같다. 이 KL은 비대칭이라, 실제 샘플이 있는 곳을 놓치면 큰 벌점이지만 없는 곳에 밀도를 주는 건 거의 벌점이 없다. 그래서 학습이 부족한 NF는 \*\*넓게 덮는 쪽(mass-covering)\*\*으로 틀린다.

## 데이터와 전처리

샤워는 세 번 변환(threshold → ÷ E\_inc → log)해서 모델에 넣고, 입사 에너지는 원본으로 저장해두었다가 모델에 넣을 때만 log → 0\~1로 바꾼다.

### Dataset 2

- `dataset_2_1.hdf5`: 학습용 10만 이벤트, `dataset_2_2.hdf5`: 테스트용 10만 이벤트 (논문 권장대로 파일 단위로 분리, 슬라이싱 없이 전부 사용)
- `showers`: (N, 6480) float64, MeV. 45층 × 144칸을 한 줄로 펼친 것이라 voxel 번호는 물리 좌표가 아니다. 평균 76%가 0
- `incident_energies`: (N, 1), MeV, 약 1 GeV \~ 1 TeV

### 샤워 변환

| 단계 | 식 | 범위 | 이유 |
| --- | --- | --- | --- |
| 원본 | E\_voxel (MeV) | 0 \~ 수천 |  |
| 1. threshold | E\_voxel < 15.15 keV → 0 |  | 데이터셋 정의. 검출기가 못 읽는 노이즈 제거 |
| 2. 비율화 | f = E\_voxel / E\_inc | 0 \~ 1 | 샤워 크기가 E\_inc에 비례하므로 스케일 제거. log에 넣을 무차원 수 |
| 3. log | y = log(1 + f × 10⁴) | 0 \~ 9.2 | 아래 참고 |

log를 취하는 이유는 세 가지다.

- f가 10⁻⁸ \~ 1로 8자릿수에 걸쳐 있어, 그대로 MSE를 쓰면 샤워 중심 몇 개만 중요해진다. 같은 10% 오차가 중심(0.1)과 바깥(0.001)에서 1만 배 다르게 잡히는데, log 공간에서는 1.2배 차이다.
- 샤워 요동은 크기에 비례(곱셈적)하는데, log는 이를 일정한 폭의 요동으로 바꿔 Gaussian(MSE) 가정과 잘 맞는다.
- `log1p`는 0을 0으로 보내고, 10⁴은 "어느 크기부터 상대 오차로 볼지"를 정하는 눈금이다.

대가도 있다. log 공간에서 평균을 내면 원래 공간으로 돌렸을 때 합이 작아진다(Jensen 부등식). 또 10⁻⁴보다 작은 f는 y 값이 거의 0이라 MSE가 신경을 쓰지 않는다. 이 두 가지가 뒤의 여러 문제의 원인이 됐다.

voxel별 표준화는 하지 않는다. 항상 0인 voxel은 표준편차가 0이라 나눌 수 없고, 표준화하면 0이 음수가 돼서 "비어 있다"는 정보가 깨진다.

### 입사 에너지 변환 (condition c)

| 단계 | 식 | 범위 | 이유 |
| --- | --- | --- | --- |
| 원본 | E\_inc (MeV) | 10³ \~ 10⁶ | threshold, 플롯 x축 등에서 원본이 필요해서 변환 없이 저장 |
| 1. log | ln E\_inc | 6.9 \~ 13.8 | 1000배 범위 압축, 데이터도 log 간격으로 샘플링됨 |
| 2. min-max | c = (ln E − min) / (max − min) | 0 \~ 1 | 신경망 입력 스케일. min, max는 `norm_params_voxel.pt`에 저장 |

생성할 때도 test 에너지에 \*\*같은 방향(정방향)\*\*으로 같은 눈금(train의 min, max)을 적용한다. c는 label이 아니라 condition(모델에게 알려주는 입력)이다.

### 생성 후 되돌리기

decoder 출력 y를 f = (eʸ − 1) / 10⁴로 되돌리고, f < 15.15 keV / E\_inc인 값을 0으로 만든다. ÷ E\_inc는 되돌리지 않고 비율 공간에서 실제와 비교한다.

### 메모리

전체를 한 번에 올리면 메모리가 98%까지 찼다. 그래서 h5py로 10,000개씩 읽고, `open_memmap`으로 디스크 파일에 바로 쓴다. 학습 때는 `np.load(..., mmap_mode='r')`와 직접 만든 `VoxelDataset`으로 필요한 줄만 읽는다. 테스트 파일들(voxel, 에너지, 생성 샤워)의 i번째 줄은 모두 같은 이벤트, 같은 에너지 조건이다.

## 첫 시도들: PCA + NF, PCA + CVAE

PCA로 6,480차원을 64차원으로 줄여 모델을 붙였지만, 병목은 모델이 아니라 PCA 자체였다. 모델을 아무리 잘 학습해도 PCA로 압축했다 복원한 샤워보다 좋아질 수 없었다.

### PCA + NF

- `IncrementalPCA`(chunk별 `partial_fit`)로 64 component, explained variance 36.8%
- unconditional NF 8층으로 PCA 좌표를 학습 → inverse PCA → voxel
- mean voxel profile은 잘 맞았지만, 총 에너지 분포(실제 0.78 피크 vs 생성 0.5 근처)와 sparsity가 틀렸다. 원인: E\_inc를 context로 주지 않았고, inverse PCA가 음수와 작은 값을 뿌려 sparsity를 망가뜨렸다.

### PCA + CVAE

- encoder `[x, c] → h → (μ_z, log σ_z²) → z`, decoder `[z, c] → x̂`. PCA 좌표는 차원별 표준화해서 MSE와 KL의 크기를 맞춘다.
- MSE는 "decoder가 σ² = ½인 Gaussian을 낸다"는 가정의 −log likelihood다. σ를 바꾸는 것은 KL 앞의 β를 바꾸는 것과 같다. `reduction='sum'`이냐 `'mean'`이냐도 사실상 β 선택이다.
- 학습: recon 61.7 → 45.5, KL 1.1 → 7.0 (KL이 커지는 건 z가 쓰이기 시작했다는 정상 신호)
- 생성 좌표의 표준편차 0.53, 분산 0.28 ≈ 1 − 45.5/64 = 0.29. decoder는 평균만 뱉어서 설명 못 한 분산이 통째로 빠지는데, 그 예측이 숫자로 맞았다.

### 비교 기준 `Real→PCA→back`

원본을 PCA로 압축했다 바로 복원한 샤워를 같이 그렸다. 이 선과 생성의 차이 = 모델이 못 배운 것, 이 선과 실제의 차이 = PCA가 버린 것이다. CVAE는 이 선을 거의 따라갔고, 차이는 대부분 PCA 쪽에 있었다.

### Response 문제와 Jensen 효과

실제 샤워는 에너지와 무관하게 총 에너지 비율이 약 0.78로 평평한데, PCA 복원은 저에너지에서 0.24까지 떨어졌다. 저에너지 샤워는 hit이 적고 랜덤한 위치에 있어서 64개 component로 표현이 안 되고, 복원이 뾰족한 값을 뭉갠다. 뭉개짐이 log 공간에서 일어나면 expm1로 돌렸을 때 합이 줄어든다. 예: expm1(6) + expm1(0) = 402인데 expm1(3) + expm1(3) = 38.

나중에 PCA 없이도 같은 곡선이 나오는 걸 보고 원인을 고쳐 적었다. 진짜 원인은 PCA가 아니라 **log 공간에서 평균을 내는 모든 장치**(PCA의 최소제곱 복원이든 MSE decoder든)였다.

## voxel CVAE와 Hurdle 모델

PCA를 빼고 voxel 6,480개에 바로 CVAE를 붙였고, 그 다음 decoder가 voxel마다 "hit이 있나(동전)"와 "있다면 얼마나(크기)"를 따로 말하게 바꾸어 sparsity를 해결했다.

### voxel CVAE

- encoder 6481 → 1024 → 512 → 256 = h, `fc_mu` / `fc_log_var` 256 → 64 = z. decoder는 거울 구조 65 → 256 → 512 → 1024 → 6480. 파라미터 약 1,300만 개
- decoder 끝에 **ReLU**: log 공간 값은 0 이상이고 76%가 정확히 0이라, ReLU가 음수를 막고 정확한 0을 표현하게 해준다. (출력 activation은 출력값의 범위로 고른다. PCA 좌표는 음수가 많아 못 붙였다.)
- 학습(30 epoch, β = 1): recon 약 1,092, KL 약 9.6. KL이 17 → 7.7 → 10으로 움직인 건, decoder가 z를 읽는 법을 배우면서 정보를 다시 담기 시작한 것이다.
- 예측과 달리 KL이 커지지 않았다. 샤워의 이벤트별 차이 대부분이 압축 가능한 패턴이 아니라 노이즈(랜덤한 hit 위치)라서, z에 담아도 recon이 별로 안 줄기 때문이다.

생성 결과는 사실상 "주어진 에너지에서의 평균 샤워"였다. sparsity는 0.63에 벽이 생겨 저에너지는 너무 빽빽하고 고에너지는 너무 비었고, 이벤트 간 다양성과 스펙트럼 양쪽 꼬리가 사라졌다.

### Hurdle: 동전 + 크기

"가끔 찍히는" voxel(예: hit 확률 30%, hit이면 값 4)에서 MSE decoder는 평균 1.2를 뱉는다. 이러면 모든 샤워에 1.2가 칠해지고 오차도 voxel당 3.4나 나온다. hurdle은 voxel 하나의 확률을 이렇게 본다.

```latex
p(y) = \begin{cases} 1 - p & y = 0 \\ p \cdot \mathcal{N}(y; \mu_x, \sigma_x^2) & y > 0 \end{cases}
```

−log를 취하면 recon이 두 항으로 나뉜다. hit 여부 h에 대해,

```latex
-\log p(y) = \underbrace{-[h \log p + (1-h) \log(1-p)]}_{\text{BCE, 모든 voxel}} + \underbrace{h \cdot \tfrac{(y-\mu_x)^2}{2\sigma_x^2} + h \log \sigma_x}_{\text{hit인 voxel만}} + \text{const}
```

- decoder 몸통은 공유하고 머리를 나눈다: `head_hit`(logit, activation 없음), `head_val`(μ\_x, Softplus), 나중에 `head_log_var`(log σ\_x², activation 없음)
- logit에 ReLU를 붙이면 p ≥ 0.5가 돼서 "거의 안 찍힌다"를 표현할 수 없다. 실제로 그 버그로 BCE가 6480 × ln 2 ≈ 4,490 근처에 갇혔고, 계산으로 알 수 있던 시작값 덕분에 3 epoch 만에 잡았다.
- head\_val은 ReLU 대신 Softplus: 이제 0은 head\_hit이 맡아서 정확한 0이 필요 없고, dying ReLU 걱정도 없다.
- `binary_cross_entropy_with_logits`는 sigmoid와 BCE를 함께 계산해 log(0) 폭발을 피한다. hit만 고르는 건 mask 곱셈(`hit * (x - val) ** 2`)이다. 정답 `hit = (x > 0).float()`은 항상 데이터에서 온다.

학습(σ 고정, 30 epoch): BCE 4,391 → 1,632, MSE 359, KL 27.7 → 7.4. BCE가 예상(수백\~천)보다 높은 건 hit 여부가 애매한 경계 voxel이 3,000개 이상이라 줄일 수 없는 BCE가 크기 때문이다. KL이 높게 시작해 줄어든 건, 초반엔 z가 c의 일(에너지 정보)을 대신하다가 decoder가 c를 쓰게 되자 중복이 됐기 때문이다.

생성에서는 p로 실제 동전을 던진다(`torch.bernoulli(torch.sigmoid(logit))`). sparsity가 실제와 거의 같아졌다. 하지만 hit 크기는 여전히 μ\_x 하나라 스펙트럼 양쪽 꼬리가 없었고, Response는 U자 모양(고에너지만 일치)이었다.

## σ 학습과 생성 절차

hit 크기에도 랜덤함을 넣고(σ 샘플링), 그 σ를 voxel마다 학습시키고, threshold 아래로 떨어진 값은 다시 뽑아 동전이 정한 hit을 지켰다.

### 모델 안의 두 Gaussian

|  | encoder 쪽 | decoder 쪽 |
| --- | --- | --- |
| 레이어 | `fc_mu`, `fc_log_var` | `head_val`, `head_log_var` |
| 무엇의 분포 | latent z (64차원) | hit인 voxel의 에너지 크기 (6,480차원) |
| 수식 | q(z \| x, c) = N(μ\_z, σ\_z²) | p(y \| z, c, hit) = N(μ\_x, σ\_x²) |
| 뜻 | 이 샤워를 요약하면 z가 어디쥐이고 얼마나 불확실한가 | 같은 z, c에서도 샤워마다 얼마나 흔들리는가 |
| 손실 항 | KL | NLL |
| 생성에 쓰나 | 안 씀 (prior가 대신) | 씀 |

"σ 샘플링", "σ 학습"의 σ는 모두 decoder 쪽 σ\_x다. 코드에서 encoder 쪽은 `_z`, decoder 쪽은 `_x`를 붙여 구분한다. 생성 결과의 랜덤함은 z를 뽑을 때와 voxel마다(동전 + σ\_x) 두 곳에서 온다.

### σ 고정 샘플링 (σ = 0.71)

스펙트럼 꼬리, 이벤트 간 std, mean profile이 크게 좋아졌다. 하지만 Response가 10 GeV에서 0.59(부족), 700 GeV에서 1.14(과다)로 반대 방향으로 틀렸는데 전체 평균은 0.801 vs 0.785로 맞아 보였다. 평균이 맞아도 조건별로 나눠 봐야 한다는 교훈이다.

- 고에너지 과다: log 공간에서 σ로 흔들면 평균이 e^(σ²/2) ≈ 1.28배로 부풀는데, 실제 고에너지 voxel은 거의 안 흔들린다.
- 저에너지 부족: μ가 threshold에 가까운 hit이 흔들리다 threshold 아래로 내려가 지워졌다 (이벤트당 1,502개 중 17%).

### σ 학습

`head_log_var`를 추가하고 MSE를 Gaussian NLL로 바꿨다. log σ\_x²를 σ\_x² = ½로 고정하면 예전 MSE + 상수가 되므로, 예전 loss의 일반화다.

```latex
\text{NLL} = \tfrac{1}{2} \sum_i h_i \left[ \frac{(x_i - \mu_{x,i})^2}{\sigma_{x,i}^2} + \log \sigma_{x,i}^2 \right]
```

- 첫 항은 σ가 크면 오차를 봐주고, 둘째 항은 σ를 키우는 핑계에 벌점을 준다. 균형점은 σ² = 실제 오차 제곱의 평균이다.
- σ → 0이면 log σ² → −∞로 도망갈 수 있어 `log_var_x.clamp(-7, 3)`으로 막는다 (σ ≈ 0.03 \~ 4.5).
- NLL은 음수가 될 수 있다. 최적점에서 voxel당 ½(1 + log σ²)라서 평균 log σ² ≈ 2 × NLL / hit 개수 − 1로 어림할 수 있다.

학습(β = 1, 30 epoch): BCE 1,633 (σ 고정 때와 같아 hit 쪽과 깔끔하게 분리됨), NLL −761 (예측 −410보다 낮음), KL 34 → 10.7. NLL로 어림한 σ ≈ 0.38이고, 생성에서 찍은 hit voxel의 평균 σ는 0.42 \~ 0.53이었다. 고정값 0.71이 너무 컸던 것이다.

### 생성 절차 (최종)

1. z를 얻는다 (N(0, I), 재구성이면 encoder, NF prior면 NF 샘플)
2. `model.decode(z, c)` → hit\_logit, μ\_x, log σ\_x² (학습 때와 같은 clamp)
3. p = sigmoid(logit), mask = bernoulli(p)
4. y = μ\_x + σ\_x · ε
5. mask가 1인데 y < threshold(log 공간, 이벤트별)이면 **새 ε로** 다시 뽑기, 최대 10번 (truncated Gaussian). 남은 건 threshold × 1.001로 올림
6. y ≤ log(1 + 10⁴) ≈ 9.21로 자름 (voxel 하나가 E\_inc를 넘을 수 없음). 학습이 부족한 바깥 voxel의 큰 σ로 비율 86짜리 이벤트가 터진 뒤 추가했다.
7. y\_gen = y × mask

실행은 2,000 이벤트 chunk로 나눠 MPS에서 한다. 한 번에 2만 개를 올리면 M1 GPU 메모리가 부족하고, MPS는 에러를 내지 않고 0을 돌려주므로 `hits from coin` 같은 출력이 안전장치 역할을 한다. chunk 안에서는 N, C, Y\_thr 대신 n, Cb, Yb를 써야 한다.

## 재구성 실험과 NF prior

N(0, I)에서 z를 뽑으면 hit이 17% 부족했는데, encoder가 만든 z로 재구성하면 거의 완벽했다. 문제는 decoder가 아니라 생성용 z였고, encoder z의 실제 분포를 NF로 배워서 해결했다.

&#91;embedded content: Hurdle CVAE + NF prior 구조 · 학습 2단계 + 생성\]

VAE는 학습이 끝난 뒤 고정하고, 두 번째 단계에서 NF만 따로 학습한다. 생성할 때는 encoder가 빠지고 NF가 그 자리를 대신한다.

### 재구성 실험 (ablation)

생성 스크립트에 MODE 스위치를 두고 z를 어디서 얻을지만 바꿨다. `reconstruct`는 test 샤워를 encoder에 넣고 `reparametrize`로 z를 샘플링한다(학습 때와 같은 조건). decoder의 성능 상한을 재는 것이고, PCA 때 `Real→PCA→back`과 같은 역할이다.

| (β = 1) | 실제 | N(0, I) 생성 | 재구성 | NF prior 생성 |
| --- | --- | --- | --- | --- |
| hit 개수/이벤트 | 1,617 | 1,346 | 1,650 | 1,644 |
| 에너지 비율 | 0.785 | 0.701 | 0.795 | 0.787 |
| hit voxel 평균 σ\_x |  | 0.53 | 0.42 | 0.42 |
| z 차원별 평균 범위 |  | −0.02 \~ 0.01 | −0.68 \~ 0.41 | −0.58 \~ 0.51 |
| z 차원별 std 범위 |  | 0.99 \~ 1.01 | 0.85 \~ 1.09 | 0.89 \~ 1.18 |

- N(0, I) 생성에서 σ\_x가 커지는 건 decoder가 낯선 z를 받아 "잘 모르겠다"고 말하는 것이다.
- 이 현상을 **prior hole**이라고 한다. encoder z들의 전체 분포(aggregate posterior)가 N(0, I)를 다 덮지 못해서 생긴다. "재구성은 좋은데 생성은 나쁘다"는 VAE의 대표적 약점이다.
- 재구성에서도 스펙트럼 왼쪽 꼬리(10⁻⁷ \~ 10⁻⁵)는 10배 부족했다. 그러므로 이건 z가 아니라 decoder 쪽 문제다.

### Conditional NF prior (two-stage VAE)

1. 학습된 encoder에 train 샤워 10만 개를 넣어 샘플링한 z를 모은다 (`calo_collect_z.py` → `train_z.npy`, `train_c.npy`). μ\_z만 모으면 퍼짐이 빠져 분포가 좁아진다.
2. 64차원 conditional NF(마스크를 번갈아 쓰는 affine coupling 8층, subnet 256)로 p(z | c)를 학습한다. encoder도 c를 받으므로 z 분포는 에너지에 따라 다를 수 있다.
3. 생성할 때 `flow.sample(1, context=c).squeeze(1)`로 이벤트마다 z를 하나씩 뽑는다. 첫 인자는 차원이 아니라 조건당 개수다.

VAE의 KL(이벤트 평균)은 "z가 담은 정보량 + z 분포와 N(0, I)의 차이"의 합이다. NF가 줄일 수 있는 건 뒷부분뿐이라, 기준선(N(0, I) 아래 NLL) 대비 개선폭은 0과 KL 사이에 있다. β = 1에서 기준선 88.43, 50 epoch 후 gain 5.51 nat(KL 10.7 중 최소 40% 이상이 prior hole이었다는 뜻).

## Classifier test

플롯은 거의 다 맞았지만, 실제와 생성을 구분하는 DNN은 **AUC 0.944**(β = 1, NF prior)로 쉽게 구분했다. 원인은 모델이 voxel 사이의 상관관계를 못 만드는 것이다.

### 방법

- 실제 = `test_voxel.npy`, 생성 = 같은 test 에너지 조건으로 만든 `gen_voxel.npy` (둘 다 log 공간), 입력은 voxel 6,480개 + c
- 이벤트 번호로 train 60% / validation 20% / test 20% 분할. validation으로 가장 좋은 epoch를 고르고(early stopping, `copy.deepcopy(state_dict)`), 채점은 test 구간에서 한 번만
- Dataset은 이벤트마다 실제(짝수 k)와 생성(홀수 k)을 번갈아 내서 정확히 반반
- classifier는 6481 → 512 → 512 → 512 → 1, BCE. epoch 2에서 validation이 바닥을 찍고 그 뒤는 외우기(overfitting) 시작. 그래서 이 AUC는 "차이가 최소한 이만큼"인 하한이다.

**AUC**는 진짜 하나와 가짜 하나를 짝지었을 때 진짜 쪽 점수가 더 높을 확률이다. 0.5 기준선을 쓰는 정확도와 달리 점수의 순서만 본다. 이 시험에서는 classifier가 채점관이고 생성 모델이 수험생이라, **0.5에 가까울수록 생성 모델이 좋다**.

### Null test

생성 자리에 다른 이벤트의 실제 샤워(`np.roll(idx, 1)`로 짝지음, 각자 자기 c)를 넣었더니 BCE가 ln 2 = 0.6931에 머물고 AUC 0.5000이 나왔다. 라벨이 데이터 순서나 c를 통해 새지 않는다는 뜻이다. 이 방식에서는 같은 샤워가 두 라벨로 한 번씩 나오므로 classifier가 모두 0.5로 답해서 정확히 0.5가 되었다.

### 에너지별 AUC

&#91;embedded content: calo\_classifier.py · test 구간 2만 이벤트(실제 1 + 생성 1씩) · β = 1과 β = 0.1\]

고에너지 샤워는 총 에너지 흔들림이 아주 작아(700 GeV에서 0.78 ± 0.015, σ/E ∝ 1/√E) 가짜가 숨을 틈이 없다. β = 0.1은 모든 구간에서 β = 1보다 조금 더 들켰다.

### Feature AUC: 무엇으로 들키는가

물리량 하나를 classifier 점수 대신 넣어 AUC를 재면 "그 양 하나만으로 얼마나 구분되나"가 나온다. 높을수록 모델이 그 양을 **틀리게** 만든 것이다. 방향을 무시하려고 max(AUC, 1 − AUC)를 쓰고, 중앙값으로 방향을 본다. 평균은 같고 퍼짐만 다른 차이는 잘 못 잡는다는 한계가 있다.

| β = 1, NF prior | 1–4 GeV | 30–60 GeV | 250–1000 GeV |
| --- | --- | --- | --- |
| 총 에너지 비율 | 0.526 | 0.560 | **0.858** (생성 0.805 > 실제 0.781) |
| hit 개수 | 0.538 | 0.564 | 0.506 |
| hit의 평균 log 값 | 0.578 | 0.559 | 0.666 |
| 최대 voxel 비율 | 0.608 | **0.787** (생성 0.0067 < 실제 0.0100) | 0.604 |

- 저에너지: 합계형 양으로는 거의 안 갈리는데 classifier는 0.88. 공간 패턴(이웃 voxel 관계)으로 구분하고 있다.
- 고에너지: hit 개수는 맞는데 총 에너지가 틀린다. 실제 voxel들은 서로 반대로 움직여(에너지 보존) 합이 거의 안 흔들리는데, 생성은 voxel마다 독립적으로 흔들어 상쇄가 안 된다.
- 중간 에너지: 최대 voxel이 작다. 예측(크다)은 틀렸다. 실제 샤워는 중심 근처 voxel 중 정확히 하나에 에너지가 몰리는데, 독립적인 동전으로는 "하나도 안 뜨거운" 이벤트가 자주 생긴다.

세 단서의 방향은 제각각이지만 뿌리는 하나다. decoder는 z가 정해지면 voxel을 **서로 독립적으로** 뽑는다. 그러므로 voxel 사이의 상관관계는 오직 z를 통해서만 생길 수 있는데, β = 1의 z는 KL ≈ 10.7 nat로 정보를 아주 적게 담고 있다.

## β = 0.1 실험과 temperature 진단

β를 0.1로 낮춰 z가 상관관계 정보를 더 담게 했더니 decoder는 좋아졌지만, z 분포가 복잡해져 NF prior가 따라가지 못했다. 병목이 decoder에서 prior로 옮겨간 것이다.

### 동기

상관관계의 유일한 통로가 z이므로, KL 비용을 줄여 z에 정보를 더 담게 하면 "어느 voxel이 뜨겁나", "샤워 축이 어디냐" 같은 정보가 들어갈 수 있다. 예전엔 KL이 커지면 prior hole이 커져 위험했지만 NF prior가 그걸 메울 거라고 봤다. 코드는 git branch `b01`, 결과 파일은 TAG `b01`로 분리했다 (branch는 코드를, TAG는 `.gitignore`된 결과 파일을 나눈다).

### 결과

|  | β = 1 | β = 0.1 |
| --- | --- | --- |
| VAE KL (30 epoch) | 10.7 | 70.4 (아직 감소 중) |
| BCE / NLL | 1,633 / −761 | 1,622 / −774 |
| NF 기준선 NLL / gain (50 epoch) | 88.43 / 5.51 | 74.28 / 9.42 |
| classifier AUC | 0.944 | 0.953 |
| 최대 voxel AUC (30–60 GeV) | 0.787 | 0.625 |
| hit 개수 (250–1000 GeV, 실제 4,152) | 4,182 | 2,132 |
| hit 개수 재구성 / NF prior (실제 1,617) | 1,650 / 1,644 | 1,606 / 885 |

- 노린 것(최대 voxel)은 좋아졌지만, NF prior 생성의 hit이 거의 반토막 났다.
- 재구성은 1,606개로 정상이다. decoder는 멀쩡하고 NF prior가 문제다.
- NF 생성에서 σ\_x가 clamp 상한(4.48)에 붙고, Y\_MAX로 잘리는 값도 늘었다. decoder가 낯선 z를 받은 증상이다.
- z 표준편차 범위: 재구성 0.48 \~ 1.10, NF 샘플 1.04 \~ 1.57. NF가 좁아야 할 차원까지 넓게 뿌린다.

넓게 뿌리는 이유는 NF loss가 KL(실제 ‖ 모델)이기 때문이다. 실제 z가 있는 곳을 놓치면 큰 벌점이지만, 없는 곳에 밀도를 주는 건 거의 벌점이 없어서, 덜 학습된 NF는 넓게 덮는 쪽(mass-covering)으로 틀린다.

### Temperature 진단

base에서 u를 표준편차 T로 뽑고 `flow._transform.inverse(u, context=c)`로 z를 얻었다. 처음엔 코드가 적용되지 않아 z std가 T와 무관하게 같게 나왔는데, "손잡이를 돌렸을 때 바뀌어야 할 값이 바뀌는가"로 버그를 찾았다.

| T (β = 0.1) | z std 범위 | hit 개수 | 평균 σ\_x | 에너지 비율 |
| --- | --- | --- | --- | --- |
| 1.0 | 1.04 \~ 1.57 | 887 | 0.54 | 0.837 |
| 0.8 | 0.83 \~ 1.29 | 1,077 | 0.51 | 0.808 |
| 0.6 | 0.64 \~ 1.04 | 1,291 | 0.47 | 0.802 |
| 재구성 (목표) | 0.48 \~ 1.10 | 1,606 | 0.42 | 0.801 |

T를 줄이면 세 숫자가 함께 재구성 쪽으로 움직여 "NF가 z를 너무 넓게 뽑는다"는 가설이 확인됐다. 하지만 T는 모든 방향을 같은 비율로 줄여서, 좁은 차원(0.64 → 목표 0.48)과 넓은 차원(1.04 → 목표 1.10)을 동시에 맞출 수 없다. NF 자체가 차원마다 다른 모양을 배워야 한다.

## 다음 단계와 배운 교훈

다음은 β = 0.1의 z 분포를 따라갈 수 있게 NF prior를 강화하는 것이다. TAG는 `b01_nf16`처럼 구조 변경을 표시해 지금 NF와 비교한다.

### NF prior 강화 계획

- [ ] coupling 사이에 `transforms.RandomPermutation(features=64)`을 넣어 매번 다른 차원 조합으로 나눔 (순서는 buffer로 저장돼 불러올 때 그대로 복원됨)
- [ ] layer 8 → 16, subnet 256 → 512
- [ ] z를 차원별로 표준화해서 학습 (`z_norm_{TAG}.npz`에 mean, std 저장), 생성 때 z = z\_n · std + mean
- [ ] 생성한 z의 std 범위가 재구성 z(0.48 \~ 1.10)에 가까워지는지, hit 개수가 1,617에 가까워지는지 확인
- [ ] test 샤워로도 z를 모아 NF의 test NLL로 overfitting 확인
- [ ] classifier test와 feature AUC를 다시 돌려 β = 1 모델(0.944)과 비교

그 뒤에 남은 후보는 이렇다: 총 에너지를 따로 학습해 생성 샤워를 스케일링 (CaloFlow식 2단계), 스펙트럼 왼쪽 꼬리(미세한 hit의 크기), β 중간값(0.3), 층별 물리량(45층 에너지, 샤워 중심과 폭).

### 교훈

1. **돌리기 전에 예측한다.** BCE 시작값(6480 × ln 2) 같은 계산 가능한 숫자가 버그를 3 epoch 만에 잡았고, 틀린 예측(PCA가 원인, 최대 voxel이 클 것)에서 오히려 원인을 더 정확히 찾았다.
2. **부품을 하나씩 바꿔 원인을 가른다.** `Real→PCA→back`, 재구성 모드, null test는 "모델 문제냐 파이프라인 문제냐", "decoder냐 prior냐", "모델이냐 채점이냐"를 갈랐다.
3. **평균이 맞아도 조건별로 나눠 본다.** σ 고정 때 전체 에너지는 맞았지만 저에너지 부족과 고에너지 과다가 상쇄된 것이었다.
4. **플롯이 맞아도 상관관계는 모른다.** 1차원 분포들이 다 맞아도 classifier는 0.944로 구분했다.
5. **두 부품이 맞물리면 병목이 옮겨간다.** β를 줄여 decoder 쪽을 풀었더니 prior 쪽이 막혔다.
6. **손잡이를 돌렸으면 바뀌어야 할 값도 본다.** 기준선(T = 1)이 맞는 건 필요조건뿐이다.
7. **조용한 실패를 조심한다.** MPS 메모리 부족은 에러 없이 0을 돌려주었고, 인자가 빠진 함수는 전역 변수를 집어 에러 없이 돌 수 있었다. `assert`와 안전장치 print를 둔다.

### 자주 나온 코드 실수

- 호출은 `()`, 인덱싱은 `[]` (`nn.Linear[...]`, `iter_chunks[...]`). method는 괄호까지 (`.numpy` vs `.numpy()`)
- 원소 하나 `G[start]` vs 구간 `G[start:end]`, `perm[a]` vs `perm[a:]`
- `&`는 비교보다 먼저 계산됨 → `(mask > 0) & (y < Yb)`
- chunk 안에서는 전체(N, C, Y\_thr) 대신 chunk(n, Cb, Yb)
- "다시 뽑기"는 새 난수로 (`torch.randn_like` 매번)
- logit → 확률은 sigmoid, `.mean()`으로 뭉개지 않기. mask는 고르기(`v[m]`, 1차원)가 아니라 곱하기(`v * m`, shape 유지)
- `(n, 1)`과 `(n,)` 섞으면 broadcasting으로 `(n, n)` → 분석용은 `[:, 0]`로 1차원
- MPS는 float64 불가 (`.float()`, 파이썬 숫자 사용). 모델, 입력, 난수를 같은 장치에
- 환경: 설치는 항상 `python -m pip`. 가상환경 경로에 `-m venv`를 다시 실행하면 `pyvenv.cfg`가 덮어써진다

## 파일과 스크립트 지도

코드(`.py`)는 git으로 맥과 윈도우를 오가고, 데이터와 모델(`.hdf5`, `.npy`, `.pt`, `.npz`)은 `.gitignore`로 빼서 각 컴퓨터에서 만든다. 실험별 결과는 파일 이름의 TAG로 나눈다.

### 파이프라인 순서

| 순서 | 스크립트 | 읽는 것 | 쓰는 것 |
| --- | --- | --- | --- |
| 1 | `calo_prepare_voxel.py` | `dataset_2_1.hdf5`, `dataset_2_2.hdf5` | `train_voxel.npy`, `test_voxel.npy`, `train_energies.npy`, `test_energies.npy` |
| 2 | `calo_cvae_hurdle_train.py` (+ `calo_cvae_hurdle_model.py`) | `train_voxel.npy`, `train_energies.npy` | `calo_cvae_hurdle_{TAG}.pt`, `norm_params_voxel.pt` |
| 3 | `calo_collect_z.py` | VAE 가중치, train 데이터 | `train_z_{TAG}.npy`, `train_c.npy` |
| 4 | `calo_z_prior_nf.py` (+ `z_prior_model.py`) | `train_z_{TAG}.npy`, `train_c.npy` | `z_prior_nf_{TAG}.pt` |
| 5a | `calo_cvae_hurdle_generate.py` (NFCVAE/classifier) | VAE, NF 가중치, test 데이터, hdf5 | 비교 플롯 6종 `.png` |
| 5b | `calo_cvae_hurdle_generate.py (NFCVAE/classifier`) | VAE, NF 가중치, `test_energies.npy` | `gen_voxel_{TAG}.npy` (10만 × 6480, log 공간) |
| 6 | `calo_classifier.py` (NULL\_TEST 스위치) | `test_voxel.npy`, `gen_voxel_{TAG}.npy` | `clf_best_{TAG}.pt`, AUC 출력 |
| 7 | `feature_auc.py` | `test_voxel.npy`, `gen_voxel_{TAG}.npy` | feature별 AUC와 중앙값 출력 |

### 파일 내용

| 파일 | shape, 자료형 | 내용 |
| --- | --- | --- |
| `train_voxel.npy`, `test_voxel.npy` | (100000, 6480) float32 | threshold → ÷ E\_inc → log(1 + f × 10⁴) |
| `train_energies.npy`, `test_energies.npy` | (100000, 1) float32 | 입사 에너지 원본, MeV |
| `train_z_{TAG}.npy` | (100000, 64) float32 | train 샤워의 encoder z 샘플 |
| `train_c.npy` | (100000, 1) float32 | train condition c (0 \~ 1) |
| `gen_voxel_{TAG}.npy` | (100000, 6480) float32 | NF prior 생성 샤워. i번째 = test 에너지 i번째 조건 |
| `norm_params_voxel.pt` | 딕셔너리 | `log_E_min`, `log_E_max` (numpy 숫자라 `weights_only=False` 필요) |
| `calo_cvae_hurdle_{TAG}.pt` | state\_dict | encoder, 몸통, 머리 3개의 가중치 |
| `z_prior_nf_{TAG}.pt` | state\_dict | coupling layer들의 가중치 (구조는 `build_flow()`가 다시 만듦) |

`.npy`는 배열 하나(헤더 + 숫자가 순서대로)라 memmap으로 필요한 줄만 읽을 수 있고, `.pt`는 pickle 기반이라 여러 객체를 담을 수 있다. state\_dict에는 구조가 없으므로 모델 클래스와 `build_flow()`를 별도 파일로 두고 import한다. 모델 파일에는 클래스만 두고, 테스트는 `if __name__ == '__main__':` 안에 둔다.



# CaloChallenge DS2 코드 설명서

Oct 29, 2026 · @Se Joon Yim

## 읽는 법

이 문서는 맥의 `MyML` 폴더, main branch(β = 1)에 있는 실제 코드를 기준으로 스크립트마다 **목적, 주요 객체와 변수, 흐름**을 설명한다. 스크립트는 아래 순서로 돌아가고, 앞 단계가 만든 파일을 뒤 단계가 읽는다.

| 순서 | 파일 (폴더) | 목적 |
| --- | --- | --- |
| 1 | `calo_prepare_voxel.py` (`cvae_without_pca/`) | hdf5 원본을 학습용 log 공간 voxel과 에너지 파일로 변환 |
| 2 | `calo_cvae_hurdle_model.py` (`hurdle/`, `NFCVAE/classifier/`) | Hurdle CVAE 모델 구조 (클래스만) |
| 3 | `calo_cvae_hurdle_train.py` (`hurdle/`) | Hurdle CVAE 학습 |
| 4 | `calo_collect_z.py` (`NFCVAE/classifier/`) | 학습된 encoder로 train z 모으기 |
| 5 | `calo_z_prior_nf.py` (`NFCVAE/`) + `z_prior_model.py` (`NFCVAE/classifier/`) | z 분포를 배우는 conditional NF 학습, NF 구조 함수 |
| 6 | `calo_cvae_hurdle_nf_generate.py` (`NFCVAE/classifier/`) | MODE별 생성 + 실제와 비교하는 플롯 6종 |
| 7 | `calo_cvae_hurdle_generate.py` (`NFCVAE/classifier/`) | NF prior로 10만 개 생성해 `gen_voxel.npy`로 저장 |
| 8 | `calo_classifier.py` (`NFCVAE/classifier/`) | 실제 vs 생성 분류기로 채점 (AUC) |
| 9 | `feature_auc.py` (`NFCVAE/classifier/`) | 물리량 하나씩 AUC를 재서 들키는 이유 찾기 |

`cvae_with_pca/`, `cvae_without_pca/`의 나머지 파일과 `practise/`는 이전 단계 기록이다. `hurdle/`과 `NFCVAE/classifier/`에는 같은 이름의 파일이 두 벌 있는데, 이 문서는 머리가 3개인 최신 버전(`NFCVAE/classifier/`)을 기준으로 한다.

### 공통 규칙

- **이름 접미사**: `_z`는 latent z의 분포(encoder 쪽), `_x`는 voxel 크기의 분포(decoder 쪽)다. `mu, log_var`(encoder)와 `val, log_var_x`(decoder)를 섞지 않기 위한 것이다.
- **대문자와 소문자**: `N`, `C`, `Y_thr`는 전체, `n`, `Cb`, `Yb`는 chunk 하나다. chunk 반복문 안에서는 소문자만 쓴다.
- **numpy와 torch**: 파일에서 읽은 건 numpy 배열, 모델에 넣는 건 torch 테서다. 경계에서 `torch.tensor(...)`, `torch.from_numpy(...)`, `.cpu().numpy()`로 바꾼다.
- **장치**: `device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')`. 모델, 입력, 난수를 같은 장치에 둔다. 윈도우에서는 같은 코드가 CPU로 돈다.
- **실행 위치**: 파일 이름은 상대경로라 터미널의 현재 폴더(cwd) 기준으로 찾는다. 데이터와 같은 폴더에서 실행한다.
- **TAG**: b01 branch에서는 결과 파일 이름에 TAG를 붙여 β = 1 결과와 나눈다. main의 코드에는 TAG가 없다.

## calo\_prepare\_voxel.py

**목적**: hdf5 원본을 읽어 샤워는 log 공간 값으로, 입사 에너지는 원본 그대로 `.npy`에 저장한다. 10만 × 6480 배열을 한 번에 메모리에 올리지 않으려고 10,000개씩 끊어서 디스크에 바로 쓴다.

### 객체와 변수

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `THRESHOLD_MEV` | float, 0.01515 | 15.15 keV. 이보다 작은 voxel은 0으로 (데이터셋 정의) |
| `CHUNK` | int, 10000 | 한 번에 메모리에 올릴 이벤트 수 |
| `preprocess(showers, energies)` | 함수 | chunk 하나를 threshold → ÷ E\_inc → log1p(× 10⁴)로 바꿔 float32로 돌려줌 |
| `convert(src, dst_x, dst_e)` | 함수 | hdf5 파일 하나를 통째로 변환해서 두 `.npy`에 저장 |
| `f` | `h5py.File` | 열린 hdf5. `f['showers']`, `f['incident_energies']`는 대괄호로 범위를 줄 때만 디스크에서 읽는다 |
| `n, d` | int | 이벤트 수(100000), voxel 수(6480) |
| `X` | memmap, (n, 6480) float32 | 샤워를 쓸 디스크 위의 배열 |
| `E` | memmap, (n, 1) float32 | 에너지를 쓸 디스크 위의 배열 |
| `start, end` | int | 이번 chunk의 범위. `end = min(start + CHUNK, n)`으로 마지막 chunk가 넘치지 않게 |
| `s`, `e` | ndarray | 이번 chunk의 샤워 (10000, 6480), 에너지 (10000, 1). 둘 다 float64, MeV |

### 흐름

```python
def preprocess(showers, energies):
    showers = showers.copy()                       # 원본을 건드리지 않으려고 복사
    showers[showers < THRESHOLD_MEV] = 0.0         # boolean indexing으로 골라서 0 대입
    return np.log1p(showers / energies * 1e4).astype(np.float32)
```

- `showers / energies`: (10000, 6480) ÷ (10000, 1). broadcasting으로 각 행(이벤트)을 자기 에너지로 나눈다.
- `np.log1p(a)` = log(1 + a). 0을 0으로 보내서 빈 voxel이 그대로 남는다.
- float32로 바꾸면 크기가 절반(5.2 → 2.6 GB)이고 PyTorch 기본 자료형과 맞는다.

```python
X = open_memmap(dst_x, mode='w+', dtype=np.float32, shape=(n, d))
for start in range(0, n, CHUNK):
    end = min(start + CHUNK, n)
    X[start:end] = preprocess(s, e)               # 이 구간만 디스크에 기록
X.flush()                                          # 남은 내용까지 확실히 기록
```

- `open_memmap`은 디스크의 `.npy`를 메모리 배열처럼 다루게 해준다. `mode='w+'`는 새로 만들고 읽기/쓰기.
- `X[start:end] = ...`는 구간(콜론) 대입이다. `X[start]`이면 한 줄이라 shape이 안 맞는다.
- `with h5py.File(...) as f:` 블록을 나가면 파일이 자동으로 닫힌다.

마지막 두 줄이 train(`dataset_2_1`)과 test(`dataset_2_2`)에 각각 `convert`를 부른다. 결과: `train_voxel.npy`, `train_energies.npy`, `test_voxel.npy`, `test_energies.npy`.

## calo\_cvae\_hurdle\_model.py

**목적**: Hurdle CVAE의 **구조만** 정의한다. 저장된 가중치(state\_dict)에는 구조가 없어서, 학습·z 모으기·생성 스크립트가 모두 이 파일을 import해 같은 구조를 만든다. 학습 코드는 넣지 않는다. import는 파일을 끝까지 실행하기 때문이다.

### 상수

| 이름 | 값 | 뜻 |
| --- | --- | --- |
| `X_DIM` | 6480 | voxel 수 (입력이자 출력) |
| `C_DIM` | 1 | condition c의 차원 |
| `Z_DIM` | 64 | latent z의 차원 |

### 클래스 `HurdleCVAE(nn.Module)`

`nn.Module`을 상속해서 `model(x, c)` 호출(내부에서 `forward`를 부름), `.parameters()`, `.state_dict()`, `.to(device)`, `.eval()`을 물려받는다. `super().__init__()`은 그 기능들을 준비하는 부모 초기화다.

**속성 (레이어)**

| 속성 | 구조 | 출력 | 역할 |
| --- | --- | --- | --- |
| `self.encoder` | Linear(6481→1024) → ReLU → Linear(1024→512) → ReLU → Linear(512→256) → ReLU | h (batch, 256) | 샤워와 c를 보고 중간 특징 h를 만듦. latent는 아님 |
| `self.fc_mu` | Linear(256→64) | μ\_z | z 분포의 평균. 이름은 관례일 뿐, 의미는 출력을 쓰는 방식에서 생김 |
| `self.fc_log_var` | Linear(256→64) | log σ\_z² | z 분포의 로그분산. log를 뱉게 해서 exp로 항상 양수 보장 |
| `self.decoder_body` | Linear(65→256) → ReLU → Linear(256→512) → ReLU → Linear(512→1024) → ReLU | (batch, 1024) | 세 머리가 공유하는 몸통 |
| `self.head_hit` | Linear(1024→6480), activation 없음 | hit logit | voxel별 hit 확률의 logit. ReLU를 붙이면 p ≥ 0.5가 돼서 안 됨 |
| `self.head_val` | Linear(1024→6480) → Softplus | μ\_x | hit이면 그 크기(log 공간)의 평균. 항상 양수 |
| `self.head_log_var` | Linear(1024→6480), activation 없음 | log σ\_x² | hit 크기가 샤워마다 흔들리는 정도 |

**method**

```python
def reparametrize(self, mu, log_var):
    sigma = torch.exp(0.5 * log_var)       # log σ² → σ
    eps = torch.randn_like(sigma)          # sigma와 같은 shape, 같은 장치의 N(0,1) 난수
    return mu + sigma * eps                # z 샘플. 난수를 밖으로 빼서 gradient가 mu, sigma로 흐름
```

```python
def decode(self, z, c):
    h = self.decoder_body(torch.cat([z, c], dim=1))     # (batch, 64) + (batch, 1) → (batch, 65)
    return self.head_hit(h), self.head_val(h), self.head_log_var(h)
```

`decode`를 따로 둔 이유는 생성에서 encoder 없이 decoder만 쓰기 위해서다. decoder가 몸통과 머리 세 개로 나뉘어 한 번에 부를 방법이 필요했다.

```python
def forward(self, x, c):
    h = self.encoder(torch.cat([x, c], dim=1))          # Sequential은 입력 하나만 받으므로 붙여서
    mu, log_var = self.fc_mu(h), self.fc_log_var(h)     # encoder 쪽 분포
    z = self.reparametrize(mu, log_var)
    hit_logit, val, log_var_x = self.decode(z, c)       # decoder 쪽 분포
    return hit_logit, val, log_var_x, mu, log_var       # loss가 다섯 개 모두 필요
```

파라미터는 약 2,100만 개이고, 대부분은 6481→1024 층과 1024→6480 머리 세 개에 있다. 구조를 바꾼 뒤에는 랜덤 테서로 shape만 먼저 확인하는 게 좋다(`if __name__ == '__main__':` 안에 두면 import할 때는 실행되지 않는다).

## calo\_cvae\_hurdle\_train.py

**목적**: train 샤워 10만 개로 HurdleCVAE를 학습하고, 가중치와 정규화 눈금을 저장한다. 2.6 GB 데이터를 memmap으로 필요한 줄만 읽으며 학습한다.

### 클래스 `VoxelDataset(Dataset)`

DataLoader가 "i번째 데이터"를 요청하면 디스크에서 그 줄만 읽어주는 데이터셋이다. `torch.tensor(전체)`를 하면 2.6 GB가 통째로 복사돼서 memmap을 쓴 의미가 없어지므로 직접 만들었다.

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `self.X` | memmap (100000, 6480) | 디스크에 연결만 한 샤워 파일 (`mmap_mode='r'`) |
| `self.C` | 텐서 (100000, 1) | condition c. 작아서 메모리에 통째로 |
| `__len__` | method | 데이터 개수. DataLoader가 batch 수를 계산할 때 씀 |
| `__getitem__(i)` | method | i번째 샤워와 c를 텐서로. `np.array(...)`로 감싼 건 memmap 값이 읽기 전용이라서 |

`dataset[0]`처럼 대괄호로 부르면 파이썬이 `__getitem__(0)`을 부른다.

### Condition

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `train_E` | ndarray (100000, 1) | 입사 에너지 원본, MeV |
| `log_E` | ndarray (100000, 1) | ln E |
| `log_E_min`, `log_E_max` | numpy 숫자 | min-max 눈금. `norm_params_voxel.pt`에 저장해 생성할 때도 같은 눈금을 씀 |
| `c_train` | ndarray (100000, 1) | (ln E − min) / (max − min), 0 \~ 1 |
| `dataset`, `loader` | Dataset, DataLoader | batch 512, 매 epoch 섞음(`shuffle=True`) |

### Loss 함수 `hurdle_loss`

```python
def hurdle_loss(hit_logit, val, log_var_x, x, mu, log_var):
    log_var_x = log_var_x.clamp(-7, 3)       # σ_x가 0이나 무한대로 도망가지 않게
    hit = (x > 0).float()                    # 정답 라벨: 실제 데이터에서
    bce = F.binary_cross_entropy_with_logits(hit_logit, hit, reduction='sum')
    nll = 0.5 * (hit * (((x - val)**2) / (log_var_x.exp()) + log_var_x)).sum()
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + nll + BETA * kl, bce, nll, kl
```

| 항 | 무엇을 맞추나 | 포인트 |
| --- | --- | --- |
| `bce` | voxel 6480개 모두의 hit 여부 | logit을 받는 함수라 sigmoid가 안에 있고 log(0)을 피한다 |
| `nll` | hit인 voxel의 크기 | `hit *`로 hit 없는 voxel을 0으로 지우고 더한다(mask). 음수 가능 |
| `kl` | encoder 분포를 N(0, I) 쪽으로 | Gaussian끼리라 공식으로 바로 계산 |
| `BETA` | KL 가중치 | 1.0 (b01 branch에서는 0.1) |

모두 `reduction='sum'`(합계)이라 VAE 원래 식(ELBO)에 맞는다. `'mean'`으로 바꾸면 β를 6480배 키운 것과 같아진다.

### 학습 루프

```python
model = HurdleCVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # 모델을 옮긴 다음에 만든다
for epoch in range(EPOCHS):
    for x, c in loader:
        x, c = x.to(device), c.to(device)
        hit_logit, val, log_var_x, mu, log_var = model(x, c)
        loss, bce, nll, kl = hurdle_loss(hit_logit, val, log_var_x, x, mu, log_var)
        optimizer.zero_grad()     # .grad는 누적되므로 비우기
        loss.backward()           # 계산 그래프를 따라 gradient 계산 후 그래프는 버려짐
        optimizer.step()          # weight 업데이트
```

- `tot, tot_b, tot_n, tot_k`: batch마다 `.item()`(테서 숫자 하나 → 파이썬 숫자)을 더해 모은다.
- `n = len(loader.dataset)` = 100000으로 나눠 **이벤트 하나당** loss를 출력한다. 항별로 따로 찍어야 어느 쪽이 문제인지 보인다.
- 끝에 `torch.save(model.state_dict(), 'calo_cvae_hurdle.pt')`.

**예상할 숫자**: BCE 시작값 ≈ 6480 × ln 2 ≈ 4,490(모든 p ≈ 0.5). 학습 후 약 1,630. 여기서 안 내려가면 head\_hit 끝에 activation이 붙었는지 의심한다.

## calo\_collect\_z.py

**목적**: 학습이 끝난 encoder에 train 샤워 10만 개를 통과시켜, decoder가 학습 중에 실제로 받았던 z들을 모은다. NF prior가 배울 데이터다. decoder는 쓰지 않는다.

### 객체와 변수

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `model` | HurdleCVAE | 학습된 가중치를 채운 모델. `load_state_dict(torch.load(..., map_location='cpu'))`로 CPU에 불러온 뒤 `.to(device)` |
| `model.eval()` | method | 평가 모드 (Dropout 같은 층이 있을 때 동작이 바뀜, 습관적으로 붙임) |
| `norm` | 딕셔너리 | `log_E_min`, `log_E_max`. numpy 숫자라 `weights_only=False` 필요 |
| `X_train` | memmap (100000, 6480) | train 샤워 (log 공간) |
| `c_train`, `C_train` | ndarray / 텐서 (100000, 1) | 저장된 눈금으로 만든 c. 눈금은 한 곳(norm)에서만 정한다 |
| `BATCH` | int, 2000 | chunk 크기 |
| `xb`, `Cb` | 텐서 (2000, 6480), (2000, 1) | 이번 chunk의 샤워와 c |
| `h` | 텐서 (2000, 256) | encoder 중간 특징 |
| `mu_z`, `lv_z` | 텐서 (2000, 64) | encoder 분포의 평균과 로그분산 |
| `z` | 텐서 (2000, 64) | `reparametrize`로 **샘플링한** z |
| `z_list` → `Z` | list → ndarray (100000, 64) | chunk 결과를 모아 `np.concatenate`로 합침 |

### 흐름

```python
with torch.no_grad():                          # 학습 안 하므로 gradient 계산 끔
    for start in range(0, N, BATCH):
        xb = torch.from_numpy(np.array(X_train[start:end])).to(device)
        Cb = C_train[start:end]
        h = model.encoder(torch.cat([xb, Cb], dim=1))
        mu_z, lv_z = model.fc_mu(h), model.fc_log_var(h)
        z = model.reparametrize(mu_z, lv_z)
        z_list.append(z.cpu().numpy())         # MPS → CPU → numpy
```

- forward의 앞부분 세 줄만 직접 부른 것이다.
- μ\_z가 아니라 샘플링한 z를 모으는 이유: decoder가 받은 건 샘플이었고, μ\_z만 모으면 σ\_z만큼의 퍼짐이 빠져 분포가 좁아진다.
- **train** 데이터로 하는 이유: decoder가 학습 때 본 z 분포를 배워야 하고, test는 마지막 평가용으로 아껴야 한다.

### 출력

`train_z.npy` (100000, 64), `train_c.npy` (100000, 1). 마지막에 차원별 평균과 표준편차의 최소/최대를 찍는다(`Z.mean(0)`, `Z.std(0)`: 이벤트 축을 없애고 차원 64개 각각). N(0, I)에서 얼마나 벗어났는지 보는 진단이다.

## z\_prior\_model.py와 calo\_z\_prior\_nf.py

**목적**: encoder가 만든 z의 실제 분포 p(z | c)를 conditional NF로 배운다. 생성할 때 N(0, I) 대신 이 NF에서 z를 뽑아 prior hole을 피한다. `z_prior_model.py`는 NF **구조**(함수), `calo_z_prior_nf.py`는 **학습**이다.

현재 `calo_z_prior_nf.py`는 `build_flow`를 import하지 않고 같은 구조를 안에서 직접 만든다. 구조가 같아서 저장된 가중치는 그대로 호환되지만, NF를 강화할 때는 학습 스크립트도 `from z_prior_model import build_flow`를 쓰도록 바꿔야 두 복사본이 어긋나지 않는다.

### z\_prior\_model.py

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `ContextNet(nn.Module)` | 클래스 | coupling layer의 subnet. nflows가 `net(x, context)`로 부르면 두 것을 이어 붙여 안의 Sequential에 넘김 |
| `self.net` | Sequential | (32 + 1) → 256 → 256 → 64. 출력 64개 = s 32개 + t 32개 |
| `build_flow(DIM, C_DIM, n_layers)` | 함수 | NF 전체를 조립해 돌려줌 (기본 64, 1, 8) |
| `make_net(in_f, out_f)` | `build_flow` 안의 함수 | nflows가 layer마다 subnet을 만들 때 부르는 함수. 바깥 함수의 `C_DIM`을 그대로 쓸 수 있게 안에 정의 |
| `mask_A`, `mask_B` | 텐서 (64,) | \[1×32, 0×32\]와 그 반대. 1 = 통과, 0 = 변환 대상. `.byte()`는 nflows가 uint8을 요구해서 |
| `layers` | list | `AffineCouplingTransform` 8개, mask를 번갈아 (`i % 2 == 0`) |
| 반환값 | `flows.Flow` | `CompositeTransform(layers)`(변환) + `StandardNormal([64])`(base), `.float()`로 buffer까지 float32 |

`transform_net_create_fn=make_net`은 함수를 **실행하지 않고** 넘겨주는 것이다. nflows가 필요할 때 `make_net(32, 64)`처럼 부른다.

### calo\_z\_prior\_nf.py

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `Z`, `C` | 텐서 (100000, 64), (100000, 1) | `train_z.npy`, `train_c.npy`. 25 MB라 통째로 MPS에 |
| `loader` | DataLoader(TensorDataset) | 메모리에 있는 텐서는 TensorDataset으로 충분 |
| `base_nll` | float | 기준선: z가 N(0, I)를 따른다고 했을 때의 NLL = ½Σz² + 32 log(2π)의 평균 |
| `flow` | flows.Flow | 학습할 NF |
| `loss` | 텐서 (숫자 하나) | `-flow.log_prob(zb, context=cb).mean()` |
| `tot`, `nll` | float | `loss.item() * len(zb)`로 합을 내고 전체 개수로 나눠 정확한 평균 (마지막 batch가 작아서) |
| `base_nll - nll` | float | gain: N(0, I)보다 몇 nat 더 실제 z에 가까워졌나 = 메운 prior hole |

`flow.log_prob(zb, context=cb)`는 이벤트마다 log p(z | c)를 돌려준다 (shape (512,)). 내부에서 z를 coupling layer에 거꾸로 통과시키며 각 subnet에 context를 넘기고, base의 log 밀도와 log |det J|를 더한다. 끝에 `z_prior_nf.pt` 저장.

### 생성에서 쓰는 법

```python
flow = build_flow()
flow.load_state_dict(torch.load('z_prior_nf.pt', map_location='cpu'))
flow.to(device); flow.eval()
z = flow.sample(1, context=Cb).squeeze(1)       # (n, 1, 64) → (n, 64)
```

`sample`의 첫 인자는 차원이 아니라 **조건 하나당 개수**다. 차원은 base 분포가 이미 알고 있다. temperature 실험에서는 이 줄 대신 `u = T * torch.randn(n, 64)` 후 `z, _ = flow._transform.inverse(u, context=Cb)`를 썼다 (nflows의 정방향은 데이터 → base라 생성은 inverse).

## calo\_cvae\_hurdle\_nf\_generate.py

**목적**: test 에너지 2만 개 조건으로 샤워를 만들어 실제와 비교하는 플롯 6종을 그린다. `MODE`로 z를 어디서 얻을지만 바꿔 같은 decoder를 시험한다.

### 스위치와 상수

| 이름 | 값 | 역할 |
| --- | --- | --- |
| `MODE` | 'generate' / 'reconstruct' / 'nf\_prior' | z 출처: N(0, I) / test 샤워를 encoder에 / NF prior |
| `N` | 20000 | 비교에 쓸 이벤트 수 (플롯용) |
| `BATCH` | 2000 | chunk 크기. 2만 개를 한 번에 하면 M1 GPU 메모리 부족 |
| `MAX_TRIES` | 10 | threshold 아래로 떨어진 hit을 다시 뽑는 최대 횟수 |
| `Y_MAX` | log(1 + 10⁴) ≈ 9.21 | 비율 1에 해당하는 log 값. voxel 하나가 E\_inc를 넘지 못하게 |
| `y_thr`, `Y_thr` | (N, 1) | 이벤트별 threshold를 log 공간으로: log(1 + 0.01515 / E × 10⁴) |

### chunk 반복문 안의 변수

| 이름 | shape | 뜻 |
| --- | --- | --- |
| `n`, `Cb`, `Yb` | (n, 1) | 이번 chunk의 크기, condition, threshold |
| `z` | (n, 64) | MODE에 따라 얻은 latent |
| `hit_logit`, `val`, `log_var_x` | (n, 6480) | decoder 세 머리의 출력 |
| `p` | (n, 6480) | `torch.sigmoid(hit_logit)`, hit 확률 |
| `mask` | (n, 6480), 0/1 | `torch.bernoulli(p)`, voxel마다 독립적으로 던진 동전 |
| `sigma` | (n, 6480) | `exp(0.5 · log_var_x)`, voxel별 σ\_x |
| `y` | (n, 6480) | μ\_x + σ\_x · ε, log 공간 크기 |
| `bad` | (n, 6480), bool | `(mask > 0) & (y < Yb)`: 동전은 hit인데 크기가 threshold 아래 |
| `y_list`, `z_list` | list | chunk 결과를 CPU로 옮겨 모음 |
| `sig_sum`, `hit_sum` | float | hit voxel의 σ 합과 개수 → 평균 σ, 이벤트당 hit 수 |

### 생성 핵심

```python
p = torch.sigmoid(hit_logit)
mask = torch.bernoulli(p)                         # 동전
sigma = torch.exp(0.5 * log_var_x)
y = val + sigma * torch.randn_like(val)           # 크기 샘플
for _ in range(MAX_TRIES):                        # truncated Gaussian
    bad = (mask > 0) & (y < Yb)                   # 비교마다 괄호 (&가 먼저 계산됨)
    if not bad.any(): break
    y = torch.where(bad, val + sigma * torch.randn_like(val), y)   # 매번 새 난수
y = torch.maximum(y, Yb * 1.001)                  # 끝까지 못 넘은 것은 threshold 바로 위로
y = torch.clamp(y, max=Y_MAX)                     # 물리적 상한
y_list.append((y * mask).cpu().numpy())           # mask는 곱해서 shape 유지
```

- `torch.where(조건, a, b)`: 원소마다 조건이 True면 a, 아니면 b. 다시 뽑을 voxel만 갈아끼운다.
- 동전이 정한 hit을 지키려는 것이 다시 뽑기의 목적이다. 출력의 `hits from coin`과 `hits after cuts`가 같아야 한다.
- reconstruct 분기는 `X_test[start:end]`를 encoder에 넣고 `reparametrize`한다. `test_voxel.npy`와 `test_energies.npy`의 i번째 줄은 같은 이벤트다.

### 비교와 플롯

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `log_to_fraction(y, E)` | 함수 | (eʸ − 1)/10⁴로 비율로 되돌리고 threshold/E 미만을 0으로. ÷ E는 되돌리지 않음 |
| `real` | (N, 6480) | hdf5 원본에 threshold 후 ÷ E. PCA나 log를 거치지 않은 기준 |
| `gen` | (N, 6480) | 생성 결과를 같은 비율 공간으로 |
| `sets` | dict | `{'Real': real, MODE: gen}`. 플롯 루프가 이걸 돌며 같은 축에 겹쳐 그린다 |
| `fig, ax` | Figure, (2, 3) Axes 배열 | `ax[행, 열]`로 패널 선택, `ax.flat`으로 전부 순회 |
| `edges`, `idx`, `centers` | ndarray | 에너지를 log 간격 10구간으로. `np.digitize(...) - 1`을 `np.clip(..., 0, 9)`로 양 끝 보정, 중심은 기하평균 |
| `bins` | `np.logspace(-8, 0, 81)` | 스펙트럼용 log 간격 경계 80칸 |

패널별 계산 (`v`는 (N, 6480), 괄호 안 숫자 = 없어지는 축):

| 패널 | 계산 | 결과 |
| --- | --- | --- |
| Total energy fraction | `v.sum(1)` | 이벤트별 총 비율 히스토그램 |
| Mean voxel profile | `v.mean(0)` | voxel별 평균 선 |
| Sparsity | `(v == 0).mean(1)` | 이벤트별 0인 voxel 비율 |
| Response vs E\_inc | 구간별 `tot[idx == k].mean()`, `.std()` | 평균 ± 표준편차(분해능) |
| Event-to-event std | `v.std(0)` | voxel별 이벤트 간 표준편차 |
| Voxel energy spectrum | `v[v > 0]` | 0이 아닌 값들을 1차원으로 모아 개수 히스토그램 (density 없이, x·y log) |

축 라벨의 `r'$E_\mathrm{voxel}/E_\mathrm{inc}$'`는 raw string + mathtext다.

## calo\_cvae\_hurdle\_generate.py (NFCVAE/classifier, 생성 저장용)

**목적**: classifier test에 쓸 가짜 샤워를 만든다. test 에너지 **10만 개 전부**를 조건으로 NF prior 생성을 하고, 플롯 대신 log 공간 그대로 `gen_voxel.npy`에 저장한다. (`hurdle/` 폴더의 같은 이름 파일은 이전 단계의 플롯용 생성 스크립트다.)

### 앞 스크립트와 다른 점

|  | nf\_generate (플롯용) | 이 스크립트 (저장용) |
| --- | --- | --- |
| 이벤트 수 | `N = 20000` | `N = len(test_E)` = 100000 |
| z 출처 | MODE로 선택 | NF prior만 |
| 결과 보관 | `y_list`에 모아 합침 (메모리) | memmap `G`에 바로 기록 (디스크) |
| 뒤 처리 | 비율로 되돌려 플롯 | log 공간 그대로 저장 (`test_voxel.npy`와 같은 공간) |

10만 × 6480 × 4 bytes ≈ 2.6 GB라 리스트에 쌓으면 메모리가 위험해서 memmap을 쓴다.

### 핵심 줄

```python
G = open_memmap('gen_voxel.npy', mode='w+', dtype=np.float32, shape=(N, 6480))
...
        z = flow.sample(1, context=Cb).squeeze(1)
        ...                                         # 동전, σ 샘플링, 다시 뽑기, clamp (앞과 동일)
        G[start:end] = (y * mask).cpu().numpy()     # 이 chunk의 줄들에 기록
G.flush()
```

- `G[start:end]`: start부터 end 직전까지 2,000줄. `G[start]`는 한 줄이라 shape이 안 맞는다.
- i번째 줄은 `test_energies.npy`의 i번째 에너지 조건으로 만들어진다. classifier가 이 짝을 그대로 쓴다.
- `torch.maximum(y, Yb * 1.001)`가 만드는 "threshold의 정확히 1.001배" 값은 실제 데이터에 없는 흔적이다. 드물지만 classifier가 단서로 쓸 수는 있다.

## calo\_classifier.py

**목적**: 실제 샤워와 생성 샤워를 섞어 DNN에게 진짜/가짜를 맞히게 하고, 학습에 안 쓴 샤워로 AUC를 잔다. AUC가 0.5에 가까울수록 생성 모델이 좋다. `NULL_TEST = True`면 가짜 자리에 다른 실제 샤워를 넣어 채점 과정 자체를 검사한다.

### 데이터 분할

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `c` | ndarray (100000, 1) | test 에너지의 condition (저장된 눈금) |
| `rng` | `np.random.default_rng(seed=0)` | 시드를 고정한 난수 생성기. 매번 같은 분할 |
| `perm` | ndarray (100000,) | 0 \~ 99999를 섞은 이벤트 번호 |
| `idx_train`, `idx_val`, `idx_test` | ndarray | 60% / 20% / 20%. 슬라이스에 콜론이 있어야 구간(없으면 숫자 하나) |

train으로 학습, validation으로 가장 좋은 epoch 선택, test로 마지막에 한 번 채점한다. 이벤트 번호로 나눠서 실제 j와 생성 j(같은 에너지 조건)는 항상 같은 구간에 들어간다.

### 클래스 `RealGenDataset(Dataset)`

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `self.R`, `self.G` | memmap | 실제(`test_voxel.npy`), 생성(`gen_voxel.npy`) |
| `self.C` | 텐서 (100000, 1) | condition |
| `self.idx` | ndarray | 이 데이터셋이 쓸 이벤트 번호들 |
| `self.partner` | ndarray | `np.roll(idx, 1)`: 한 칸씩 밀린 다른 이벤트 (null test용) |
| `__len__` | method | `2 * len(idx)`: 이벤트당 실제 1 + 생성 1 |
| `__getitem__(k)` | method | `j = idx[k // 2]`. 짝수 k → 실제(label 1), 홀수 k → 생성(label 0). 입력은 `torch.cat([x, cj])` (6481,), 라벨은 `torch.tensor([label])` (1,) |

`//`는 몫, `%`는 나머지다. 진짜와 가짜가 정확히 반반이어야 "많은 쪽으로 찍기"로 점수가 오르지 않는다. `make_loader(idx, shuffle)`는 이 데이터셋으로 DataLoader를 만드는 도우미 함수다 (test는 순서를 유지하려고 `shuffle=False`).

### 분류기와 학습

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `clf` | Sequential | 6481 → 512 → 512 → 512 → 1 (logit). 맨 처음 배운 DNN |
| `run_epoch(loader, train)` | 함수 | 한 epoch를 돌며 평균 BCE를 돌려줌. `train=True`면 학습도 함 |
| `clf.train()` / `clf.eval()` | method | 모드 전환. 한 줄 if-else로 |
| `torch.set_grad_enabled(train)` | context | train일 때만 gradient 계산 |
| `best_val`, `best_state` | float, dict | validation BCE가 가장 낮았던 순간의 값과 가중치 |
| `copy.deepcopy(clf.state_dict())` | dict | 그 순간의 가중치를 **별도로 복제**. 그냥 담으면 학습이 계속되며 같이 바뀐다 |

분류기는 epoch 2\~3부터 train 샤워를 외우기 시작한다(train BCE는 계속 내려가고 validation은 올라감). 그래서 EPOCHS = 5와 early stopping을 쓰고, 학습 직후 `clf_best.pt`로 저장해 뒤에서 에러가 나도 남게 했다.

### 채점

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `probs` | ndarray (40000,) | test 구간 샤워마다 "진짜일 확률" (sigmoid 후). `.ravel()`로 1차원 |
| `labels` | ndarray (40000,) | 정답 1/0 |
| `roc_auc_score(labels, probs)` | sklearn 함수 | 진짜·가짜 짝 중 진짜 점수가 더 높은 비율. 기준선 없이 순서만 봄 |
| `(probs > 0.5) == labels` | bool 배열 | 0.5 기준 정확도 (`.mean()`) |
| `E_items` | ndarray (40000,) | `np.repeat(test_E[idx_test, 0], 2)`: 샘플 순서(실제, 생성, 실제, ...)에 맞춘 에너지 |
| `edges`, `bin_id`, `sel` | ndarray | 에너지 10구간과 구간별 boolean 선택. 구간마다 AUC를 따로 계산 |

에너지에 `[:, 0]`을 붙이는 건 (n, 1)을 (n,)으로 바꿔 다른 1차원 배열과 섞을 때 broadcasting 사고를 막기 위해서다.

## feature\_auc.py

**목적**: classifier가 무엇을 보고 구분하는지 짐작하려고, 물리량 하나를 classifier 점수 대신 넣어 AUC를 잰다. AUC는 점수의 순서만 보므로 아무 숫자나 점수로 쓸 수 있다. 값이 크면 모델이 그 물리량을 **틀리게** 만든 것이다. 분류기 학습은 필요 없다.

### 객체와 변수

| 이름 | 종류 | 역할 |
| --- | --- | --- |
| `R`, `G` | memmap (100000, 6480) | 실제와 생성 (log 공간) |
| `E` | ndarray (100000,) | test 에너지. `[:, 0]`으로 1차원 |
| `features(Y)` | 함수 | (n, 6480) → 이벤트마다 물리량 4개를 담은 딕셔너리 |
| `f` | ndarray | `np.expm1(Y)/1e4`: 비율 공간으로 되돌린 값 |
| `sel` | ndarray | `np.where((E >= lo) & (E < hi))[0][:5000]`: 그 에너지 구간 이벤트 번호 최대 5,000개 |
| `fr`, `fg` | dict | 실제와 생성의 물리량 |
| `scores` | ndarray (10000,) | 실제 5,000개 뒤에 생성 5,000개를 이어 붙인 값 |
| `labels` | ndarray (10000,) | `np.ones` + `np.zeros`: 같은 순서의 정답 꼬리표 |

### 물리량

| 키 | 계산 | 뜻 |
| --- | --- | --- |
| `total energy fraction` | `f.sum(1)` | 입사 에너지 중 기록된 비율 |
| `number of hits` | `(Y > 0).sum(1)` | 0이 아닌 voxel 수 |
| `mean log-value of hits` | `Y.sum(1) / np.maximum((Y > 0).sum(1), 1)` | hit 하나의 평균 크기. maximum은 0으로 나누기 방지 |
| `max voxel fraction` | `f.max(1)` | 가장 뜨거운 voxel의 비율 |

### 채점

```python
auc = roc_auc_score(labels, scores)
print(max(auc, 1 - auc), np.median(fr[name]), np.median(fg[name]))
```

- `max(auc, 1 - auc)`: 생성이 더 크든 작든 "얼마나 갈리나"만 보려고 방향을 지운다. 방향은 중앙값 두 개로 확인한다.
- 한계: 한쪽이 대체로 큰지를 재는 방식이라, 평균은 같고 퍼짐만 다른 차이는 0.5 근처로 나오기 쉽다. 그럴 땐 히스토그램을 같이 본다.

## 공통 문법과 도구

여러 스크립트에 반복해서 나오는 문법과 도구를 한 곳에 모았다.

### numpy와 torch

| 하려는 일 | numpy | torch |
| --- | --- | --- |
| 파일에서 읽기 | `np.load(path)`, `np.load(path, mmap_mode='r')` | `torch.load(path, map_location='cpu')` |
| 서로 바꾸기 | `torch.tensor(a)` (복사), `torch.from_numpy(a)` (공유) | `t.cpu().numpy()` |
| 자료형 | `a.astype(np.float32)` | `t.float()`, 모듈은 `m.float()` |
| 축 지정 | `a.sum(axis=1)` | `t.sum(dim=1)` |
| 숫자 하나 꺼내기 | 줄이면 바로 numpy 숫자 | `t.item()` (원소가 하나일 때만) |

`.sum()`, `.mean()`, `.std()`, `.max()`는 각 라이브러리의 배열/테서 클래스가 가진 method다. 괄호 안이 비면 모든 축을 없애 숫자 하나를 만든다.

### axis: 괄호 안 숫자 = 없어지는 축

샤워 배열은 (이벤트, voxel)이다. `sum(1)`은 voxel 축을 없애 이벤트별 값 (N,)을, `mean(0)`은 이벤트 축을 없애 voxel별 값 (6480,)을 만든다. `keepdim=True`(numpy는 `keepdims`)는 없어진 축을 크기 1로 남겨 (1, 64)처럼 만든다.

### 고르기와 지우기

| 쓰임 | 코드 | 결과 |
| --- | --- | --- |
| 조건에 맞는 값을 바꾸기 | `a[a < thr] = 0` | shape 유지, 제자리 수정 |
| 조건에 맞는 값만 모으기 | `a[a > 0]` | 1차원으로 펼쳐짐 (스펙트럼) |
| 일부를 0으로 지우되 자리는 유지 | `a * mask` | shape 유지 (loss의 hit, 생성의 mask) |
| 원소마다 둘 중 하나 | `torch.where(cond, a, b)` | shape 유지 |
| 두 조건 동시에 | `(a > 0) & (b < c)` | 비교마다 괄호. 테서에는 `and` 대신 `&` |
| True인 위치의 번호 | `np.where(cond)[0]` | 인덱스 배열 |

텐서에는 for문 대신 이런 연산을 통째로 건다 (vectorization). 테서를 for문으로 돌면 원소가 아니라 행이 나오고 느리다.

### Broadcasting

(10000, 6480) ÷ (10000, 1)은 각 행을 자기 값으로 나눈다. 하지만 (n, 1)과 (n,)을 더하면 (n, n)이 된다. 모델 입력(c)은 샤워 옆에 붙이려고 (n, 1)로, 분석용 양(에너지, 총합, labels)은 (n,)으로 맞춘다.

### 슬라이싱과 호출

- `a[i]`는 원소(행) 하나, `a[i:j]`는 구간, `a[i:]`는 끝까지, `a[:, 0]`은 모든 행의 0번째 열
- `()`는 호출, `[]`는 인덱싱. `x.numpy`는 method 자체, `x.numpy()`가 실행 결과
- 함수 안에서 쓰는 이름은 모두 인자로 받는다. 안 받으면 파이썬이 바깥(전역) 변수를 집어 에러 없이 틀린 값으로 돈다

### Dataset과 DataLoader

`Dataset`을 상속하면 `__len__`(개수)과 `__getitem__(i)`(i번째 하나)만 채우면 된다. `DataLoader(ds, batch_size=512, shuffle=True)`는 섞인 번호 512개로 `__getitem__`을 부르고 결과를 쌓아 batch를 만든다. 메모리에 이미 있는 텐서는 `TensorDataset`으로 충분하다.

### 장치 (MPS)

- 모델 `model.to(device)`, 입력 `x.to(device)`, 난수 `torch.randn(..., device=device)` 또는 `torch.randn_like(기존 텐서)`
- optimizer는 모델을 옮긴 다음에 만든다
- MPS는 float64를 못 다룬다. numpy float64 숫자 대신 파이썬 숫자를 쓰고, nflows Flow는 `.float()`
- 메모리가 부족하면 에러 없이 0을 돌려줄 수 있다. chunk로 나누고 결과를 숫자로 확인한다. 이상한 에러는 `device = 'cpu'`로 돌려보면 원인 줄에서 멈춘다

### 저장 형식

| 형식 | 담는 것 | 읽기 |
| --- | --- | --- |
| `.npy` | 배열 하나 | `np.load`, memmap 가능 |
| `.npz` | 이름 붙은 배열 여러 개 (`np.savez`) | `d = np.load(...)`, `d['mean']` |
| `.pt` | 아무 파이썬 객체 (state\_dict, 딕셔너리) | `torch.load`. 텐서 외의 것이 있으면 `weights_only=False` |
| `.hdf5` | 이름 붙은 배열 여러 개 | `h5py.File`, 대괄호로 범위를 줄 때만 읽음 |

### 기타

- `if __name__ == '__main__':` 안의 코드는 직접 실행할 때만 돈다. 모델 파일의 shape 테스트에 쓴다
- `assert 조건, 메시지`는 조건이 거짓이면 멈춘다. 조용한 실패를 막는 안전장치
- `f"{x:.3f}"`, `f"{name:5s}"`는 f-string 포맷. `r'...'`는 `\`를 글자로 취급하는 raw string
- 설치는 `python -m pip install ...`로, 실행하는 파이썬과 같은 곳에
