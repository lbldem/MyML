# twostage/s3_flow1_generate.py - using Flow 1 for layers, generate layer energies for test E_inc and evaluate.

import numpy as np
import torch
from common import HERE, DATA, device
from s2_flow1_model_v3 import build_flow1_v3

torch.manual_seed(1)
norm = torch.load(HERE / 'flow1_norm.pt', weights_only=False)
THRESHOLD = norm['THRESHOLD']
LOG_THR = np.log10(THRESHOLD)

flow = build_flow1_v3()
flow.load_state_dict(torch.load(HERE / 'flow1_v3.pt', map_location='cpu'))
flow.to(device).eval()

E_inc = np.load(DATA / 'test_energies.npy')
logE = np.log(E_inc)
El_real = np.load(HERE / 'test_Elayer.npy')                         # These have never been preprocessed!
c = (logE - norm['logE_min']) / (norm['logE_max'] - norm['logE_min'])
print("c range: %.3f ~ %.3f" % (c.min(), c.max()))                  # 0.000 ~ 1.000 근처여야 함
C = torch.tensor(c.astype(np.float32)).to(device)
xm = torch.tensor(norm['x_mean']).to(device)
xs = torch.tensor(norm['x_std']).to(device)


# Sampling from our model ─────────────────────────────────────────────────────────────────────────────────────
out, out_u = [], []
with torch.no_grad():
    for s in range(0, len(C), 2000):
        x = flow.sample(1, context=C[s:s+2000]).squeeze(1)            # Before squeezing axis 1, x ~ (100000, 1, 45), we're sampling from model.
        out_u.append(x.cpu().numpy())
        x = x * xs + xm
        out.append((10 ** x).cpu().numpy())                           # Undoing log10 preprocess done in training.
El_gen = np.concatenate(out)
El_gen[El_gen < THRESHOLD] = 0.0
u_gen = np.concatenate(out_u)
np.save(HERE / 'gen_Elayer.npy', El_gen.astype(np.float32))

# Evaluation ────────────────────────────────────────────────────────────────────────────────────────────────────
def sep_power(a, b, bins):
    ha, _ = np.histogram(a, bins)
    hb, _ = np.histogram(b, bins)
    ha, hb = ha / max(ha.sum(), 1), hb / max(hb.sum(), 1)
    return 0.5 * ((ha - hb)**2 / (ha + hb + 1e-16)).sum()

sp = [sep_power(El_real[:, l], El_gen[:, l], np.logspace(np.log10(THRESHOLD), np.log10(El_real[:, l].max()), 40))
      for l in range(45)]                                   # separation power of each layer as a list


print("E layer separation power: ")
for l in [0, 5, 10, 20, 30, 40, 44]:
    print(f"    layer{l:2d}: {sp[l]:.5f}")
print(f" mean over layers: {np.mean(sp):.5f}")
print("Etot/Einc separation power: %.5f" %
      sep_power(El_real.sum(1) / E_inc[:, 0], El_gen.sum(1) / E_inc[:, 0], np.linspace(0.5, 1.5, 101)))
print("zero-layer fraction  real %.3f | gen %.3f" % ((El_real == 0).mean(), (El_gen == 0).mean()))

print("")
print("Etot/Einc mean   real %.3f | gen %.3f" %
      ((El_real.sum(1) / E_inc[:, 0]).mean(), (El_gen.sum(1) / E_inc[:, 0]).mean()))
print("layer mean [MeV] real", El_real.mean(0)[[0, 10, 20, 30, 44]].round(1))
print("                 gen ", El_gen.mean(0)[[0, 10, 20, 30, 44]].round(1))

# ---- 진단 1: real test의 NLL (val 42.58과 비교) ----
rng_t = np.random.default_rng(0)
x_real = np.where(El_real > 0, np.log10(np.maximum(El_real, THRESHOLD)), LOG_THR - rng_t.random(El_real.shape)).astype(np.float32)
xr = torch.from_numpy(x_real).to(device)
xr = (xr - xm) / xs                      # 학습 때와 같은 표준화
nll = []
with torch.no_grad():
    for s in range(0, len(xr), 2000):
        nll.append(-flow.log_prob(xr[s:s+2000], context=C[s:s+2000]).cpu())  # 배치별 -log p(x|c)
nll = torch.cat(nll)
print(f"test NLL mean {nll.mean():.3f} | median {nll.median():.3f}")

# ---- 진단 2: 표준화 공간 percentile, real vs gen ----
u_real = xr.cpu().numpy()
for name, u in [('real', u_real), ('gen ', u_gen)]:
    p = np.percentile(u, [0.1, 1, 50, 99, 99.9])
    print(name, np.round(p, 2), "| max(u)", np.abs(u).max().round(1))

# ---- 진단 3: 이벤트 단위 이상치 ----
E_inc_t = E_inc.reshape(-1)     
ratio_gen = El_gen.sum(1) / E_inc_t
ratio_real = El_real.sum(1) / E_inc_t
print("median Etot/Einc real", np.median(ratio_real).round(3), "| gen", np.median(ratio_gen).round(3))
bad = (El_gen > E_inc_t[:, None]).any(axis=1)                    # 어떤 layer라도 E_inc보다 큰 이벤트 (bool, shape (N,))
print("fraction of events with any layer > E_inc:", bad.mean())
zf_r, zf_g = (El_real == 0).mean(0), (El_gen == 0).mean(0)
for l in [0, 5, 10, 20, 30, 40, 44]:
    print(f"layer {l:2d} zero fraction: real {zf_r[l]:.3f} | gen {zf_g[l]:.3f}")
xg = np.log10(np.maximum(El_gen, 1e-30))
for l in [20, 30, 44]:
    r = El_real[:, l]; g = El_gen[:, l]
    near_r = ((r >= THRESHOLD) & (r < THRESHOLD * 2)).mean()
    near_g = ((g >= THRESHOLD) & (g < THRESHOLD * 2)).mean()
    print(f"layer{l:2d} THR ~ 2THR fraction real {near_r:.3f} | gen {near_g:.3f}")


# ---- 진단 4: forward latent의 축별 std ----
lat = []
with torch.no_grad():
    for s in range(0, len(xr), 2000):
        u, _ = flow._transform(xr[s:s+2000], context=C[s:s+2000])   # data → base(the NF result u, we say) 방향
        lat.append(u.cpu().numpy())
lat = np.concatenate(lat)
std_lat = np.sort(lat.std(axis=0))
print("latent std (smallest 10):", std_lat[:10].round(3))
print("latent std (largest 5)  :", std_lat[-5:].round(3))