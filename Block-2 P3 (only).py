"""
FM-PINN — BLOCK 2, P3 ONLY: COVERAGE OF alpha
"""

from __future__ import annotations
import time
import numpy as np
from math import gamma as Gamma
from scipy import stats
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
NP = np.float64

GAMMA, D_COEF = 1.0, 0.02
XMIN, XMAX, T_END, S_IC = -0.6, 0.6, 1.0, 0.07
NX, NT = 101, 100
NX_FINE, NT_FINE = 301, 600

# calibration pooled over these alphas; coverage tested on the held-out alphas
CAL_ALPHAS = (0.60, 0.70, 0.80)
TEST_ALPHAS = (0.65, 0.75)
N_CAL_PER_ALPHA = 20      # calibration datasets/alpha (pooled 60 >> 19 needed)
N_TEST = 40              # test datasets per held-out alpha
ENSEMBLE = 6
N_ITER = 3000
NOISE_PCT = 5.0


# =====================================================================
#  SOLVER + FINE-GRID TRUTH + observation
# =====================================================================
def l1_weights_np(alpha, n):
    if abs(alpha - 1.0) < 1e-12:
        b = np.zeros(n, dtype=NP); b[0] = 1.0; return b
    j = np.arange(n, dtype=np.float64)
    return ((j + 1.0) ** (1.0 - alpha) - j ** (1.0 - alpha)).astype(NP)


def op_noflux(x, gamma, D):
    Nx = len(x); dx = x[1] - x[0]; A = np.zeros((Nx, Nx), dtype=NP)
    for i in range(Nx - 1):
        xh = 0.5 * (x[i] + x[i + 1]); a = -gamma * xh
        cf_i = a * 0.5 + D / dx; cf_ip = a * 0.5 - D / dx
        A[i, i] += -cf_i / dx; A[i, i + 1] += -cf_ip / dx
        A[i + 1, i] += +cf_i / dx; A[i + 1, i + 1] += +cf_ip / dx
    return A


def solve_ref(alpha, gamma, D, x_min, x_max, Nx, T, Nt, s_ic):
    x = np.linspace(x_min, x_max, Nx).astype(NP); dx = x[1] - x[0]; dt = T / Nt
    p0 = np.exp(-(x**2)/(2*s_ic**2)); p0 /= dx * p0.sum()
    sigma = 1.0 / (Gamma(2.0 - alpha) * dt ** alpha)
    A = op_noflux(x, gamma, D); b = l1_weights_np(alpha, Nt)
    P = np.zeros((Nt + 1, Nx), dtype=NP); P[0] = p0
    Minv = np.linalg.inv(sigma * b[0] * np.eye(Nx, dtype=NP) - A)
    for n in range(1, Nt + 1):
        h = np.zeros(Nx, dtype=NP)
        for j in range(1, n):
            h += b[j] * (P[n - j] - P[n - j - 1])
        P[n] = Minv @ (sigma * b[0] * P[n - 1] - sigma * h)
    t = np.linspace(0.0, T, Nt + 1).astype(NP)
    return x, t, P, dx


def solve_truth_fine(alpha, gamma, D, x_min, x_max, T, s_ic,
                     Nx_coarse=NX, Nt_coarse=NT, Nx_fine=NX_FINE, Nt_fine=NT_FINE):
    xf = np.linspace(x_min, x_max, Nx_fine).astype(NP)
    _, _, Pf, _ = solve_ref(alpha, gamma, D, x_min, x_max, Nx_fine, T, Nt_fine, s_ic)
    xc = np.linspace(x_min, x_max, Nx_coarse).astype(NP); dxc = xc[1] - xc[0]
    tc = np.linspace(0.0, T, Nt_coarse + 1).astype(NP)
    xi = np.clip(np.searchsorted(xf, xc), 0, Nx_fine - 1)
    ti = np.clip(np.round(tc * Nt_fine / T).astype(int), 0, Nt_fine)
    Pc = Pf[ti][:, xi].copy()
    Pc = Pc / (dxc * Pc.sum(axis=1, keepdims=True))
    return xc, tc, Pc, dxc


def observe(P, x, dx, N_sample, noise_pct, obs_indices, rng):
    Nt1, Nx = P.shape
    P_obs = P.copy()
    for it in obs_indices:
        pdf = np.clip(P[it], 0, None); pdf = pdf / (dx * pdf.sum())
        cdf = np.cumsum(pdf) * dx; cdf = cdf / cdf[-1]
        u = rng.random(N_sample)
        draws = np.interp(u, cdf, x)
        counts, _ = np.histogram(draws, bins=np.concatenate([x - dx/2, [x[-1] + dx/2]]))
        dens = counts.astype(NP)
        if noise_pct > 0:
            dens = dens * (1.0 + (noise_pct/100.0) * rng.standard_normal(Nx))
            dens = np.clip(dens, 0, None)
        s = dx * dens.sum()
        P_obs[it] = dens / s if s > 0 else pdf
    return P_obs


# =====================================================================
#  RECOVERY ENGINE 
# =====================================================================
class EnergyNet(nn.Module):
    def __init__(self, hidden=48, layers=3):
        super().__init__()
        seq = []; prev = 2
        for _ in range(layers):
            seq += [nn.Linear(prev, hidden), nn.Tanh()]; prev = hidden
        seq.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*seq)
        for mm in self.net.modules():
            if isinstance(mm, nn.Linear):
                nn.init.xavier_uniform_(mm.weight); nn.init.zeros_(mm.bias)

    def energy(self, X, T, shape):
        return self.net(torch.cat([X, T], 1)).view(*shape)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def recover_alpha(P_obs, obs_mask, x, t, dx, n_iter=3000, lr=5e-3, seed=0, device=None):
    if device is None: device = DEVICE
    torch.manual_seed(seed); np.random.seed(seed)
    Nx = len(x); Nt = len(t) - 1
    xg = torch.tensor(x, device=device); tg = torch.tensor(t, device=device)
    Pdata = torch.tensor(P_obs, device=device)
    mask = torch.tensor(obs_mask, device=device).view(-1, 1)
    X = xg.view(1, Nx).expand(Nt + 1, Nx).reshape(-1, 1)
    Tt = tg.view(Nt + 1, 1).expand(Nt + 1, Nx).reshape(-1, 1)
    xin = xg[1:-1].view(1, -1)
    net = EnergyNet().to(device)
    alpha_raw = nn.Parameter(torch.tensor(0.0, device=device))
    def A(): return 0.05 + 0.9 * torch.sigmoid(alpha_raw)
    opt = torch.optim.Adam(list(net.parameters()) + [alpha_raw], lr=lr)
    dtt = 1.0 / Nt
    idx = (torch.arange(Nt, device=device).view(Nt, 1)
           - torch.arange(Nt, device=device).view(1, Nt))
    tri = (idx >= 0)
    for it in range(n_iter):
        f = net.energy(X, Tt, (Nt + 1, Nx))
        logZ = torch.logsumexp(f + np.log(dx), 1, keepdim=True)
        P = torch.exp(f - logZ)
        a = A()
        jj = torch.arange(Nt + 1, dtype=torch.float64, device=device)
        bw = (jj + 1.0) ** (1.0 - a) - jj ** (1.0 - a)
        sigma = 1.0 / (torch.exp(torch.lgamma(2 - a)) * dtt ** a)
        px = (P[:, 2:] - P[:, :-2]) / (2 * dx)
        pxx = (P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]) / dx**2
        Lp = GAMMA * (P[:, 1:-1] + xin * px) + D_COEF * pxx
        dP = P[1:, 1:-1] - P[:-1, 1:-1]
        W = torch.where(tri, bw[idx.clamp(min=0)], torch.zeros_like(bw[0]))
        cap = sigma * (W @ dP)
        l_pde = ((cap - Lp[1:]) ** 2).mean()
        diff = (P - Pdata) * mask
        l_data = (diff ** 2).sum() / (mask.sum().clamp(min=1) * Nx)
        loss = l_pde + 100.0 * l_data
        opt.zero_grad(); loss.backward(); opt.step()
    return float(A().item())


# =====================================================================
#  helper: one dataset - ensemble mean + half-width
# =====================================================================
# Cache the FINE-grid truth once per alpha (identical across datasets; only the
# noise draw differs). This is the key speed fix — previously the 301x600 fine
# solve ran once PER DATASET (hundreds of times).
_TRUTH_CACHE = {}
def _truth(true_alpha):
    if true_alpha not in _TRUTH_CACHE:
        _TRUTH_CACHE[true_alpha] = solve_truth_fine(true_alpha, GAMMA, D_COEF,
                                                    XMIN, XMAX, T_END, S_IC)
    return _TRUTH_CACHE[true_alpha]


def ensemble_interval(true_alpha, dataset_id, ensemble=ENSEMBLE, n_iter=N_ITER):
    x, t, P, dx = _truth(true_alpha)                      # cached fine solve
    mask = np.ones(len(t), bool)
    rng = np.random.default_rng(5000 + int(true_alpha*1000) + dataset_id)
    P_obs = observe(P, x, dx, 1000, NOISE_PCT, np.where(mask)[0], rng)
    ens = np.array([recover_alpha(P_obs, mask, x, t, dx, seed=e, n_iter=n_iter)
                    for e in range(ensemble)])
    return ens.mean(), 1.96 * ens.std()


def conformal_factor(ratios, level=0.95):
    """finite-sample split-conformal quantile: ceil((n+1)*level)/n -th."""
    n = len(ratios); q = min(np.ceil((n + 1) * level) / n, 1.0)
    return float(np.quantile(ratios, q, method="higher"))


def binom_ci(hits, n, conf=0.95):
    a = (1 - conf) / 2
    lo = stats.beta.ppf(a, hits, n - hits + 1) if hits > 0 else 0.0
    hi = stats.beta.ppf(1 - a, hits + 1, n - hits) if hits < n else 1.0
    return float(np.nan_to_num(lo)), float(np.nan_to_num(hi))


# =====================================================================
#  P3 — pooled cross-alpha calibration, held-out coverage, binomial CI
# =====================================================================
def coverage_test_v2():
    print("\n" + "="*64)
    print("  P3 (v2) — calibrated coverage: pooled cross-alpha, held-out test")
    print("="*64)

    # --- calibration: pool residual ratios across CAL_ALPHAS (alpha known only
    #     here, during calibration; the FACTOR is shared and alpha-agnostic) ---
    print(f"  Calibration pooled over alphas {CAL_ALPHAS}, "
          f"{N_CAL_PER_ALPHA} datasets each ...")
    ratios = []
    import time as _t
    for ta in CAL_ALPHAS:
        for d in range(N_CAL_PER_ALPHA):
            _t0 = _t.time()
            mean, hw = ensemble_interval(ta, d)
            ratios.append(abs(mean - ta) / max(hw, 1e-9))
            print(f"    [cal] alpha={ta} dataset {d+1}/{N_CAL_PER_ALPHA}  "
                  f"mean={mean:.4f}  ({_t.time()-_t0:.0f}s)", flush=True)
    ratios = np.array(ratios)
    c = conformal_factor(ratios, 0.95)
    naive_cal = np.mean(ratios <= 1.0)
    print(f"  pooled datasets = {len(ratios)},  naive coverage on calibration = "
          f"{naive_cal*100:.0f}%  ->  conformal factor c = {c:.2f}")

    # --- test coverage on HELD-OUT alphas, with binomial CI ---
    print(f"\n  Held-out test on alphas {TEST_ALPHAS} ({N_TEST} datasets each):")
    summary = {}
    for ta in TEST_ALPHAS:
        naive_hits = 0; calib_hits = 0
        for d in range(N_TEST):
            mean, hw = ensemble_interval(ta, 10_000 + d)
            if abs(mean - ta) <= hw: naive_hits += 1
            if abs(mean - ta) <= c * hw: calib_hits += 1
            if (d+1) % 10 == 0:
                print(f"    [test] alpha={ta} dataset {d+1}/{N_TEST}", flush=True)
        nlo, nhi = binom_ci(naive_hits, N_TEST)
        clo, chi = binom_ci(calib_hits, N_TEST)
        print(f"    alpha={ta} (HELD OUT):")
        print(f"      naive      coverage = {naive_hits}/{N_TEST} = {naive_hits*100/N_TEST:.0f}%"
              f"  95%CI [{nlo*100:.0f}%, {nhi*100:.0f}%]")
        print(f"      CALIBRATED coverage = {calib_hits}/{N_TEST} = {calib_hits*100/N_TEST:.0f}%"
              f"  95%CI [{clo*100:.0f}%, {chi*100:.0f}%]")
        summary[ta] = (naive_hits/N_TEST, calib_hits/N_TEST, c)
    print("\n  VALID calibration = calibrated coverage CI contains 95% at held-out alpha,")
    print("  using a single alpha-agnostic factor -> applicable when alpha is unknown.")
    return c, summary


if __name__ == "__main__":
    print("#"*64)
    print("#  FM-PINN BLOCK 2 (P3 v2) — CORRECTED CALIBRATION")
    print("#  fix #1 fine-grid truth; fix #3 pooled/held-out/(n+1)-quantile/binomial-CI")
    print("#"*64)
    t0 = time.time()
    c, summary = coverage_test_v2()
    print(f"\n  total wall {time.time()-t0:.1f}s")
    print("\n  SUMMARY (held-out alphas):")
    for ta, (nv, cal, cc) in summary.items():
        print(f"    alpha={ta}: naive={nv*100:.0f}%  calibrated={cal*100:.0f}%  factor c={cc:.2f}")
