from __future__ import annotations
import time
import numpy as np
from math import gamma as Gamma
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
NP = np.float64

GAMMA, D_COEF = 1.0, 0.02
XMIN, XMAX, T_END, S_IC = -0.6, 0.6, 1.0, 0.07
NX, NT = 101, 100
ALPHA_GRID = np.round(np.arange(0.50, 0.911, 0.02), 3)   # baseline search grid

try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False


# =====================================================================
#  SHARED: validated L1 reference solver + honest observation model
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
#  METHOD 0 — PINN
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


def pinn_recover(P_obs, obs_mask, x, t, dx, n_iter=3000, lr=5e-3, seed=0, device="cpu"):
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
#  BASELINES
# =====================================================================
def baseline_grid_search(P_obs, obs_mask, x, t, dx):
    """B1: run the forward solver at each alpha on the grid, pick the alpha
       whose forward density best fits the observed density (at observed slices)."""
    obs = np.where(obs_mask)[0]
    best = (None, np.inf)
    for af in ALPHA_GRID:
        _, _, Pa, _ = solve_ref(af, GAMMA, D_COEF, XMIN, XMAX, len(x), T_END, len(t)-1, S_IC)
        mis = np.mean((Pa[obs] - P_obs[obs]) ** 2)
        if mis < best[1]: best = (af, mis)
    return float(best[0])


def baseline_lsq_residual(P_obs, obs_mask, x, t, dx):
    """B2: plug the OBSERVED density into the L1-discretized PDE; pick the alpha
       minimizing the residual. Differentiates noisy data directly -> fragile."""
    Nt = len(t) - 1; xin = x[1:-1]
    best = (None, np.inf)
    for af in ALPHA_GRID:
        bw = l1_weights_np(af, Nt + 1)
        sigma = 1.0 / (Gamma(2 - af) * (1.0 / Nt) ** af)
        px = (P_obs[:, 2:] - P_obs[:, :-2]) / (2 * dx)
        pxx = (P_obs[:, 2:] - 2 * P_obs[:, 1:-1] + P_obs[:, :-2]) / dx**2
        Lp = GAMMA * (P_obs[:, 1:-1] + xin * px) + D_COEF * pxx
        dP = P_obs[1:, 1:-1] - P_obs[:-1, 1:-1]
        idx = np.arange(Nt)[:, None] - np.arange(Nt)[None, :]
        W = np.where(idx >= 0, bw[np.clip(idx, 0, None)], 0.0)
        cap = sigma * (W @ dP)
        r = np.mean((cap - Lp[1:]) ** 2)
        if r < best[1]: best = (af, r)
    return float(best[0])


def baseline_fixed_alpha(P_obs, obs_mask, x, t, dx):
    """B3: null model — assume no memory, alpha=1."""
    return 1.0


# =====================================================================
#  PART A — BASELINE COMPARISON
# =====================================================================
def part_A_comparison(true_alpha=0.7, n_seeds=5, n_iter=3000):
    print("\n" + "="*64)
    print("  PART A — baseline comparison (identical data + noise, error bars)")
    print("="*64)
    x, t, P, dx = solve_ref(true_alpha, GAMMA, D_COEF, XMIN, XMAX, NX, T_END, NT, S_IC)
    mask = np.ones(len(t), bool)
    methods = {
        "PINN (ours)":       lambda Po, s: pinn_recover(Po, mask, x, t, dx, n_iter=n_iter, seed=s),
        "L1 grid-search":    lambda Po, s: baseline_grid_search(Po, mask, x, t, dx),
        "LSQ PDE-residual":  lambda Po, s: baseline_lsq_residual(Po, mask, x, t, dx),
        "fixed-alpha=1":     lambda Po, s: baseline_fixed_alpha(Po, mask, x, t, dx),
    }
    results = {m: {} for m in methods}
    for noise in (0, 5, 20):
        print(f"\n  --- noise = {noise}% (N=1000 returns/window, full observation) ---")
        for mname, fn in methods.items():
            errs = []
            for s in range(n_seeds):
                rng = np.random.default_rng(100 + s)
                Po = observe(P, x, dx, 1000, noise, np.where(mask)[0], rng)
                errs.append(abs(fn(Po, s) - true_alpha))
            errs = np.array(errs)
            results[mname][noise] = (errs.mean(), errs.std())
            print(f"    {mname:18s}: |err| = {errs.mean():.4f} +/- {errs.std():.4f}")
    return results


def plot_comparison(results):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    noises = [0, 5, 20]
    colors = {"PINN (ours)": "#d62728", "L1 grid-search": "#1f77b4",
              "LSQ PDE-residual": "#2ca02c", "fixed-alpha=1": "#7f7f7f"}
    for m, byn in results.items():
        ys = [byn[n][0] for n in noises]; es = [byn[n][1] for n in noises]
        ax.errorbar(noises, ys, yerr=es, fmt='o-', capsize=4,
                    color=colors.get(m, None), label=m)
    ax.set_xlabel("measurement noise (%)"); ax.set_ylabel("|recovered - true| alpha")
    ax.set_yscale("log"); ax.set_title("Part A: alpha-recovery error vs noise (lower = better)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()


# =====================================================================
#  PART B — REAL-DATA BLOCK-BOOTSTRAP ERROR BARS (Task 4 refinement)
# =====================================================================
def load_sp500(start="1985-01-01", end="2025-12-31"):
    if not HAVE_YF:
        raise RuntimeError("yfinance not available. Run: !pip install yfinance")
    df = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    px = df["Close"].to_numpy().astype(float).ravel()
    return np.diff(np.log(px))


def block_bootstrap_stat(series, stat_fn, block_len=63, n_boot=500, rng=None):
    """Moving-block bootstrap: resample contiguous blocks (preserving
       autocorrelation), recompute the statistic, return mean and 95% band."""
    if rng is None: rng = np.random.default_rng(0)
    n = len(series); vals = []
    n_blocks = int(np.ceil(n / block_len))
    starts_pool = n - block_len
    for _ in range(n_boot):
        starts = rng.integers(0, starts_pool, size=n_blocks)
        boot = np.concatenate([series[s:s+block_len] for s in starts])[:n]
        vals.append(stat_fn(boot))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def part_B_realdata():
    print("\n" + "="*64)
    print("  PART B — real S&P 500 statistics with block-bootstrap 95% bands")
    print("="*64)
    r = load_sp500()
    print(f"  loaded {len(r)} daily log-returns")
    ar = np.abs(r - r.mean())

    # D1: |return| autocorrelation at several lags, with bootstrap bands
    print("\n  |return| autocorrelation (mean [95% band]):")
    for L in (1, 5, 10, 20, 50, 100):
        def ac_L(s, L=L):
            s = np.abs(s - s.mean())
            return np.corrcoef(s[:-L], s[L:])[0, 1] if L < len(s) else np.nan
        m, lo, hi = block_bootstrap_stat(r, ac_L, block_len=252, n_boot=400)
        print(f"    lag {L:4d}: {m:.4f}  [{lo:.4f}, {hi:.4f}]")

    # decay-shape statistic: is the autocorr decay power-law or exponential?
    lags = np.array([1, 5, 10, 20, 50, 100], float)
    def decay_gap(s):
        s = np.abs(s - s.mean())
        acs = np.array([np.corrcoef(s[:-int(L)], s[int(L):])[0,1] for L in lags])
        acs = np.clip(acs, 1e-6, None)
        r_pl = _r2(np.log(lags), np.log(acs)); r_ex = _r2(lags, np.log(acs))
        return r_pl - r_ex          # >0 favors power-law, <0 favors exponential
    m, lo, hi = block_bootstrap_stat(r, decay_gap, block_len=252, n_boot=400)
    verdict = ("power-law" if lo > 0 else "exponential" if hi < 0 else "INCONCLUSIVE")
    print(f"\n  decay-shape gap (R2_powerlaw - R2_exp): {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
    print("  (band spanning 0 => data cannot cleanly distinguish power-law vs exponential")
    print("   memory decay — the honest Task-4 conclusion, now with a confidence band)")


def _r2(x, y):
    p = np.polyfit(x, y, 1); pred = np.polyval(p, x)
    ss = np.sum((y - y.mean())**2)
    return 1 - np.sum((y - pred)**2) / ss if ss > 0 else 0.0


# =====================================================================
#  ENTRY
# =====================================================================
if __name__ == "__main__":
    print("#"*64)
    print("#  FM-PINN BLOCK 3 — BASELINE COMPARISON + REAL-DATA ERROR BARS")
    print("#"*64)
    t0 = time.time()

    results = part_A_comparison(true_alpha=0.7, n_seeds=5, n_iter=3000)
    plot_comparison(results)

    try:
        part_B_realdata()
    except Exception as e:
        print(f"\n  Part B skipped (data load issue): {e}")

    print(f"\n  total wall {time.time()-t0:.1f}s")
    print("")
    print("  Part A: on the simplest single-parameter fully-observed problem, the L1")
    print("     grid-search is expected to MATCH the PINN (and be faster). LSQ-residual")
    print("     breaks under noise (differentiates noisy data). fixed-alpha=1 is the")
    print("     no-memory control. The PINN's ADVANTAGE (Blocks 1-2) is joint alpha/gamma/D")
    print("     recovery, sparse-observation robustness, and calibrated uncertainty —")
    print("     NOT raw single-alpha accuracy. Report this honestly.")
    print("  Part B: real-data statistics now carry block-bootstrap 95% bands. If the")
    print("     decay-shape band spans 0, the data cannot cleanly call the memory")
    print("     power-law vs exponential — the honest Task-4 conclusion, quantified.")
