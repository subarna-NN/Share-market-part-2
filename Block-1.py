from __future__ import annotations
import time
import numpy as np
from math import gamma as Gamma
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
NP = np.float64

# ---- fixed physical constants (same as Task 1/2/3) ----
GAMMA, D_COEF = 1.0, 0.02
XMIN, XMAX, T_END, S_IC = -0.6, 0.6, 1.0, 0.07
NX, NT = 101, 100


# =====================================================================
#  GROUND-TRUTH L1 REFERENCE 
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


# =====================================================================
# OBSERVATION MODEL
# =====================================================================
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
#  fPINN RECOVERY ENGINE 
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


def recover_alpha(P_obs, obs_mask, x, t, dx, n_iter=4000, lr=5e-3,
                  seed=0, device='cpu'):
    """
    Recover alpha from the observed (sparse, noisy) densities. 
    """
    torch.manual_seed(seed); np.random.seed(seed)
    Nx = len(x); Nt = len(t) - 1
    xg = torch.tensor(x, device=device); tg = torch.tensor(t, device=device)
    Pdata = torch.tensor(P_obs, device=device)
    mask = torch.tensor(obs_mask, device=device).view(-1, 1)   # (Nt+1,1)
    X = xg.view(1, Nx).expand(Nt + 1, Nx).reshape(-1, 1)
    Tt = tg.view(Nt + 1, 1).expand(Nt + 1, Nx).reshape(-1, 1)
    xin = xg[1:-1].view(1, -1)

    net = EnergyNet().to(device)
    alpha_raw = nn.Parameter(torch.tensor(0.0, device=device))
    def get_alpha(): return 0.05 + 0.9 * torch.sigmoid(alpha_raw)
    opt = torch.optim.Adam(list(net.parameters()) + [alpha_raw], lr=lr)
    dtt = 1.0 / Nt
    idx = (torch.arange(Nt, device=device).view(Nt, 1)
           - torch.arange(Nt, device=device).view(1, Nt))
    tri = (idx >= 0)

    for it in range(1, n_iter + 1):
        f = net.energy(X, Tt, (Nt + 1, Nx))
        logZ = torch.logsumexp(f + np.log(dx), 1, keepdim=True)
        P = torch.exp(f - logZ)
        a = get_alpha()
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
        # data fit ONLY at observed slices
        diff = (P - Pdata) * mask
        n_obs = mask.sum().clamp(min=1)
        l_data = (diff ** 2).sum() / (n_obs * Nx)
        loss = l_pde + 100.0 * l_data
        opt.zero_grad(); loss.backward(); opt.step()

    return float(get_alpha().item())


# =====================================================================
#  SWEEP HELPERS
# =====================================================================
def make_obs_mask(Nt1, n_snapshots=None, interval=None):
    """Build the boolean observation mask over time indices 1."""
    mask = np.zeros(Nt1, dtype=bool)
    mask[0] = True                                  # IC always known
    if interval is not None:
        mask[::interval] = True
    elif n_snapshots is None or n_snapshots >= Nt1:
        mask[:] = True
    else:
        sel = np.linspace(1, Nt1 - 1, n_snapshots).round().astype(int)
        mask[sel] = True
    return mask


def multi_seed(P_obs, mask, x, t, dx, n_seeds, n_iter):
    vals = [recover_alpha(P_obs, mask, x, t, dx, seed=s, n_iter=n_iter)
            for s in range(n_seeds)]
    return float(np.mean(vals)), float(np.std(vals))


# =====================================================================
#  EXPERIMENT
# =====================================================================
TRUE_ALPHA = 0.7
N_SEEDS = 5
N_ITER = 4000


def run_block1():
    x, t, P, dx = solve_ref(TRUE_ALPHA, GAMMA, D_COEF, XMIN, XMAX, NX, T_END, NT, S_IC)
    Nt1 = len(t)
    full_mask = make_obs_mask(Nt1)   # observe everything

    # AXIS 1 — sample size (full observation, no extra noise)
    print("\n" + "="*60 + "\n  AXIS 1 — sample size N (one sample/window, full obs)\n" + "="*60)
    ax1 = []
    for N in (250, 1000, 5000):
        rng = np.random.default_rng(7)
        P_obs = observe(P, x, dx, N, 0.0, np.where(full_mask)[0], rng)
        mrec, srec = multi_seed(P_obs, full_mask, x, t, dx, N_SEEDS, N_ITER)
        print(f"    N={N:5d}: alpha={mrec:.4f} +/- {srec:.4f}  (true {TRUE_ALPHA}, err {abs(mrec-TRUE_ALPHA):.4f})")
        ax1.append((N, mrec, srec))

    # AXIS 2 — noise level (N=1000, full observation)
    print("\n" + "="*60 + "\n  AXIS 2 — measurement noise level (N=1000, full obs)\n" + "="*60)
    ax2 = []
    for noise in (0, 1, 5, 10, 20):
        rng = np.random.default_rng(7)
        P_obs = observe(P, x, dx, 1000, noise, np.where(full_mask)[0], rng)
        mrec, srec = multi_seed(P_obs, full_mask, x, t, dx, N_SEEDS, N_ITER)
        print(f"    noise={noise:2d}%: alpha={mrec:.4f} +/- {srec:.4f}  (err {abs(mrec-TRUE_ALPHA):.4f})")
        ax2.append((noise, mrec, srec))

    # AXIS 3 — number of observed snapshots (N=1000, no extra noise)
    print("\n" + "="*60 + "\n  AXIS 3 — # observed snapshots (N=1000)\n" + "="*60)
    ax3 = []
    for ns in (1, 5, 10, 25, Nt1):
        mask = make_obs_mask(Nt1, n_snapshots=ns)
        rng = np.random.default_rng(7)
        P_obs = observe(P, x, dx, 1000, 0.0, np.where(mask)[0], rng)
        mrec, srec = multi_seed(P_obs, mask, x, t, dx, N_SEEDS, N_ITER)
        lbl = "full" if ns >= Nt1 else str(ns)
        print(f"    snapshots={lbl:>4}: alpha={mrec:.4f} +/- {srec:.4f}  (err {abs(mrec-TRUE_ALPHA):.4f})")
        ax3.append((min(ns, Nt1), mrec, srec))

    # AXIS 4 — observation interval (N=1000, no extra noise)
    print("\n" + "="*60 + "\n  AXIS 4 — observation interval (N=1000)\n" + "="*60)
    ax4 = []
    for iv in (1, 5, 10, 20):
        mask = make_obs_mask(Nt1, interval=iv)
        rng = np.random.default_rng(7)
        P_obs = observe(P, x, dx, 1000, 0.0, np.where(mask)[0], rng)
        mrec, srec = multi_seed(P_obs, mask, x, t, dx, N_SEEDS, N_ITER)
        print(f"    every {iv:2d} steps ({mask.sum()} obs): alpha={mrec:.4f} +/- {srec:.4f}  (err {abs(mrec-TRUE_ALPHA):.4f})")
        ax4.append((iv, mrec, srec))

    return ax1, ax2, ax3, ax4


def plot_block1(ax1, ax2, ax3, ax4):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
    def eb(a, data, xlabel, title, logx=False):
        xs = [d[0] for d in data]; ys = [d[1] for d in data]; es = [d[2] for d in data]
        a.errorbar(xs, ys, yerr=es, fmt='o-', capsize=4, color='#3182bd')
        a.axhline(TRUE_ALPHA, color='k', ls=':', label=f'true α={TRUE_ALPHA}')
        if logx: a.set_xscale('log')
        a.set_xlabel(xlabel); a.set_ylabel('recovered α'); a.set_title(title)
        a.legend(); a.grid(alpha=0.3)
    eb(ax[0,0], ax1, 'sample size N', 'Axis 1: sample size', logx=True)
    eb(ax[0,1], ax2, 'noise level (%)', 'Axis 2: measurement noise')
    eb(ax[1,0], ax3, '# observed snapshots', 'Axis 3: snapshot count', logx=True)
    eb(ax[1,1], ax4, 'observation interval (steps)', 'Axis 4: observation interval')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    print("#"*62)
    print("")
    print(f"#  true alpha={TRUE_ALPHA}, {N_SEEDS} seeds, {N_ITER} iters, grid {NX}x{NT}")
    print("#  honest observation: ONE finite sample per observed window")
    print("#"*62)
    t0 = time.time()
    ax1, ax2, ax3, ax4 = run_block1()
    plot_block1(ax1, ax2, ax3, ax4)
    print(f"\n  total wall {time.time()-t0:.1f}s")
    print("\n  HOW TO READ:")
    print("  Axis 1: error should fall as N grows (more data = better).")
    print("  Axis 2: error should rise with noise.")
    print("  Axis 3: error should fall as more snapshots are observed;")
    print("          1 snapshot should be badly non-identifiable (Task 3 Part C).")
    print("  Axis 4: sparser observation (larger interval) should degrade recovery.")
    print("  Together these map the REGIME where alpha is identifiable from")
    print("  realistic, sparse, noisy observations — the honest core of the paper.")
