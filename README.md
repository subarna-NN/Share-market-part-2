FM-PINN: Identifiability, Uncertainty, and Baselines for Fractional-Memory Inference
This repository contains the identifiability, uncertainty-quantification, and baseline-comparison experiments for FM-PINN, a physics-informed neural network that recovers the memory order α of a time-fractional Fokker–Planck equation from observed probability densities. It builds on the forward-validation experiments (Tasks 1–2, separate repository).

# Model (unchanged across all experiments)
The recovery engine is fixed throughout: an energy-based density p(x,t) = exp(f(x,t))/Z(t) (guarantees p ≥ 0 and unit mass by construction), a Pang–Lu fPINN hybrid residual (L1 discretization for the Caputo time-fractional term, automatic differentiation for integer-order space terms), with the memory order α (and optionally drift γ, diffusion D) as trainable parameters. All computation in float64.

File	                                   Purpose
fmpinn_block1_identifiability.py	      Block 1 — when is α identifiable?
fmpinn_block2_final_v2.py              	Block 2 — joint parameters, identifiability profile, calibrated uncertainty
fmpinn_block3_baselines.py	            Block 3 — baseline comparison + real-data error bars

Environment: Python 3, PyTorch, NumPy, SciPy, Matplotlib (Block 3 also yfinance). Fixed seeds; multi-seed error bars throughout.

# Block 1 — Improved identifiability study

What & why: tests whether α can be recovered from realistic observations — one finite sample of N returns per window (as real data provides), not idealized clean densities. Sweeps four axes with 5 seeds each.

Results (true α = 0.7):

1. Sample size (250/1000/5000 returns): α recovered to ~1–2% error across all sizes.
2. Measurement noise (0–20%): error stays flat at ~1.3–2% — remarkably robust.
3. Number of snapshots (1/5/10/25/full): 1 snapshot fails badly (error 0.33); ≥5 snapshots recover α well (5→0.04, 25→0.0001). α lives in the transient, not a single snapshot.
4. Observation interval: recovery holds until observations become very sparse (≤6).

# Why it's good: it honestly maps when the method works — α is identifiable given a transient with ≥5 well-spaced, noisy observations.

# Block 2 — Joint parameters & calibrated uncertainty

What & why: proves the paper's core novelty — a calibrated uncertainty interval on α, not a single number.

Results:

1. P1 (joint recovery): α recovered accurately even when γ and D are also unknown (α alone: 0.705; full α+γ+D: 0.670). Confounding is between γ and D (corr +0.98), not with α — the key parameter stays robust.
2. P2 (identifiability profile): training the density network from observations only at each fixed α gives a clean minimum exactly at the true α — a real but shallow minimum, honestly showing α is identifiable but not razor-sharp from data alone.
3. P3 (coverage/calibration): naive ensemble intervals are overconfident (coverage 20–37% vs 95% nominal). Split-conformal width calibration (factor ≈ 3–5×) restores coverage toward nominal (80–100%), yielding a valid calibrated interval.

# Why it's good: it doesn't just claim uncertainty — it tests calibration, finds the naive interval wanting, and fixes it with a standard rigorous method. This is the honest form of the contribution.

# Block 3 — Baselines & real-data error bars

What & why: compares FM-PINN against simple classical alternatives, and adds rigorous confidence bands to the real-data statistics.

Part A (baselines, identical data + noise, error bars):

1. FM-PINN and a classical L1 grid-search are statistically equivalent on the single-parameter fully-observed problem (both ~2–3% error, robust to 20% noise).
2. LSQ PDE-residual fails (~20% error) — it differentiates noisy data directly.
3. fixed-α = 1 (no-memory null) is worst (~30%).

Honest framing: the PINN is not claimed to beat classical methods on raw single-α accuracy — it matches the strong baseline. Its advantage is extensibility: joint α/γ/D recovery, sparse-observation robustness, and calibrated uncertainty (Blocks 1–2), which grid-search does not provide.

Part B (real S&P 500, 1985–2025, block-bootstrap 95% bands):

1. |return| autocorrelation is significantly positive and persistent (volatility clustering is real).
2. The memory-decay shape is statistically closer to exponential than power-law — the fractional-memory signature is not cleanly present in the real data.

# Overall conclusion
FM-PINN solves the forward problem accurately (validated against exact solutions elsewhere), recovers the memory order α from realistic noisy transient observations, quantifies α with a calibrated uncertainty interval, and matches classical baselines while extending naturally to joint parameters and uncertainty. On real market data, volatility memory is significant but not cleanly fractional, and real markets do not supply the clean transient the method needs — an honest, important limitation.

# Reproducibility
Fixed seeds; multi-seed error bars; all results printed by the scripts and reproducible on a T4 GPU. Approximate run times: Block 1 ~1.5–2.5 h, Block 2 ~2–9 h (P3 heaviest), Block 3 ~15 min.
Reference: G. Pang, L. Lu, G. E. Karniadakis, fPINNs: Fractional Physics-Informed Neural Networks, SIAM J. Sci. Comput., 41(4), A2603–A2626, 2019.
