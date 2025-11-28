#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTG: Discrete Kuramoto on graphs with phase-lag frustration + Jacobian probe.

- K3 (triangle) with target U(1) flux F_target set by alpha_ij lags
- Two-plaquette graph (two triangles) with independent fluxes
- U(1) flux + SU(2) trace–angle check
- Jacobian of the one-step map, with *gauge removal* (projects out global phase)
- K-sweep, detuning sweep
- All figures/CSVs are saved to ./figs_jacobian (no interactive windows)

Dependencies: numpy, matplotlib (Agg backend). No seaborn, no external I/O.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensure figure windows are not opened
import matplotlib.pyplot as plt

# ------------------------------- Utilities -------------------------------- #

TWOPI = 2.0 * np.pi

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def mod2pi(x):
    """Map scalar/array x to (-pi, pi]. Returns a new array (no in-place writes)."""
    return (x + np.pi) % (2.0*np.pi) - np.pi

def angle_unwrap(x1d):
    """Unwrap a 1D angle array (radians)."""
    return np.unwrap(x1d)

def order_param(theta_vec):
    """Kuramoto order parameter r = | mean_j e^{i θ_j} | for one time slice."""
    return np.abs(np.mean(np.exp(1j * theta_vec)))

def energy_on_edges(theta, edges, alpha, kappa_hat=None):
    """
    Simple diagnostic energy:
        E = - sum_{(i,j) in edges,i<j} [ w_ij * cos( (θ_j-θ_i) - α_ij ) ]
    If kappa_hat is None, all weights are 1.
    """
    E = 0.0
    for (i, j) in edges:
        phi = theta[j] - theta[i] - alpha[i, j]
        w = 1.0 if (kappa_hat is None) else kappa_hat[i, j]
        E -= w * np.cos(phi)
    return E

def savefig(fig, path, dpi=160):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

# ------------------------------- Graphs ----------------------------------- #

def build_k3(F_target):
    """
    Triangle K3 with signed couplings = +1 and phase lags α_ij such that
    the measured flux equals -F_target. We put all lag on edge 2->0.
    """
    N = 3
    neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    edges = [(0,1), (1,2), (0,2)]
    deg = np.array([2.0, 2.0, 2.0])

    kappa_hat = np.ones((N, N), dtype=float)
    np.fill_diagonal(kappa_hat, 0.0)
    kappa_hat = 0.5*(kappa_hat + kappa_hat.T)

    alpha = np.zeros((N, N), dtype=float)
    alpha[2, 0] = F_target
    alpha[0, 2] = -F_target
    return N, neighbors, edges, deg, kappa_hat, alpha

def build_two_plaquettes(F1, F2):
    """
    Two triangles sharing edge (0,2):
      tri A: (0,1,2,0)   -> put lag on 0->1
      tri B: (0,2,3,0)   -> put lag on 2->3
    """
    N = 4
    neighbors = {
        0: [1, 2, 3],
        1: [0, 2],
        2: [0, 1, 3],
        3: [0, 2],
    }
    edges = [(0,1), (0,2), (0,3), (1,2), (2,3)]
    deg = np.array([3.0, 2.0, 3.0, 2.0])

    kappa_hat = np.ones((N, N), dtype=float)
    np.fill_diagonal(kappa_hat, 0.0)
    kappa_hat = 0.5*(kappa_hat + kappa_hat.T)

    alpha = np.zeros((N, N), dtype=float)
    alpha[0,1] = F1; alpha[1,0] = -F1
    alpha[2,3] = F2; alpha[3,2] = -F2
    return N, neighbors, edges, deg, kappa_hat, alpha

# ----------------------------- Dynamics ----------------------------------- #

def step_discrete_kuramoto(theta, K, neighbors, deg, kappa_hat, alpha, delta):
    """
    θ_v(t+1) = θ_v(t) + δ_v
               + (K/deg(v)) * Σ_{w∈N(v)} κ̂_{vw} sin(θ_w - θ_v - α_{vw})
    """
    N = len(theta)
    dtheta = np.array(delta, dtype=float)
    for v in range(N):
        s = 0.0
        for w in neighbors[v]:
            s += kappa_hat[v, w]*np.sin(theta[w] - theta[v] - alpha[v, w])
        dtheta[v] += (K/deg[v]) * s
    return theta + dtheta   # keep unwrapped

def simulate_map(N, neighbors, edges, deg, kappa_hat, alpha,
                 K=1.5, T=20000, burn=15000, detuning=None, seed=1):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=N)

    delta = np.zeros(N, dtype=float) if detuning is None else np.array(detuning, dtype=float)

    theta_t = np.zeros((T, N), dtype=float)
    r_t     = np.zeros(T, dtype=float)
    E_t     = np.zeros(T, dtype=float)

    for t in range(T):
        theta_t[t] = theta
        r_t[t]     = order_param(theta)
        E_t[t]     = energy_on_edges(theta, edges, alpha, kappa_hat)
        theta      = step_discrete_kuramoto(theta, K, neighbors, deg, kappa_hat, alpha, delta)

    return {
        "theta_t":   theta_t,
        "r_t":       r_t,
        "E_t":       E_t,
        "theta_post": theta_t[burn:],
        "r_post":     r_t[burn:],
        "theta_last": theta_t[-1],
    }

# -------------------------- Holonomy / Flux ------------------------------- #

def flux_on_cycle(theta, alpha, cycle_nodes):
    """
    U(1) flux on a directed cycle like [0,1,2,0] using corrected differences
    (θ_j - θ_i - α_ij). Returns F in (-π, π].
    """
    total = 0.0
    for i, j in zip(cycle_nodes[:-1], cycle_nodes[1:]):
        total += mod2pi(theta[j] - theta[i] - alpha[i, j])
    return mod2pi(total)

def flux_series_on_cycle(theta_series, alpha, cycle_nodes):
    """Unwrapped flux over time on a given cycle."""
    Fs = np.array([flux_on_cycle(th, alpha, cycle_nodes) for th in theta_series], dtype=float)
    return angle_unwrap(Fs)

def half_trace_su2_from_flux(F):
    """0.5 Tr(H) = cos(F/2) for SU(2) holonomy with rotation angle F."""
    return np.cos(0.5*F)

# --------------------------- Jacobian Probe -------------------------------- #

def jacobian_matrix(theta, K, neighbors, deg, kappa_hat, alpha):
    """
    Jacobian J = ∂θ(t+1)/∂θ(t) of the discrete map at state 'theta'.
    Entries:
      J_vv = 1 + (K/deg(v)) * Σ_{w∈N(v)} [ -κ̂_{vw} cos(θ_w-θ_v-α_{vw}) ]
      J_vw =     (K/deg(v)) *       κ̂_{vw} cos(θ_w-θ_v-α_{vw})      for w∈N(v), w≠v
    """
    N = len(theta)
    J = np.eye(N, dtype=float)
    for v in range(N):
        for w in neighbors[v]:
            c = kappa_hat[v, w]*np.cos(theta[w] - theta[v] - alpha[v, w])
            J[v, v] += (K/deg[v]) * (-c)
            J[v, w] += (K/deg[v]) * ( c)
    return J

def gauge_reduce(J):
    """
    Remove the global-phase (gauge) direction.  Let u = 1/√N * (1,1,...,1)^T.
    Build an orthonormal basis Q = [u | Q_⊥] via QR; return J_⊥ = Q_⊥^T J Q_⊥.
    """
    N = J.shape[0]
    u = np.ones((N, 1), dtype=float) / np.sqrt(N)
    A = np.eye(N, dtype=float)
    A[:, 0] = u[:, 0]
    Q, _ = np.linalg.qr(A)
    Qperp = Q[:, 1:]                   # N x (N-1)
    J_reduced = Qperp.T @ J @ Qperp    # (N-1) x (N-1)
    return J_reduced, Qperp

def jacobian_probe(theta_series, K, neighbors, deg, kappa_hat, alpha,
                   outdir, tag="k3", last_M=200):
    """
    Evaluate the Jacobian on the last M states:
      - non-gauge spectral radius vs time
      - eigenvalues in complex plane (last snapshot), both full and reduced
      - CSV dump of eigenvalues (full and reduced) for reproducibility
    """
    ensure_dir(outdir)
    M = min(last_M, theta_series.shape[0])
    thetas = theta_series[-M:]

    sr_full, sr_nongauge = [], []
    eigs_full_all, eigs_ng_all = [], []

    for th in thetas:
        J = jacobian_matrix(th, K, neighbors, deg, kappa_hat, alpha)
        vals_full = np.linalg.eigvals(J)
        J_red, _ = gauge_reduce(J)
        vals_ng   = np.linalg.eigvals(J_red)

        eigs_full_all.append(vals_full)
        eigs_ng_all.append(vals_ng)
        sr_full.append(np.max(np.abs(vals_full)))
        sr_nongauge.append(np.max(np.abs(vals_ng)))

    sr_full = np.asarray(sr_full)
    sr_nongauge = np.asarray(sr_nongauge)

    # --- Spectral radius vs time (non-gauge) ---
    fig = plt.figure(figsize=(6, 3))
    plt.plot(np.arange(M), sr_nongauge)
    plt.axhline(1.0, linestyle="--", label="unit circle")
    plt.xlabel("time index (last M steps)")
    plt.ylabel("max |λ| (non-gauge)")
    plt.title("Jacobian spectral radius (non-gauge) vs time")
    plt.legend()
    savefig(fig, os.path.join(outdir, f"{tag}_jac_sr_nongauge_vs_time.png"))

    # --- Complex-plane eigenvalues (last snapshot) ---
    last_full = eigs_full_all[-1]
    last_ng   = eigs_ng_all[-1]
    ang = np.linspace(0, TWOPI, 360)

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(last_full.real, last_full.imag, s=22, label="full eigs")
    plt.plot(np.cos(ang), np.sin(ang), linestyle="--", label="unit circle")
    plt.axhline(0, color="k", linewidth=0.7)
    plt.axvline(0, color="k", linewidth=0.7)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Jacobian eigs (last snapshot) — FULL")
    plt.legend()
    savefig(fig, os.path.join(outdir, f"{tag}_jac_eigs_full_last.png"))

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(last_ng.real, last_ng.imag, s=22, label="non-gauge eigs")
    plt.plot(np.cos(ang), np.sin(ang), linestyle="--", label="unit circle")
    plt.axhline(0, color="k", linewidth=0.7)
    plt.axvline(0, color="k", linewidth=0.7)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Jacobian eigs (last snapshot) — NON‑GAUGE")
    plt.legend()
    savefig(fig, os.path.join(outdir, f"{tag}_jac_eigs_nongauge_last.png"))

    # --- CSV dump (full & non-gauge) ---
    csv_path = os.path.join(outdir, f"{tag}_jacobian_spectrum.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # headers
        hf = [f"eig_full_{k}_real" for k in range(len(last_full))] + \
             [f"eig_full_{k}_imag" for k in range(len(last_full))]
        hg = [f"eig_ng_{k}_real" for k in range(len(last_ng))] + \
             [f"eig_ng_{k}_imag" for k in range(len(last_ng))]
        w.writerow(["time_index"] + hf + hg)
        # rows
        for ti, (vf, vg) in enumerate(zip(eigs_full_all, eigs_ng_all)):
            row = [ti] + [v.real for v in vf] + [v.imag for v in vf] + \
                        [u.real for u in vg] + [u.imag for u in vg]
            w.writerow(row)

    return {
        "sr_full_mean": float(np.mean(sr_full)),
        "sr_full_last": float(np.max(np.abs(last_full))),
        "sr_ng_mean": float(np.mean(sr_nongauge)),
        "sr_ng_last": float(np.max(np.abs(last_ng))),
        "top_ng_moduli_last": np.sort(np.abs(last_ng))[::-1][:3].tolist(),
    }

# ------------------------- Experiment helpers ------------------------------ #

def plot_time_series(r_post, F_post, E_post, outdir, prefix):
    ensure_dir(outdir)

    fig = plt.figure(figsize=(6, 3))
    plt.plot(np.arange(len(r_post)), r_post)
    plt.xlabel("time (steps after burn-in)")
    plt.ylabel("r(t)")
    plt.title("Order parameter r(t)")
    savefig(fig, os.path.join(outdir, f"{prefix}_r_vs_time.png"))

    fig = plt.figure(figsize=(6, 3))
    plt.plot(np.arange(len(F_post)), F_post)
    plt.xlabel("time (steps after burn-in)")
    plt.ylabel("unwrapped flux F(t) [rad]")
    plt.title("Flux on triangle vs time (unwrapped)")
    savefig(fig, os.path.join(outdir, f"{prefix}_flux_vs_time.png"))

    fig = plt.figure(figsize=(6, 3))
    plt.plot(np.arange(len(E_post)), E_post)
    plt.xlabel("time step")
    plt.ylabel("E(t)")
    plt.title("Energy vs time")
    savefig(fig, os.path.join(outdir, f"{prefix}_energy_vs_time.png"))

def k_sweep(builder, K_values, T, burn, outdir, tag):
    rs, Fs, sr_ng = [], [], []
    for K in K_values:
        N, neigh, edges, deg, kappa, alpha = builder()
        sim = simulate_map(N, neigh, edges, deg, kappa, alpha, K=K, T=T, burn=burn)
        r_mean = float(np.mean(sim["r_post"]))
        rs.append(r_mean)
        F = flux_on_cycle(sim["theta_last"], alpha, [0,1,2,0]) if N==3 else np.nan
        Fs.append(F)

        stats = jacobian_probe(sim["theta_post"], K, neigh, deg, kappa, alpha,
                               outdir=outdir, tag=f"{tag}_K{K:.2f}", last_M=120)
        sr_ng.append(stats["sr_ng_last"])

    # figures
    fig = plt.figure(figsize=(14, 4))
    ax1 = plt.subplot(1,3,1)
    ax1.plot(K_values, rs, marker="o")
    ax1.set_xlabel("K")
    ax1.set_ylabel("mean r (post-burn)")
    ax1.set_title("Coherence vs K")

    ax2 = plt.subplot(1,3,2)
    ax2.plot(K_values, Fs, marker="o")
    ax2.set_xlabel("K"); ax2.set_ylabel("F_triangle (rad)")
    ax2.set_title("Flux vs K")

    ax3 = plt.subplot(1,3,3)
    ax3.plot(K_values, sr_ng, marker="o")
    ax3.axhline(1.0, linestyle="--", label="|λ|=1")
    ax3.set_xlabel("K"); ax3.set_ylabel("max |λ| (non-gauge)")
    ax3.set_title("Jacobian spectral radius vs K")
    ax3.legend()

    savefig(fig, os.path.join(outdir, f"{tag}_sweep_r_F_eigradius_vs_K.png"))

    # CSV
    with open(os.path.join(outdir, f"{tag}_k_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "r_mean", "F_triangle", "max_abs_lambda_nongauge"])
        for K, r, Fv, sr in zip(K_values, rs, Fs, sr_ng):
            w.writerow([K, r, Fv, sr])

def detuning_sweep(builder, deltas, K, T, burn, outdir, tag):
    r_means, F_vals = [], []
    for D in deltas:
        N, neigh, edges, deg, kappa, alpha = builder()
        det = np.zeros(N, dtype=float)
        # opposite detunings on nodes 0 and 2 to keep mean-zero
        if N >= 1: det[0] = -D
        if N >= 3: det[2] =  D
        sim = simulate_map(N, neigh, edges, deg, kappa, alpha, K=K, T=T, burn=burn, detuning=det)
        r_means.append(float(np.mean(sim["r_post"])))
        F_vals.append(flux_on_cycle(sim["theta_last"], alpha, [0,1,2,0]) if N==3 else np.nan)

    fig = plt.figure(figsize=(6, 3))
    plt.plot(deltas, r_means, marker="o")
    plt.xlabel("detuning magnitude Δ")
    plt.ylabel("mean r (post-burn)")
    plt.title("Coherence vs detuning (K fixed)")
    savefig(fig, os.path.join(outdir, f"{tag}_coherence_vs_detuning.png"))

    if not any(np.isnan(F_vals)):
        fig = plt.figure(figsize=(6, 3))
        plt.plot(deltas, F_vals, marker="o")
        plt.xlabel("detuning magnitude Δ")
        plt.ylabel("F_triangle (rad)")
        plt.title("Flux vs detuning (K fixed)")
        savefig(fig, os.path.join(outdir, f"{tag}_flux_vs_detuning.png"))

    with open(os.path.join(outdir, f"{tag}_detuning_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detuning", "r_mean", "F_triangle"])
        for D, r, Fv in zip(deltas, r_means, F_vals):
            w.writerow([D, r, Fv])

# -------------------------------- Main ------------------------------------ #

def main():
    outdir = "figs_jacobian"
    ensure_dir(outdir)

    # --- Single-run on K3 with a target flux ---
    K = 1.5
    F_target = 2.0*np.pi/3.0    # we construct α so that measured flux = -F_target
    N, neigh, edges, deg, kappa, alpha = build_k3(F_target)
    sim = simulate_map(N, neigh, edges, deg, kappa, alpha, K=K, T=20000, burn=15000, seed=1)

    r_mean = float(np.mean(sim["r_post"]))
    F_last = flux_on_cycle(sim["theta_last"], alpha, [0,1,2,0])
    ht = half_trace_su2_from_flux(F_last)

    print(f"K = {K:.3f}")
    print(f"Mean order parameter r (post-burn) = {r_mean:.3f}")
    print(f"Measured U(1) flux F_triangle (last mean) = {F_last:.6f} rad")
    print(f"0.5 * Tr(H_su2) = {ht:.6f}  (cos(F/2) = {np.cos(0.5*F_last):.6f})\n")

    # time-series diagnostics
    F_series = flux_series_on_cycle(sim["theta_post"], alpha, [0,1,2,0])
    E_post   = np.array([energy_on_edges(th, edges, alpha, kappa) for th in sim["theta_post"]], dtype=float)
    plot_time_series(sim["r_post"], F_series, E_post, outdir, prefix="k3_single")

    # Jacobian probe (non-gauge)
    stats = jacobian_probe(sim["theta_post"], K, neigh, deg, kappa, alpha,
                           outdir=outdir, tag="k3_single", last_M=200)
    print("=== Jacobian probe summary (single run) ===")
    print(f"K={K:.3f}, F={F_last:.6f} rad, r_mean={r_mean:.3f}, "
          f"max|λ| (nongauge)={stats['sr_ng_last']:.4f}")
    print("Top |λ| (nongauge, last snapshot):", stats["top_ng_moduli_last"])

    # --- K-sweep (uses the same target flux) ---
    K_values = np.linspace(0.20, 2.50, 12)
    k_sweep(lambda: build_k3(F_target), K_values, T=6000, burn=4000, outdir=outdir, tag="k3")

    # --- Detuning sweep (visualizing robustness / NS picture) ---
    deltas = np.linspace(0.0, 0.05, 11)
    detuning_sweep(lambda: build_k3(F_target), deltas, K=K, T=8000, burn=6000, outdir=outdir, tag="k3")

    # --- Two-plaquette demo ---
    F1 = 2.0*np.pi/3.0
    F2 = np.pi/2.0
    N2, n2, e2, d2, kap2, a2 = build_two_plaquettes(F1, F2)
    sim2 = simulate_map(N2, n2, e2, d2, kap2, a2, K=K, T=16000, burn=11000, seed=2)

    triA = [0,1,2,0]; triB = [0,2,3,0]
    FA = flux_series_on_cycle(sim2["theta_post"], a2, triA)
    FB = flux_series_on_cycle(sim2["theta_post"], a2, triB)

    fig = plt.figure(figsize=(6, 3))
    plt.plot(np.arange(len(FA)), FA, label="triangle A (0-1-2-0)")
    plt.plot(np.arange(len(FB)), FB, label="triangle B (0-2-3-0)")
    plt.xlabel("time (steps after burn-in)")
    plt.ylabel("unwrapped flux F(t) [rad]")
    plt.title("Two plaquettes: flux vs time")
    plt.legend()
    savefig(fig, os.path.join(outdir, "two_plaquettes_flux_vs_time.png"))

    print(f"\n=== Two-plaquette demo ===")
    print(f"Measured tri A flux (last): {FA[-1]:.6f} rad  (target -F1)")
    print(f"Measured tri B flux (last): {FB[-1]:.6f} rad  (target -F2)")

    stats2 = jacobian_probe(sim2["theta_post"], K, n2, d2, kap2, a2,
                            outdir=outdir, tag="two_plaquettes", last_M=200)
    print(f"Two-plaquette non-gauge |λ|max (last snapshot) = {stats2['sr_ng_last']:.6f}")

    print(f"\nAll figures and CSVs saved under: {outdir}")

if __name__ == "__main__":
    main()
