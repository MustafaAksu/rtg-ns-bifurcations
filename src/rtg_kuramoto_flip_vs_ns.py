#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTG: Flip vs Neimark–Sacker on K3 with phase-lag frustration
- Discrete Kuramoto map with per-edge phase-lag alpha_ij
- U(1) flux and SU(2) trace–angle check
- Jacobian probe (gauge-removed)
- Sweep K: detect first instability and classify as flip (real, arg≈π) or NS (complex pair)
- Save plots and CSV (no plt.show())

Author: RTG Collaboration
"""

import os
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # ensure headless
import matplotlib.pyplot as plt

PI = np.pi
TAU = 2 * np.pi
RNG = np.random.default_rng(1)

# -----------------------------
# Utilities
# -----------------------------

def mod2pi(x):
    """Wrap to (-pi, pi]. Works on scalars/arrays."""
    return (x + PI) % TAU - PI

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def savefig(path, bbox_inches="tight", dpi=140):
    plt.savefig(path, bbox_inches=bbox_inches, dpi=dpi)
    plt.close()

# -----------------------------
# K3 graph + phase lag
# -----------------------------

def build_k3():
    """
    K3 neighbors and degrees (undirected graph, we store both directions in neighbors)
    """
    neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    deg = {0: 2, 1: 2, 2: 2}
    undirected_edges = [(0,1), (1,2), (0,2)]  # for energy reporting
    return neighbors, deg, undirected_edges

def build_alpha_matrix(F_target):
    """
    Create directed phase-lag matrix alpha_ij so that sum_{0->1->2->0} alpha_ij = F_target.
    Convention: alpha_{ji} = - alpha_{ij}.
    """
    a = F_target / 3.0
    alpha = np.zeros((3,3), dtype=float)
    alpha[0,1] = a; alpha[1,0] = -a
    alpha[1,2] = a; alpha[2,1] = -a
    alpha[2,0] = a; alpha[0,2] = -a
    return alpha

def build_Kij(K, asym_edge=None, eta=0.0):
    """
    Per-directed-edge coupling matrix Kij.
    If asym_edge=(u,v) and eta!=0, scale that directed edge by (1+eta) and the reverse by (1-eta)
    to break symmetry slightly.
    """
    Kij = np.zeros((3,3), dtype=float)
    for i in range(3):
        for j in range(3):
            if i != j:
                Kij[i,j] = K
    if asym_edge is not None and eta != 0.0:
        u, v = asym_edge
        # small opposite tweaks to preserve average; ensures asymmetry
        Kij[u, v] = K * (1.0 + eta)
        Kij[v, u] = K * (1.0 - eta)
    return Kij

# -----------------------------
# Dynamics
# -----------------------------

def step_map(theta, Kij, alpha, neighbors, deg, delta):
    """
    One synchronous step of the discrete Kuramoto map with directed phase-lag alpha_ij
    and directed per-edge coupling Kij.
    theta: shape (3,)
    Kij, alpha: shape (3,3)
    delta: per-node detunings (3,)
    """
    N = 3
    theta_next = np.empty_like(theta)
    for i in range(N):
        s = 0.0
        for j in neighbors[i]:
            s += (Kij[i,j] / deg[i]) * np.sin(theta[j] - theta[i] - alpha[i,j])
        theta_next[i] = theta[i] + delta[i] + s
    return theta_next

def run_sim(K, F_target, T=20000, burn=15000,
            detune_amp=0.0, asym_edge=None, asym_eta=0.0,
            seed=1):
    """
    Run the map; return dict with time traces and last snapshot.
    """
    RNG = np.random.default_rng(seed)
    neighbors, deg, undirected_edges = build_k3()
    alpha = build_alpha_matrix(F_target)
    Kij = build_Kij(K, asym_edge=asym_edge, eta=asym_eta)

    # small heterogeneity in detunings if requested
    if detune_amp > 0:
        delta = RNG.uniform(-detune_amp, detune_amp, size=3)
    else:
        delta = np.zeros(3)

    # initial phases
    theta = RNG.uniform(-PI, PI, size=3)

    # traces (post-burn)
    r_list = []
    F_list = []
    E_list = []

    for t in range(T):
        theta = step_map(theta, Kij, alpha, neighbors, deg, delta)

        if t >= burn:
            # order parameter
            z = np.exp(1j * theta)
            r = np.abs(np.mean(z))
            r_list.append(r)

            # U(1) flux on the oriented cycle 0->1->2->0
            F = mod2pi( (theta[1]-theta[0]-alpha[0,1]) +
                        (theta[2]-theta[1]-alpha[1,2]) +
                        (theta[0]-theta[2]-alpha[2,0]) )
            F_list.append(F)

            # simple energy proxy: sum over undirected edges
            E = 0.0
            for (i,j) in undirected_edges:
                E += -np.cos(theta[j]-theta[i]-alpha[i,j])
            E_list.append(E)

    out = {
        "theta_last": theta.copy(),
        "r_list": np.array(r_list),
        "F_list": np.array(F_list),
        "E_list": np.array(E_list),
        "alpha": alpha,
        "Kij": Kij,
        "detune": delta,
        "neighbors": neighbors,
        "deg": deg
    }
    return out

# -----------------------------
# Flux & SU(2) check
# -----------------------------

def su2_trace_angle_from_flux(F):
    """Return 0.5*Tr(H) and cos(F/2) for the SU(2) check (they should match)."""
    return np.cos(0.5 * F), np.cos(0.5 * F)

# -----------------------------
# Jacobian and eigen-analysis
# -----------------------------

def jacobian_at(theta, Kij, alpha, neighbors, deg):
    """
    Jacobian of the discrete map f(theta) at snapshot theta.
    J_ii = 1 - (K/d_i) sum_j cos(Δθ_ij - α_ij)
    J_ij = (K/d_i) cos(Δθ_ij - α_ij), j in N(i)
    with K replaced by Kij[i,j] per directed edge.
    """
    N = len(theta)
    J = np.zeros((N,N), dtype=float)
    for i in range(N):
        s = 0.0
        for j in neighbors[i]:
            c = np.cos(theta[j] - theta[i] - alpha[i,j])
            J[i,j] = (Kij[i,j] / deg[i]) * c
            s += (Kij[i,j] / deg[i]) * c
        J[i,i] = 1.0 - s
    return J

def nongauge_eigs(J, tol_near_one=1e-6):
    """
    Remove the gauge eigenvalue closest to 1 and return remaining eigenvalues.
    """
    evs = np.linalg.eigvals(J)
    idx_gauge = np.argmin(np.abs(evs - 1.0))
    mask = np.ones_like(evs, dtype=bool)
    mask[idx_gauge] = False
    return evs[mask], evs

def leading_mode(evs_nongauge, tol_im=1e-7):
    """
    Return leading eigenvalue (max |λ|), its radius and phase, and a mode tag: 'real' or 'complex'.
    """
    idx = np.argmax(np.abs(evs_nongauge))
    lam = evs_nongauge[idx]
    rad = np.abs(lam)
    ang = np.angle(lam)
    mode = "complex" if abs(np.imag(lam)) > tol_im else "real"
    return lam, rad, ang, mode

# -----------------------------
# Sweeps and classification
# -----------------------------

def sweep_K(K_values, F_target,
            detune_amp=0.0, asym_edge=None, asym_eta=0.0,
            T=20000, burn=15000, seed=1):
    """
    For each K, run and probe the Jacobian at the last snapshot.
    Return a dict of arrays and classification of first crossing (if any).
    """
    r_mean_list, F_last_list = [], []
    rad_list, ang_list, mode_list = [], [], []
    K_star, cross_type = None, None
    lam_star = None

    for K in K_values:
        sim = run_sim(K, F_target, T=T, burn=burn,
                      detune_amp=detune_amp, asym_edge=asym_edge, asym_eta=asym_eta, seed=seed)
        r_mean = float(np.mean(sim["r_list"]))
        F_last = float(np.mean(sim["F_list"][-50:])) if len(sim["F_list"])>=50 else float(sim["F_list"][-1])
        J = jacobian_at(sim["theta_last"], sim["Kij"], sim["alpha"], sim["neighbors"], sim["deg"])
        evs_nongauge, evs_all = nongauge_eigs(J)
        lam, rad, ang, mode = leading_mode(evs_nongauge)

        r_mean_list.append(r_mean)
        F_last_list.append(F_last)
        rad_list.append(rad)
        ang_list.append(ang)
        mode_list.append(mode)

    # detect first crossing of |λ|=1
    rad_arr = np.array(rad_list)
    cross_idx = None
    for k in range(1, len(K_values)):
        if rad_arr[k-1] < 1.0 and rad_arr[k] >= 1.0:
            cross_idx = k
            break

    if cross_idx is not None:
        K_star = float(K_values[cross_idx])
        cross_type = mode_list[cross_idx]  # 'real' -> flip, 'complex' -> NS
        lam_star = complex(rad_list[cross_idx] * np.exp(1j*ang_list[cross_idx]))
    return {
        "K_values": np.array(K_values, float),
        "r_mean": np.array(r_mean_list, float),
        "F_last": np.array(F_last_list, float),
        "rad": np.array(rad_list, float),
        "ang": np.array(ang_list, float),
        "mode": np.array(mode_list, dtype=object),
        "K_star": K_star,
        "cross_type": cross_type,
        "lam_star": lam_star
    }

# -----------------------------
# Plot helpers
# -----------------------------

def plot_sweep(summary, title_prefix, outdir, basename):
    K = summary["K_values"]
    r = summary["r_mean"]
    F = summary["F_last"]
    rad = summary["rad"]
    ang = summary["ang"]
    mode = summary["mode"]
    K_star = summary["K_star"]
    cross_type = summary["cross_type"]

    plt.figure(figsize=(13,3.6))
    # r(K)
    ax = plt.subplot(1,3,1)
    ax.plot(K, r, marker='o')
    ax.set_title("Coherence vs K")
    ax.set_xlabel("K")
    ax.set_ylabel("mean r (post-burn)")

    # flux(K)
    ax = plt.subplot(1,3,2)
    ax.plot(K, F, marker='o')
    ax.set_title("Flux vs K (should be invariant)")
    ax.set_xlabel("K")
    ax.set_ylabel("F_triangle (rad)")

    # spectral radius(K)
    ax = plt.subplot(1,3,3)
    ax.plot(K, rad, marker='o', label='max |λ| (non-gauge)')
    ax.axhline(1.0, ls='--', color='k', alpha=0.6)
    if K_star is not None:
        ax.axvline(K_star, ls='--', color='tab:red', alpha=0.7,
                   label=f'cross @ K≈{K_star:.2f} ({cross_type})')
    ax.set_title("Jacobian spectral radius vs K")
    ax.set_xlabel("K"); ax.set_ylabel("max |λ|")
    ax.legend(loc='best')
    plt.suptitle(f"{title_prefix}  —  flip vs NS diagnosis")
    savefig(os.path.join(outdir, f"{basename}_sweep.png"))

    # angle plot
    plt.figure(figsize=(6,3.6))
    plt.plot(K, ang, marker='o')
    plt.axhline(np.pi, ls='--', color='k', alpha=0.5)
    plt.axhline(-np.pi, ls='--', color='k', alpha=0.5)
    if K_star is not None:
        plt.axvline(K_star, ls='--', color='tab:red', alpha=0.7)
    plt.title(f"{title_prefix} — arg(λ_max) vs K")
    plt.xlabel("K"); plt.ylabel("arg λ_max (rad)")
    savefig(os.path.join(outdir, f"{basename}_angles.png"))

def write_csv(summary, outpath):
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "r_mean", "F_last", "rad_max", "arg_max", "mode"])
        for i in range(len(summary["K_values"])):
            w.writerow([f"{summary['K_values'][i]:.6f}",
                        f"{summary['r_mean'][i]:.6f}",
                        f"{summary['F_last'][i]:.6f}",
                        f"{summary['rad'][i]:.6f}",
                        f"{summary['ang'][i]:.6f}",
                        summary["mode"][i]])

# -----------------------------
# Main
# -----------------------------

def main():
    outdir = "figs_flip_ns"
    ensure_dir(outdir)

    # Common settings
    F_target = 2.0 * np.pi / 3.0  # target curvature (+2π/3); measured F will be -F_target with our orientation
    T = 20000
    burn = 15000
    seed = 1

    # K grid
    K_values = np.round(np.linspace(0.2, 2.5, 12), 2)

    # ---------- Case A: symmetric K3 (flip expected) ----------
    sumA = sweep_K(K_values, F_target,
                   detune_amp=0.0, asym_edge=None, asym_eta=0.0,
                   T=T, burn=burn, seed=seed)

    # ---------- Case B: weak symmetry breaking (NS expected) ----------
    # Option 1: tiny detunings (uncomment both to try alt routes)
    detune_amp = 1e-2
    # Option 2: slight edge asymmetry on (0,1)
    asym_edge = (0,1)
    asym_eta = 0.02

    # You can use either detune OR asym (both work); we'll use detune by default and keep asym=0
    sumB = sweep_K(K_values, F_target,
                   detune_amp=detune_amp, asym_edge=None, asym_eta=0.0,
                   T=T, burn=burn, seed=seed)

    # ------------- Save figures & CSV -------------
    write_csv(sumA, os.path.join(outdir, "flip_symmetric_k_sweep.csv"))
    write_csv(sumB, os.path.join(outdir, "ns_heterogeneous_k_sweep.csv"))

    plot_sweep(sumA, title_prefix="Symmetric K3 (flip expected)",
               outdir=outdir, basename="flip_case")
    plot_sweep(sumB, title_prefix=f"Heterogeneous K3 (NS expected, detune={detune_amp})",
               outdir=outdir, basename="ns_case")

    # ------------- Print concise report -------------
    def report(name, s):
        print(f"\n=== {name} ===")
        if s["K_star"] is None:
            print("No |λ|=1 crossing found on the grid.")
        else:
            typ = "flip (period-doubling)" if s["cross_type"] == "real" else "Neimark–Sacker"
            print(f"First crossing at K ≈ {s['K_star']:.3f} as {typ}.")
            print(f"Leading λ at crossing: |λ|≈{np.abs(s['lam_star']):.4f}, arg λ≈{np.angle(s['lam_star']):.4f} rad")

    report("Symmetric K3", sumA)
    report("Heterogeneous K3", sumB)

    # ------------- Single run diagnostics (optional quick check) -------------
    K_demo = 1.5
    sim = run_sim(K_demo, F_target, T=T, burn=burn, detune_amp=0.0, asym_edge=None, asym_eta=0.0, seed=seed)
    F_last = float(np.mean(sim["F_list"][-50:])) if len(sim["F_list"])>=50 else float(sim["F_list"][-1])
    half_tr, cos_half = su2_trace_angle_from_flux(F_last)
    print(f"\nSingle-run SU(2) check at K={K_demo:.2f}: 0.5 Tr(H)={half_tr:.6f}, cos(F/2)={cos_half:.6f}")
    print(f"Measured flux F ≈ {F_last:.6f} rad (should be ≈ {-F_target:.6f} rad with this orientation)")
    print(f"All outputs written to: {outdir}")

if __name__ == "__main__":
    main()
