# holonomygap.py
# Compute SU(2) holonomy gaps for the tuned K4 torus.
#
# Step 0: simulate K4 trajectory at K_run from a YAML config.
# Step 1: construct an analytic SU(2) lift from a face-flux pattern
#         (here F = [-triad_phi, 0, 0, -triad_phi]) using a least-squares
#         edge-angle solve B phi = F, and verify that the SU(2) holonomies
#         match the abelian trace-angle relation to machine precision.
# Step 2 (dynamic): compute fluxes F_f(t) from theta_traj and holonomy
#         gaps along the trajectory.
#
# Usage (example):
#   python holonomygap.py \
#       --config ./configs/k4_quat_baseline.yaml \
#       --T 200000 --burn 100000 --subsample 100
#
# Requires: numpy, PyYAML. The K4 simulation itself is provided by
#           rtg_k4_quat_scout_v5.simulate_traj.

from __future__ import annotations

import os
import argparse
import json
from typing import Dict, Tuple, List

import numpy as np
import yaml

from rtg_core import ensure_dir, now_tag, wrap_pi
from rtg_k4_quat_scout_v5 import K4Params, build_k4_spec, simulate_traj

# ------------------------------------------------------------
# K4 combinatorics: edges & face incidences
# ------------------------------------------------------------

# Canonical edge indexing for K4:
#   e0: (0,1)
#   e1: (0,2)
#   e2: (0,3)
#   e3: (1,2)
#   e4: (1,3)
#   e5: (2,3)
K4_EDGES: List[Tuple[int, int]] = [
    (0, 1),  # e0
    (0, 2),  # e1
    (0, 3),  # e2
    (1, 2),  # e3
    (1, 3),  # e4
    (2, 3),  # e5
]

# Faces (triangles) with oriented edge incidences.
# Entries are (edge_index, sign), where sign=+1 means we traverse the
# edge in its canonical direction (as in K4_EDGES), and sign=-1 means
# the opposite orientation. This is consistent with the 4x6 incidence
# matrix B used in the addendum.
#
#   Face 0: triangle (0,1,2)
#   Face 1: triangle (0,1,3)
#   Face 2: triangle (0,2,3)
#   Face 3: triangle (1,2,3)
K4_FACE_EDGES: Dict[int, List[Tuple[int, int]]] = {
    0: [(0, +1), (1, -1), (3, +1)],  # row [1, -1, 0, 1, 0, 0]
    1: [(0, +1), (2, -1), (4, +1)],  # row [1,  0, -1, 0, 1, 0]
    2: [(1, +1), (2, -1), (5, +1)],  # row [0,  1, -1, 0, 0, 1]
    3: [(3, +1), (4, -1), (5, +1)],  # row [0,  0,  0, 1,-1, 1]
}


def k4_incidence_matrix() -> np.ndarray:
    """Return the 4x6 face-edge incidence matrix B consistent with K4_FACE_EDGES."""
    B = np.zeros((4, 6), dtype=float)
    for f in range(4):
        for e_idx, sgn in K4_FACE_EDGES[f]:
            B[f, e_idx] = float(sgn)
    return B


# ------------------------------------------------------------
# SU(2) helpers (aligned-axis rotations)
# ------------------------------------------------------------

def su2_from_vec(g: np.ndarray) -> np.ndarray:
    """
    Map a real 3-vector g to an su(2) matrix A = (i/2) g · σ,
    where σ = (σ_x, σ_y, σ_z) are the Pauli matrices.

    If ||g|| = θ, then exp(A) has trace 2 cos(θ/2).
    """
    g = np.asarray(g, dtype=float)
    gx, gy, gz = g
    # Pauli matrices
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    A = 0.5j * (gx * sigma_x + gy * sigma_y + gz * sigma_z)
    return A


def su2_edge_unitary(phi: float, axis: np.ndarray | None = None) -> np.ndarray:
    """
    Build an SU(2) edge transport U_e = exp(A_e) with A_e = (i/2) g·σ,
    where g = phi * axis, and 'axis' is a unit 3-vector.

    For aligned axes, exp(sum A_e) = exp(A_total) with g_total = sum g_e,
    so the holonomy around a face whose flux is F satisfies
        Tr(H_f)/2 = cos(F/2).
    """
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("Axis vector must be non-zero.")
    axis = axis / norm
    g = phi * axis
    A = su2_from_vec(g)
    # Matrix exponential via eigendecomposition (2x2, cheap)
    w, V = np.linalg.eig(A)
    exp_w = np.exp(w)
    U = V @ np.diag(exp_w) @ np.linalg.inv(V)
    return U


def su2_face_holonomy(U_edges: List[np.ndarray], face_idx: int) -> np.ndarray:
    """
    Compute holonomy H_f = product of U_e^{sign} around face f.

    Because all U_e are aligned-axis rotations, the product order does
    not matter, but we follow the face-edge incidence list for clarity.
    """
    H = np.eye(2, dtype=complex)
    for e_idx, sgn in K4_FACE_EDGES[face_idx]:
        U = U_edges[e_idx]
        if sgn < 0:
            U = U.conj().T  # inverse in SU(2)
        H = U @ H
    return H


# ------------------------------------------------------------
# Flux from trajectory
# ------------------------------------------------------------

def dynamic_face_flux_from_theta(theta: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    Compute abelian face fluxes F_f(theta) from a single theta snapshot,
    using the oriented face-edge structure and coupling phases alphas.

    For each oriented edge (i->j) we use the phase difference
      Δ_ij = theta_j - theta_i - alphas[i,j],
    and sum with the appropriate sign around each face, then wrap to (-π, π].
    """
    theta = np.asarray(theta, dtype=float)
    F = np.zeros(4, dtype=float)
    for f in range(4):
        acc = 0.0
        for e_idx, sgn in K4_FACE_EDGES[f]:
            i, j = K4_EDGES[e_idx]
            acc += sgn * (theta[j] - theta[i] - alphas[i, j])
        F[f] = wrap_pi(acc)
    return F


# ------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------

def load_k4_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_params_from_config(cfg: Dict[str, object]) -> Tuple[K4Params, float]:
    """
    Build K4Params from YAML config dict and choose K_run.

    Priority for K_run:
      1. explicit K_run in config (if present),
      2. K_fit in config (fallback),
      3. midpoint of K1_star, K2_star.
    """
    kwargs: Dict[str, object] = {}
    if "scheme" in cfg:
        kwargs["scheme"] = cfg["scheme"]
    if "dt" in cfg:
        kwargs["dt"] = cfg["dt"]
    if "gamma" in cfg:
        kwargs["gamma"] = cfg["gamma"]
    if "directed" in cfg:
        kwargs["directed"] = bool(cfg["directed"])
    if "eps_asym" in cfg:
        kwargs["eps_asym"] = cfg["eps_asym"]
    if "deg_norm" in cfg:
        kwargs["deg_norm"] = bool(cfg["deg_norm"])
    if "topology" in cfg:
        kwargs["topology"] = cfg["topology"]
    if "w_diag" in cfg:
        kwargs["w_diag"] = cfg["w_diag"]
    if "sigma2" in cfg:
        kwargs["sigma2"] = cfg["sigma2"]
    if "triad_phi" in cfg:
        kwargs["triad_phi"] = cfg["triad_phi"]

    p = K4Params(**kwargs)

    if "K_run" in cfg:
        K_run = float(cfg["K_run"])
    elif "K_fit" in cfg:
        K_run = float(cfg["K_fit"])
    elif "K1_star" in cfg and "K2_star" in cfg:
        K_run = 0.5 * (float(cfg["K1_star"]) + float(cfg["K2_star"]))
    else:
        raise ValueError("Config must contain K_run, or K_fit, or both K1_star and K2_star.")

    return p, K_run


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="K4 SU(2) holonomy gap diagnostics")
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file (e.g. configs/k4_quat_baseline.yaml)",
    )
    ap.add_argument(
        "--T",
        type=int,
        default=200000,
        help="Total simulation steps for K4 trajectory",
    )
    ap.add_argument(
        "--burn",
        type=int,
        default=100000,
        help="Burn-in steps (discarded before sampling)",
    )
    ap.add_argument(
        "--subsample",
        type=int,
        default=100,
        help="Subsampling factor for dynamic flux/gap diagnostics",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for JSON summary (default figs_k4_holonomy_*)",
    )
    args = ap.parse_args()

    cfg = load_k4_config(args.config)
    p, K_run = build_params_from_config(cfg)
    g = build_k4_spec(p)

    outdir = args.outdir or os.path.join(f"figs_k4_holonomy_{now_tag()}")
    ensure_dir(outdir)

    print(f"[config] Using K_run = {K_run:.9f}")
    print(
        f"[config] topology={p.topology}, dt={p.dt}, gamma={p.gamma}, "
        f"w_diag={p.w_diag}, sigma2={p.sigma2}, triad_phi={getattr(p, 'triad_phi', 0.0)}"
    )

    # --------------------------------------------------------
    # 0) Simulate K4 trajectory at K_run
    # --------------------------------------------------------
    T = int(args.T)
    burn = int(args.burn)
    subsample = max(1, int(args.subsample))

    print(f"[sim] T={T}, burn={burn}, subsample={subsample}")
    theta_traj, omega_traj, r_series, vel_traj = simulate_traj(
        K_run, p, g, T=T, burn=burn, noise=0.0, rng_seed=0
    )
    T_eff = theta_traj.shape[0]
    if T_eff == 0:
        raise RuntimeError("Effective trajectory length T_eff=0; increase T or decrease burn.")
    r_mean = float(r_series.mean())
    r_std = float(r_series.std())
    print(
        f"[sim] Got theta_traj with shape {theta_traj.shape}, "
        f"mean r={r_mean:.4f}, std r={r_std:.4f}"
    )

    # --------------------------------------------------------
    # 1) Analytic flux pattern and least-squares SU(2) lift
    # --------------------------------------------------------
    triad_phi = float(cfg.get("triad_phi", 2.0 * np.pi / 3.0))
    # Based on the addendum: two frustrated faces (0 and 3) carry flux -triad_phi,
    # and the other two are flat.
    F_faces_analytic = np.array([-triad_phi, 0.0, 0.0, -triad_phi], dtype=float)

    print("\n[flux] Using analytic face flux pattern (addendum assumption):")
    for f in range(4):
        Ff = F_faces_analytic[f]
        print(
            f"  Face {f}: F_target = {Ff:+.6f} rad, "
            f"cos(F/2) = {np.cos(Ff/2):+.8f}"
        )

    # Build incidence matrix B and its minimal-norm solver for B phi = F
    B = k4_incidence_matrix()
    print("\n[lift] Incidence matrix B (faces x edges):")
    print(B)

    # Minimal-norm solution phi = B^T (B B^T)^+ F
    BBt = B @ B.T
    BBt_inv = np.linalg.pinv(BBt)
    B_pinv = B.T @ BBt_inv
    phi_edges = B_pinv @ F_faces_analytic  # shape (6,)

    print("\n[lift] Edge angles φ (one per K4 edge):")
    for e_idx, phi_e in enumerate(phi_edges):
        print(f"  φ_e{e_idx} = {phi_e:+.6f} rad")

    B_phi = B @ phi_edges
    residual = B_phi - F_faces_analytic
    C_star = float(np.sum(phi_edges**2))

    print("\n[lift] Linear constraint check: B φ vs F_target")
    print(f"  F_target = {F_faces_analytic}")
    print(f"  B φ      = {B_phi}")
    print(f"  residual = {residual}")
    print(f"  gauge energy C* = sum_e φ_e^2 ≈ {C_star:.6f}")

    # Build aligned-axis SU(2) edge transports
    axis_z = np.array([0.0, 0.0, 1.0], dtype=float)
    U_edges_analytic: List[np.ndarray] = [
        su2_edge_unitary(float(phi_edges[e]), axis_z) for e in range(6)
    ]

    # Per-face holonomy and gap for analytic lift
    analytic_gaps: Dict[int, Dict[str, float]] = {}
    print("\nHolonomy per face (analytic lift):")
    for f in range(4):
        Ff = F_faces_analytic[f]
        H_f = su2_face_holonomy(U_edges_analytic, f)
        tr_half = float(np.real(np.trace(H_f) / 2.0))
        cos_half = float(np.cos(Ff / 2.0))
        gap = tr_half - cos_half
        print(
            f"  Face {f}: F={Ff:+.6f}, cos(F/2)={cos_half:+.8f}, "
            f"Tr(H_f)/2={tr_half:+.8f}, gap={gap:+.3e}"
        )
        analytic_gaps[f] = {
            "F": Ff,
            "cos_half": cos_half,
            "tr_half": tr_half,
            "gap": gap,
            "abs_gap": abs(gap),
        }

    print("\nHolonomy gap summary (analytic lift):")
    for f in range(4):
        print(
            f"  Face {f}: |gap| ≈ {analytic_gaps[f]['abs_gap']:.3e}"
        )

    # --------------------------------------------------------
    # 2) Dynamic flux and holonomy gaps from trajectory
    # --------------------------------------------------------
    print("\n[dynamic] Computing F_f(t) and holonomy gaps along theta_traj ...")

    sample_indices = list(range(0, T_eff, subsample))
    n_samples = len(sample_indices)
    F_dyn = np.zeros((n_samples, 4), dtype=float)
    gaps_dyn = np.zeros((n_samples, 4), dtype=float)

    for idx, t_idx in enumerate(sample_indices):
        theta_t = theta_traj[t_idx]
        F_t = dynamic_face_flux_from_theta(theta_t, g.alphas)
        F_dyn[idx, :] = F_t

        # Use the same analytic SU(2) edge transports for all t;
        # since the flux pattern is fixed by alphas in this tuned case,
        # this should reproduce the trace-angle relation up to roundoff.
        for f in range(4):
            H_f = su2_face_holonomy(U_edges_analytic, f)
            tr_half = float(np.real(np.trace(H_f) / 2.0))
            cos_half = float(np.cos(F_t[f] / 2.0))
            gaps_dyn[idx, f] = tr_half - cos_half

    # Dynamic statistics per face
    dyn_flux_stats: Dict[int, Dict[str, float]] = {}
    dyn_gap_stats: Dict[int, Dict[str, float]] = {}

    print("\n[dynamic] Flux statistics per face (using sampled timesteps):")
    for f in range(4):
        mean_F = float(F_dyn[:, f].mean())
        std_F = float(F_dyn[:, f].std())
        dyn_flux_stats[f] = {"mean_F": mean_F, "std_F": std_F}
        print(f"  Face {f}: mean(F)={mean_F:+.4f} rad, std(F)={std_F:+.4e} rad")

    print("\n[dynamic] Holonomy gap statistics per face (along trajectory):")
    for f in range(4):
        gabs = np.abs(gaps_dyn[:, f])
        mean_abs = float(gabs.mean())
        std_abs = float(gabs.std())
        dyn_gap_stats[f] = {
            "mean_abs_gap": mean_abs,
            "std_abs_gap": std_abs,
        }
        print(
            f"  Face {f}: mean_abs_gap={mean_abs:.3e}, std={std_abs:.3e}"
        )

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------
    summary = {
        "config_path": args.config,
        "K_run": float(K_run),
        "T": T,
        "burn": burn,
        "subsample": subsample,
        "triad_phi": triad_phi,
        "r_mean": r_mean,
        "r_std": r_std,
        "F_faces_analytic": F_faces_analytic.tolist(),
        "phi_edges_analytic": phi_edges.tolist(),
        "gauge_energy_analytic": C_star,
        "B": B.tolist(),
        "B_phi_analytic": B_phi.tolist(),
        "B_residual_analytic": residual.tolist(),
        "analytic_gaps": analytic_gaps,
        "dynamic_flux_stats": dyn_flux_stats,
        "dynamic_gap_stats": dyn_gap_stats,
    }

    out_path = os.path.join(outdir, "holonomy_gap_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] Summary -> {out_path}")


if __name__ == "__main__":
    main()
