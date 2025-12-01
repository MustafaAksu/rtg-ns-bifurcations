# rtg_k4_fit_ns.py
# Fit coupled Neimark–Sacker normal-form coefficients on K4
#
# Usage (example):
#   python rtg_k4_fit_ns.py \
#       --config ./configs/k4_quat_baseline.yaml \
#       --outdir figs_k4_ns_fit_baseline \
#       --bursts 80 --T_burst 60 \
#       --eps_min 1e-3 --eps_max 1e-1 \
#       --ridge 0.0 \
#       --d_rho_dK1 1.8e-3 --d_rho_dK2 -2.1e-3 --deltaK_post 5e-7 \
#       --sim_nf 1000
#
# Requires: PyYAML (`pip install pyyaml`), scipy (`pip install scipy`).

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import fsolve

from rtg_core import (
    ensure_dir,
    now_tag,
    gauge_free_basis_theta,
    block_diag,
    jacobian_inertial,
    step_inertial,
)
from rtg_k4_quat_scout_v5 import K4Params, build_k4_spec


# ------------------------------------------------------------
# Eigen-mode selection with biorthogonal left eigenvectors
# ------------------------------------------------------------

def select_ns_modes_biorth(Jr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex, complex]:
    """
    From the reduced Jacobian Jr (on gauge-free subspace),
    pick two dominant complex NS eigenmodes, compute left eigenvectors,
    and biorthonormalize them so that <w_j, v_k> = δ_{jk}.

    Returns
    -------
    v1, v2, w1, w2 : np.ndarray
        Right/left eigenvectors in reduced coordinates (length 2(n-1)).
    lam1, lam2 : complex
        Eigenvalues associated with v1, v2.
    """
    # Right eigen-decomposition: Jr v = λ v
    eigvals_R, eigvecs_R = np.linalg.eig(Jr)

    # Left eigen-decomposition: Jr^H w = \bar{λ} w
    eigvals_L, eigvecs_L = np.linalg.eig(Jr.conj().T)

    # Select indices with positive imaginary part (one per complex pair)
    idx_complex: List[int] = [i for i, lam in enumerate(eigvals_R) if lam.imag > 1e-10]
    if len(idx_complex) < 2:
        # Fallback: just take two eigenvalues of largest modulus
        order = np.argsort(np.abs(eigvals_R))[::-1]
        i1, i2 = order[0], order[1]
    else:
        # Sort by modulus descending and take the top two
        idx_complex.sort(key=lambda i: -abs(eigvals_R[i]))
        i1, i2 = idx_complex[0], idx_complex[1]

    lam1, lam2 = eigvals_R[i1], eigvals_R[i2]
    v1 = eigvecs_R[:, i1]
    v2 = eigvecs_R[:, i2]

    def match_left(lam: complex, v: np.ndarray) -> np.ndarray:
        # Find left eigenvector whose eigenvalue matches conj(lam)
        idx = np.argmin(np.abs(eigvals_L - np.conj(lam)))
        w_raw = eigvecs_L[:, idx]
        denom = np.dot(w_raw.conj(), v)
        if abs(denom) > 1e-12:
            w = w_raw / denom
        else:
            # Fallback: just normalize to unit norm
            w = w_raw / (np.linalg.norm(w_raw) + 1e-16)
        return w

    w1 = match_left(lam1, v1)
    w2 = match_left(lam2, v2)

    return v1, v2, w1, w2, lam1, lam2


# ------------------------------------------------------------
# Fixed point solver for frustrated K4
# ------------------------------------------------------------

def get_fixed_pt(K: float, p: K4Params, g) -> np.ndarray:
    """
    Solve for a phase-locked fixed point θ* satisfying

        sum_j kappa[i,j] sin(θ_j - θ_i - alpha[i,j]) = 0   for all i

    The factor K/deg is common and omitted, so K does not appear explicitly.
    """
    n = g.n

    def eqs(th: np.ndarray) -> np.ndarray:
        res = np.zeros(n, dtype=float)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i and g.kappa[i, j] != 0.0:
                    s += g.kappa[i, j] * math.sin(th[j] - th[i] - g.alphas[i, j])
            res[i] = s
        return res

    # v2-inspired initial guess; good enough for fsolve
    th_guess = np.array([0.0, 0.12, 0.25, -0.08], dtype=float)
    th_star = fsolve(eqs, th_guess, xtol=1e-10)
    return np.mod(th_star, 2.0 * math.pi)


# ------------------------------------------------------------
# Burst generation & projection onto NS modes
# ------------------------------------------------------------

def generate_bursts(
    K_fit: float,
    p: K4Params,
    g,
    v1: np.ndarray,
    v2: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    Q: np.ndarray,
    n_bursts: int = 64,
    T_burst: int = 50,
    eps_min: float = 1e-3,
    eps_max: float = 1e-1,
    trim: int = 5,
    rng_seed: int = 123,
    theta_star: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate short nonlinear bursts near the fixed point at K_fit,
    subtract x* = (θ*, 0), project onto NS modes using left eigenvectors,
    unwrap phases per burst, and collect z1, z2 trajectories.

    Returns
    -------
    z1_all, z2_all : np.ndarray
        Flattened complex time series used for regression.
    """
    rng = np.random.default_rng(rng_seed)
    n = g.n

    if theta_star is None:
        theta_star = get_fixed_pt(K_fit, p, g)

    x_star = np.concatenate([theta_star, np.zeros(n, dtype=float)])

    z1_list: List[np.ndarray] = []
    z2_list: List[np.ndarray] = []

    for _ in range(n_bursts):
        # Random amplitude scale in [eps_min, eps_max] (log-uniform)
        log_eps = rng.uniform(math.log(eps_min), math.log(eps_max))
        eps = float(math.exp(log_eps))

        # Random real coefficients for Re/Im combinations
        coeffs = rng.normal(size=4)

        # Initial reduced perturbation in span{v1, v2}
        x_red0 = (
            coeffs[0] * np.real(v1)
            + coeffs[1] * np.imag(v1)
            + coeffs[2] * np.real(v2)
            + coeffs[3] * np.imag(v2)
        )
        x_red0 = eps * x_red0

        # Lift back to full state and shift by fixed point
        x0 = Q @ x_red0 + x_star
        theta = x0[:n].copy()
        vel = x0[n:].copy()

        z1_b: List[complex] = []
        z2_b: List[complex] = []

        # Run T_burst+1 steps so we have pairs (t, t+1)
        for _t in range(T_burst + 1):
            x = np.concatenate([theta, vel])
            xi = x - x_star  # subtract fixed point
            x_red = Q.T @ xi

            # Biorthogonal projection onto NS modes
            z1 = np.dot(w1.conj(), x_red)
            z2 = np.dot(w2.conj(), x_red)

            z1_b.append(z1)
            z2_b.append(z2)

            theta, vel = step_inertial(theta, vel, K_fit, p, g, noise=0.0)

        # Phase-unwrapped versions to reduce artificial Im(β) from jumps
        z1_b = np.asarray(z1_b, dtype=np.complex128)
        z2_b = np.asarray(z2_b, dtype=np.complex128)

        z1_ang = np.angle(z1_b)
        z1_un_ang = np.cumsum(np.diff(z1_ang, prepend=z1_ang[0]))
        z1_b = np.abs(z1_b) * np.exp(1j * z1_un_ang)

        z2_ang = np.angle(z2_b)
        z2_un_ang = np.cumsum(np.diff(z2_ang, prepend=z2_ang[0]))
        z2_b = np.abs(z2_b) * np.exp(1j * z2_un_ang)

        if T_burst + 1 <= trim + 1:
            continue

        z1_list.append(z1_b[trim:])
        z2_list.append(z2_b[trim:])

    if not z1_list:
        raise RuntimeError("No valid bursts collected; reduce trim or increase T_burst.")

    # Concatenate across bursts
    z1_all = np.concatenate(z1_list)
    z2_all = np.concatenate(z2_list)
    return z1_all, z2_all


# ------------------------------------------------------------
# Complex ridge least squares and residuals
# ------------------------------------------------------------

def ridge_complex_ls(Phi: np.ndarray, y: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """
    Solve complex ridge least squares:

        y ≈ Phi @ beta

    via (Phi^H Phi + λ I) beta = Phi^H y.
    """
    y = y.reshape(-1)
    Phi = Phi.reshape(y.shape[0], -1)

    A = Phi.conj().T @ Phi
    b = Phi.conj().T @ y

    if lam > 0.0:
        A = A + lam * np.eye(A.shape[0], dtype=A.dtype)

    beta = np.linalg.solve(A, b)
    return beta


def rel_residual(Phi: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """
    Relative residual ||y - Phi beta|| / ||y||.
    """
    num = np.linalg.norm(y - Phi @ beta)
    den = np.linalg.norm(y) + 1e-16
    return float(num / den)


# ------------------------------------------------------------
# Fit NS coefficients from projected data
# ------------------------------------------------------------

def fit_ns_coeffs_from_data(
    z1: np.ndarray,
    z2: np.ndarray,
    lam1: complex,
    lam2: complex,
    ridge: float = 0.0,
    d_rho_dK1: float = 1.8e-3,
    d_rho_dK2: float = -2.1e-3,
    deltaK_post: float = 5e-7,
) -> Dict[str, object]:
    """
    Given flattened z1,z2 time series and linear multipliers lam1,lam2,
    fit cubic NS coefficients:

        z1' = lam1 z1 - β11 |z1|^2 z1 - β12 |z2|^2 z1
        z2' = lam2 z2 - β21 |z1|^2 z2 - β22 |z2|^2 z2

    using complex ridge LS (with optional cross-validated λ).

    Returns a JSON-safe dict with β_jk, a_jk = Re β_jk, Δ, μ_j,
    several amplitude predictions, and diagnostics.
    """
    if z1.size < 2 or z2.size < 2:
        raise ValueError("Not enough data to fit NS coefficients.")

    # Align lengths and form (z_curr, z_next) pairs
    N = min(z1.size, z2.size) - 1
    z1_curr = z1[:N]
    z1_next = z1[1:N + 1]
    z2_curr = z2[:N]
    z2_next = z2[1:N + 1]

    # Mode-1 residual (cubic part)
    r1 = z1_next - lam1 * z1_curr
    phi11 = (np.abs(z1_curr) ** 2) * z1_curr
    phi12 = (np.abs(z2_curr) ** 2) * z1_curr
    Phi1 = np.column_stack([phi11, phi12])

    # Mode-2 residual
    r2 = z2_next - lam2 * z2_curr
    phi21 = (np.abs(z1_curr) ** 2) * z2_curr
    phi22 = (np.abs(z2_curr) ** 2) * z2_curr
    Phi2 = np.column_stack([phi21, phi22])

    # We want r ≈ -Φ β  =>  -r ≈ Φ β
    y1 = -r1
    y2 = -r2

    # Determine ridge parameter(s)
    if ridge is not None and ridge > 0.0:
        ridge1 = ridge2 = float(ridge)
        print(f"[ridge] Using user-specified λ={ridge:.3f}")
    else:
        lam_grid = np.array([0.0, 0.02, 0.05, 0.1], dtype=float)

        def best_lambda(Phi: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
            best_lam = lam_grid[0]
            best_res = float("inf")
            for lam in lam_grid:
                beta_try = ridge_complex_ls(Phi, y, lam)
                res = rel_residual(Phi, y, beta_try)
                if res < best_res:
                    best_res = res
                    best_lam = lam
            return best_lam, best_res

        lam1_opt, r1_cv = best_lambda(Phi1, y1)
        lam2_opt, r2_cv = best_lambda(Phi2, y2)
        ridge1, ridge2 = lam1_opt, lam2_opt
        print(
            f"[ridge] CV λ1_opt={lam1_opt:.3f} (r1_min={r1_cv:.3f}), "
            f"λ2_opt={lam2_opt:.3f} (r2_min={r2_cv:.3f})"
        )

    # Final fits with chosen ridge
    beta1 = ridge_complex_ls(Phi1, y1, ridge1)
    beta2 = ridge_complex_ls(Phi2, y2, ridge2)

    beta11, beta12 = beta1[0], beta1[1]
    beta21, beta22 = beta2[0], beta2[1]

    resid1 = rel_residual(Phi1, y1, beta1)
    resid2 = rel_residual(Phi2, y2, beta2)

    # Real parts for amplitude equations
    a11 = float(beta11.real)
    a12 = float(beta12.real)
    a21 = float(beta21.real)
    a22 = float(beta22.real)

    Delta = a11 * a22 - a12 * a21

    # Small-μ approximation from λ_j ≈ exp(μ_j + i ω_j)
    mu1_fit = math.log(abs(lam1))
    mu2_fit = math.log(abs(lam2))
    omega1 = float(np.angle(lam1))
    omega2 = float(np.angle(lam2))

    if abs(Delta) > 1e-12:
        s1_fit = (a22 * mu1_fit - a12 * mu2_fit) / Delta
        s2_fit = (a11 * mu2_fit - a21 * mu1_fit) / Delta
    else:
        s1_fit = float("nan")
        s2_fit = float("nan")

    # Signed μ_post from slopes d|λ|/dK and δK_post
    mu1_post = d_rho_dK1 * deltaK_post
    mu2_post = d_rho_dK2 * deltaK_post

    if abs(Delta) > 1e-12:
        s1_post = (a22 * mu1_post - a12 * mu2_post) / Delta
        s2_post = (a11 * mu2_post - a21 * mu1_post) / Delta
    else:
        s1_post = float("nan")
        s2_post = float("nan")

    # Symmetric test with μ1 = μ2 = |μ_post| (rough magnitude)
    mu_sym = 0.5 * (abs(mu1_post) + abs(mu2_post))
    if abs(Delta) > 1e-12:
        s1_sym = (a22 - a12) * mu_sym / Delta
        s2_sym = (a11 - a21) * mu_sym / Delta
        ratio_sym = math.sqrt(s1_sym / s2_sym) if (s1_sym > 0 and s2_sym > 0) else float("nan")
    else:
        s1_sym = float("nan")
        s2_sym = float("nan")
        ratio_sym = float("nan")

    # Condition numbers of normal matrices (rough well-conditioning check)
    cond1 = float(np.linalg.cond(Phi1.conj().T @ Phi1))
    cond2 = float(np.linalg.cond(Phi2.conj().T @ Phi2))

    # Imaginary-to-frequency ratios as a crude "phase artifact" indicator
    im_ratio11 = float(abs(beta11.imag) / abs(omega1)) if abs(omega1) > 1e-12 else float("nan")
    im_ratio22 = float(abs(beta22.imag) / abs(omega2)) if abs(omega2) > 1e-12 else float("nan")

    result: Dict[str, object] = {
        # β_jk (split into real/imag for JSON)
        "beta11_real": float(beta11.real),
        "beta11_imag": float(beta11.imag),
        "beta12_real": float(beta12.real),
        "beta12_imag": float(beta12.imag),
        "beta21_real": float(beta21.real),
        "beta21_imag": float(beta21.imag),
        "beta22_real": float(beta22.real),
        "beta22_imag": float(beta22.imag),
        # a_jk and determinant
        "a11": a11,
        "a12": a12,
        "a21": a21,
        "a22": a22,
        "Delta": float(Delta),
        # Linear eigen-data
        "lam1_real": float(lam1.real),
        "lam1_imag": float(lam1.imag),
        "lam2_real": float(lam2.real),
        "lam2_imag": float(lam2.imag),
        "mu1_fit": float(mu1_fit),
        "mu2_fit": float(mu2_fit),
        "omega1": omega1,
        "omega2": omega2,
        # Residuals
        "resid1_rel": resid1,
        "resid2_rel": resid2,
        # Fit-scale amplitudes
        "s1_fit": float(s1_fit),
        "s2_fit": float(s2_fit),
        "A1_fit": float(math.sqrt(s1_fit)) if s1_fit > 0 else float("nan"),
        "A2_fit": float(math.sqrt(s2_fit)) if s2_fit > 0 else float("nan"),
        # Post-bifurcation (signed) amplitudes from slopes
        "mu1_post": float(mu1_post),
        "mu2_post": float(mu2_post),
        "s1_post": float(s1_post),
        "s2_post": float(s2_post),
        "A1_post": float(math.sqrt(max(0.0, s1_post))),
        "A2_post": float(math.sqrt(max(0.0, s2_post))),
        # Symmetric-μ test
        "s1_sym": float(s1_sym),
        "s2_sym": float(s2_sym),
        "A1_sym": float(math.sqrt(s1_sym)) if s1_sym > 0 else float("nan"),
        "A2_sym": float(math.sqrt(s2_sym)) if s2_sym > 0 else float("nan"),
        "ratio_sym": float(ratio_sym),
        # Conditioning / phase diagnostics
        "cond_Phi1": cond1,
        "cond_Phi2": cond2,
        "im_ratio11": im_ratio11,
        "im_ratio22": im_ratio22,
        # Ridge choices and sample count
        "ridge1": float(ridge1),
        "ridge2": float(ridge2),
        "N_samples": int(N),
    }
    return result


# ------------------------------------------------------------
# Polar NF simulation for validation
# ------------------------------------------------------------

def sim_nf_polar(
    a11: float,
    a12: float,
    a21: float,
    a22: float,
    mu1: float,
    mu2: float,
    omega1: float,
    omega2: float,
    T_sim: int = 1000,
    rho1_0: float = 0.01,
    rho2_0: float = 0.01,
    outdir: str = ".",
) -> None:
    """
    Simulate the (discrete-time) amplitude equations

        rho1_{t+1} = rho1_t (1 + mu1 - a11 rho1_t^2 - a12 rho2_t^2),
        rho2_{t+1} = rho2_t (1 + mu2 - a21 rho1_t^2 - a22 rho2_t^2),

    as a crude check on the sign pattern of a_{jk} and μ_j.
    """
    rho1 = float(rho1_0)
    rho2 = float(rho2_0)
    rho_traj = np.zeros((T_sim, 2), dtype=float)

    for t in range(T_sim):
        rho1_new = rho1 * (1.0 + mu1 - a11 * rho1**2 - a12 * rho2**2)
        rho2_new = rho2 * (1.0 + mu2 - a21 * rho1**2 - a22 * rho2**2)
        rho1, rho2 = rho1_new, rho2_new
        rho_traj[t, 0] = rho1
        rho_traj[t, 1] = rho2

    plt.figure()
    plt.plot(rho_traj[:, 0], label="rho1")
    plt.plot(rho_traj[:, 1], label="rho2")
    plt.xlabel("steps")
    plt.ylabel("amplitude")
    plt.title(f"NF amplitude sim (mu1={mu1:.2e}, mu2={mu2:.2e})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "nf_sim_amps.png"), dpi=150)
    plt.close()


# ------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------

def load_k4_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_params_from_config(cfg: Dict[str, object]) -> Tuple[K4Params, float, float | None]:
    """
    Build K4Params from YAML config dict and choose K_fit.

    Priority for K_fit:
      1. explicit K_fit in config (if present),
      2. K_run from config,
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

    # Choose K_fit
    if "K_fit" in cfg:
        K_fit = float(cfg["K_fit"])
    elif "K_run" in cfg:
        K_fit = float(cfg["K_run"])
    elif "K1_star" in cfg and "K2_star" in cfg:
        K_fit = 0.5 * (float(cfg["K1_star"]) + float(cfg["K2_star"]))
    else:
        raise ValueError("Config must contain K_fit or K_run or both K1_star and K2_star.")

    K_run = float(cfg["K_run"]) if "K_run" in cfg else None
    return p, K_fit, K_run


# ------------------------------------------------------------
# Main CLI
# ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Fit K4 NS normal-form coefficients from config")

    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config file (e.g. k4_quat_baseline.yaml)",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for ns_fit.json and diagnostics",
    )
    ap.add_argument(
        "--bursts",
        type=int,
        default=64,
        help="Number of short bursts",
    )
    ap.add_argument(
        "--T_burst",
        type=int,
        default=50,
        help="Length of each burst (time steps)",
    )
    ap.add_argument(
        "--eps_min",
        type=float,
        default=1e-3,
        help="Minimum burst size for initial condition (log-uniform)",
    )
    ap.add_argument(
        "--eps_max",
        type=float,
        default=1e-1,
        help="Maximum burst size for initial condition (log-uniform)",
    )
    ap.add_argument(
        "--trim",
        type=int,
        default=5,
        help="Drop first 'trim' steps of each burst for fitting",
    )
    ap.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        help="Ridge regularization λ (if >0, overrides internal CV)",
    )
    ap.add_argument(
        "--amp_min",
        type=float,
        default=0.0,
        help="Minimum joint amplitude sqrt(|z1|^2 + |z2|^2) to keep a sample",
    )
    ap.add_argument(
        "--amp_max",
        type=float,
        default=1e9,
        help="Maximum joint amplitude sqrt(|z1|^2 + |z2|^2) to keep a sample",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=123,
        help="RNG seed for burst generation",
    )
    ap.add_argument(
        "--d_rho_dK1",
        type=float,
        default=1.8e-3,
        help="Slope d|λ1|/dK for post-NS μ1_post",
    )
    ap.add_argument(
        "--d_rho_dK2",
        type=float,
        default=-2.1e-3,
        help="Slope d|λ2|/dK for post-NS μ2_post",
    )
    ap.add_argument(
        "--deltaK_post",
        type=float,
        default=5e-7,
        help="Post-NS shift δK for μ_post = d_rho_dK_j * δK",
    )
    ap.add_argument(
        "--sim_nf",
        type=int,
        default=0,
        help="Number of steps for optional NF polar simulation (0 = skip)",
    )

    args = ap.parse_args()

    cfg = load_k4_config(args.config)
    p, K_fit, K_run_cfg = build_params_from_config(cfg)
    g = build_k4_spec(p)

    outdir = args.outdir or os.path.join(f"figs_k4_ns_fit_{now_tag()}")
    ensure_dir(outdir)

    print(f"[config] Using K_fit = {K_fit:.9f}")
    print(
        f"[config] topology={p.topology}, dt={p.dt}, gamma={p.gamma}, "
        f"w_diag={p.w_diag}, sigma2={p.sigma2}, triad_phi={p.triad_phi}"
    )

    # Fixed point and reduced Jacobian / NS eigenmodes at K_fit
    theta_star = get_fixed_pt(K_fit, p, g)

    J = jacobian_inertial(K_fit, p, g, theta=theta_star)
    Qtheta = gauge_free_basis_theta(g.n)  # (n, n-1)
    Q = block_diag(Qtheta, Qtheta)        # (2n, 2(n-1))
    Jr = Q.T @ J @ Q                      # reduced Jacobian

    v1, v2, w1, w2, lam1, lam2 = select_ns_modes_biorth(Jr)

    print(
        f"[lin] lam1 = {lam1.real:+.6f}{lam1.imag:+.6f}j, "
        f"lam2 = {lam2.real:+.6f}{lam2.imag:+.6f}j"
    )
    print(f"[lin] |lam1|={abs(lam1):.6f}, |lam2|={abs(lam2):.6f}")
    print(f"[lin] ω1={np.angle(lam1):+.6f} rad, ω2={np.angle(lam2):+.6f} rad")

    # Generate bursts and collect z1, z2 samples
    z1_all, z2_all = generate_bursts(
        K_fit,
        p,
        g,
        v1,
        v2,
        w1,
        w2,
        Q,
        n_bursts=args.bursts,
        T_burst=args.T_burst,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        trim=args.trim,
        rng_seed=args.seed,
        theta_star=theta_star,
    )

    n_raw = z1_all.size
    mean_abs_z1 = float(np.mean(np.abs(z1_all)))
    mean_abs_z2 = float(np.mean(np.abs(z2_all)))
    print(f"[data] Collected {n_raw} samples for fitting.")
    print(f"[data] mean|z1|={mean_abs_z1:.3e}, mean|z2|={mean_abs_z2:.3e}")

    # Optional amplitude filter in joint (z1,z2) amplitude
    amp_min = float(args.amp_min)
    amp_max = float(args.amp_max)

    if amp_min > 0.0 or amp_max < 1e8:
        joint_amp = np.sqrt(np.abs(z1_all) ** 2 + np.abs(z2_all) ** 2)
        mask = np.ones_like(joint_amp, dtype=bool)
        if amp_min > 0.0:
            mask &= joint_amp >= amp_min
        if amp_max < 1e8:
            mask &= joint_amp <= amp_max
        kept = int(np.count_nonzero(mask))
        if kept < 100:
            print(
                f"[warn] Amplitude filter (amp_min={amp_min:g}, amp_max={amp_max:g}) "
                f"left only {kept} samples; results may be unstable."
            )
        else:
            print(
                f"[filter] amp_min={amp_min:g}, amp_max={amp_max:g} "
                f"-> kept {kept} samples"
            )
        z1_all = z1_all[mask]
        z2_all = z2_all[mask]

    # Fit NS coefficients
    fit_res = fit_ns_coeffs_from_data(
        z1_all,
        z2_all,
        lam1,
        lam2,
        ridge=args.ridge,
        d_rho_dK1=args.d_rho_dK1,
        d_rho_dK2=args.d_rho_dK2,
        deltaK_post=args.deltaK_post,
    )

    # Save JSON
    out_path = os.path.join(outdir, "ns_fit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fit_res, f, indent=2)
    print(f"[save] NS fit results -> {out_path}")

    # Console summary of the most important numbers
    print("\n[summary] Real cubic coefficients a_jk = Re β_jk")
    print(f"  a11 = {fit_res['a11']:.6e}")
    print(f"  a12 = {fit_res['a12']:.6e}")
    print(f"  a21 = {fit_res['a21']:.6e}")
    print(f"  a22 = {fit_res['a22']:.6e}")
    print(f"  Δ   = {fit_res['Delta']:.6e}")
    print(
        f"  resid1_rel ≈ {fit_res['resid1_rel']:.3e}, "
        f"resid2_rel ≈ {fit_res['resid2_rel']:.3e}"
    )
    print(
        f"  μ1_fit = {fit_res['mu1_fit']:.3e}, "
        f"μ2_fit = {fit_res['mu2_fit']:.3e}"
    )
    print(
        f"  s1_fit = {fit_res['s1_fit']:.3e}, s2_fit = {fit_res['s2_fit']:.3e}, "
        f"A1_fit = {fit_res['A1_fit']:.3e}, A2_fit = {fit_res['A2_fit']:.3e}"
    )
    print(
        f"  μ1_post = {fit_res['mu1_post']:.3e}, "
        f"μ2_post = {fit_res['mu2_post']:.3e}"
    )
    print(
        f"  s1_post = {fit_res['s1_post']:.3e}, s2_post = {fit_res['s2_post']:.3e}, "
        f"A1_post = {fit_res['A1_post']:.3e}, A2_post = {fit_res['A2_post']:.3e}"
    )
    print(
        f"  s1_sym = {fit_res['s1_sym']:.3e}, s2_sym = {fit_res['s2_sym']:.3e}, "
        f"A1_sym = {fit_res['A1_sym']:.3e}, A2_sym = {fit_res['A2_sym']:.3e}, "
        f"ratio_sym ≈ {fit_res['ratio_sym']:.3e}"
    )

    # Heuristic warning if residuals are very large
    if (fit_res["resid1_rel"] > 0.7) or (fit_res["resid2_rel"] > 0.7):
        print(
            "\n[warn] Relative residuals are very large (>0.7). "
            "The cubic normal form explains only a small fraction of the variance.\n"
            "      This usually means the bursts stay too close to the linear regime,\n"
            "      or additional higher-order / non-normal effects are significant.\n"
            "      Consider re-running with larger eps_min/eps_max, longer T_burst,\n"
            "      and a modest ridge (e.g. --ridge 0.05), and/or tightening amp_min/amp_max."
        )

    # Optional NF simulation using post-bifurcation μ
    if args.sim_nf > 0:
        sim_nf_polar(
            fit_res["a11"],
            fit_res["a12"],
            fit_res["a21"],
            fit_res["a22"],
            fit_res["mu1_post"],
            fit_res["mu2_post"],
            fit_res["omega1"],
            fit_res["omega2"],
            T_sim=args.sim_nf,
            rho1_0=0.01,
            rho2_0=0.01,
            outdir=outdir,
        )
        print(f"[sim] NF amplitude sim saved to {os.path.join(outdir, 'nf_sim_amps.png')}")


if __name__ == "__main__":
    main()
