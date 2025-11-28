# rtg_core.py
# Core math, model, stability analysis, diagnostics (plots & Lyapunov).
# Numpy + Matplotlib only. ASCII console; UTF-8 files.

from __future__ import annotations
import os, math, time, json
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Utilities
# -------------------------------

def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def wrap_pi(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2*np.pi) - np.pi

def default_Delta() -> np.ndarray:
    return np.array([0.01, 0.0, -0.01], dtype=float)

def gauge_free_basis_theta(n: int) -> np.ndarray:
    """Orthonormal basis for the subspace orthogonal to the all-ones vector."""
    assert n >= 2
    e = np.eye(n)
    one = np.ones((n,1)) / np.sqrt(n)
    cols = []
    for k in range(n):
        v = e[:,k:k+1]
        v = v - one @ (one.T @ v)
        for c in cols:
            v = v - c @ (c.T @ v)
        nv = np.linalg.norm(v)
        if nv > 1e-12:
            cols.append(v / nv)
        if len(cols) == n-1:
            break
    return np.hstack(cols)  # (n, n-1)

def block_diag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    Z1 = np.zeros((A.shape[0], B.shape[1]))
    Z2 = np.zeros((B.shape[0], A.shape[1]))
    return np.block([[A, Z1],
                     [Z2, B]])

# -------------------------------
# Model specs
# -------------------------------

@dataclass
class GraphSpec:
    n: int = 3
    directed: bool = True
    eps_asym: float = 0.2
    F_triangle: float = 0.0
    kappa: np.ndarray = field(default_factory=lambda: np.zeros((0,0), dtype=float))
    alphas: np.ndarray = field(default_factory=lambda: np.zeros((0,0), dtype=float))
    deg_vec: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=float))
    deg_norm: bool = False

@dataclass
class Params:
    # core integrator
    family: str = "inertial"       # only inertial family implemented
    scheme: str = "explicit"       # "explicit" or "ec"
    dt: float = 0.1
    gamma: float = 0.3

    # graph options
    directed: bool = True
    eps_asym: float = 0.2
    deg_norm: bool = False
    Delta: np.ndarray = field(default_factory=default_Delta)

    # sweep
    K_min: float = 1.0
    K_max: float = 6.0
    K_pts: int = 80
    ang_tol: float = 0.10
    outdir: Optional[str] = None

    # post-NS diagnostics
    post_ns: bool = False
    deltaK: float = 0.03
    T_diag: int = 60000
    burn_diag: int = 30000
    noise: float = 0.0

    # lyapunov
    lyap_demo: bool = False
    le_q: int = 1

# -------------------------------
# K3 builder
# -------------------------------

def build_k3_spec(p: Params) -> GraphSpec:
    n = 3
    kappa = np.zeros((n,n), dtype=float)
    # Ring orientation 0->1->2->0
    for i in range(n):
        j = (i+1) % n
        kappa[i, j] = 1.0 + (p.eps_asym if p.directed else 0.0)
        kappa[j, i] = 1.0 - (p.eps_asym if p.directed else 0.0)
    np.fill_diagonal(kappa, 0.0)

    # Put all frustration on edge 2->0 so total oriented triangle flux = -2π/3
    alphas = np.zeros((n,n), dtype=float)
    alphas[2,0] = +2.0*np.pi/3.0  # makes sum -alpha_ij = -2π/3

    deg_vec = np.sum(np.abs(kappa), axis=1)
    if not p.deg_norm:
        deg_vec = np.ones_like(deg_vec)

    return GraphSpec(n=n, directed=p.directed, eps_asym=p.eps_asym,
                     F_triangle=-2.0*np.pi/3.0, kappa=kappa,
                     alphas=alphas, deg_vec=deg_vec, deg_norm=p.deg_norm)

# -------------------------------
# Linearization & Jacobians
# -------------------------------

def M_coupling_linear(theta: Optional[np.ndarray],
                      K: float, spec: GraphSpec) -> np.ndarray:
    """Linear coupling for small perturbations around theta (default 0)."""
    n = spec.n
    if theta is None:
        theta = np.zeros(n, dtype=float)
    C = np.zeros((n,n), dtype=float)
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i == j: 
                continue
            c = (spec.kappa[i,j] / spec.deg_vec[i]) * np.cos(theta[j] - theta[i] - spec.alphas[i,j])
            C[i,j] = K * c
            row_sum += K * c
        C[i,i] = -row_sum
    return C

def jacobian_inertial(K: float, p: Params, spec: GraphSpec,
                      theta: Optional[np.ndarray] = None) -> np.ndarray:
    """2n×2n Jacobian of the inertial map around theta (default 0)."""
    n = spec.n
    dt, gamma = p.dt, p.gamma
    A = 1.0 - gamma*dt
    M = M_coupling_linear(theta, K, spec)
    I = np.eye(n)

    if p.scheme.lower() in ("explicit", "e", "exp"):
        # theta_{t+1} = theta_t + dt * v_t
        # v_{t+1}     = A v_t + dt M theta_t
        return np.block([[I,       dt*I],
                         [dt*M,    A*I ]])
    elif p.scheme.lower() in ("ec", "euler_cromer", "euler-cromer"):
        # v_{t+1}     = A v_t + dt M theta_t
        # theta_{t+1} = theta_t + dt v_{t+1}
        return np.block([[I + (dt*dt)*M, dt*A*I],
                         [dt*M,          A*I   ]])
    else:
        raise ValueError(f"Unknown scheme: {p.scheme}")

# -------------------------------
# Gauge-free eigen analysis
# -------------------------------

def _leading_eigs_reduced(J: np.ndarray, n: int) -> np.ndarray:
    """Eigenvalues of J on the gauge-free subspace (drop the mean theta & mean v)."""
    Qtheta = gauge_free_basis_theta(n)   # (n, n-1)
    Q = block_diag(Qtheta, Qtheta)       # (2n, 2(n-1))
    Jr = Q.T @ J @ Q
    return np.linalg.eigvals(Jr)

def leading_eig(J: np.ndarray, n: int) -> Tuple[complex, float]:
    """Largest-modulus eigenvalue and its angle in [0, π]."""
    w = _leading_eigs_reduced(J, n)
    lam = w[np.argmax(np.abs(w))]
    ang = abs(np.angle(lam))
    if ang > np.pi:
        ang = 2*np.pi - ang
    return lam, ang

def classify_mode(lam: complex, ang: float, ang_tol: float) -> str:
    if abs(lam.imag) < 1e-9:
        return "real+" if lam.real >= 0 else "real-"
    near_0 = (ang < ang_tol)
    near_pi = (abs(ang - np.pi) < ang_tol)
    return "complex-near-real" if (near_0 or near_pi) else "complex"

# -------------------------------
# NS scan + refine
# -------------------------------

def scan_sweep(p: Params, spec: GraphSpec) -> Dict[str, np.ndarray | List[str]]:
    K_grid = np.linspace(p.K_min, p.K_max, p.K_pts)
    rho = np.zeros_like(K_grid)
    ang = np.zeros_like(K_grid)
    lbl: List[str] = []
    for i, K in enumerate(K_grid):
        J = jacobian_inertial(K, p, spec, theta=None)
        lam, a = leading_eig(J, spec.n)
        rho[i] = abs(lam); ang[i] = a
        lbl.append(classify_mode(lam, a, p.ang_tol))
    return {"K": K_grid, "rho": rho, "ang": ang, "label": lbl}

def find_ns_bracket(K: np.ndarray, rho: np.ndarray, ang: np.ndarray, lbl: List[str],
                    ang_tol: float) -> Optional[Tuple[float, float]]:
    for i in range(len(K)-1):
        ci = ("complex" in lbl[i]) and (ang[i] > ang_tol) and (abs(ang[i]-np.pi) > ang_tol)
        cj = ("complex" in lbl[i+1]) and (ang[i+1] > ang_tol) and (abs(ang[i+1]-np.pi) > ang_tol)
        if ci and cj:
            if (rho[i]-1.0) * (rho[i+1]-1.0) <= 0.0:
                return (K[i], K[i+1])
    return None

def refine_ns(K_lo: float, K_hi: float, p: Params, spec: GraphSpec,
              iters: int = 32) -> Tuple[float, float, float, str]:
    bestK, bestr, besta, bestlbl = None, None, None, ""
    for _ in range(iters):
        K_mid = 0.5*(K_lo + K_hi)
        J = jacobian_inertial(K_mid, p, spec, theta=None)
        lam, a = leading_eig(J, spec.n)
        r = abs(lam); lbl = classify_mode(lam, a, p.ang_tol)
        bestK, bestr, besta, bestlbl = K_mid, r, a, lbl
        if r >= 1.0: K_hi = K_mid
        else:        K_lo = K_mid
    return bestK, bestr, besta, bestlbl

def drho_dK_numeric(K: float, p: Params, spec: GraphSpec, h: float = 1e-3) -> float:
    J1 = jacobian_inertial(K - h, p, spec, theta=None)
    J2 = jacobian_inertial(K + h, p, spec, theta=None)
    lam1, _ = leading_eig(J1, spec.n)
    lam2, _ = leading_eig(J2, spec.n)
    return (abs(lam2) - abs(lam1)) / (2*h)

# -------------------------------
# Nonlinear stepper & diagnostics
# -------------------------------

def step_inertial(theta: np.ndarray, vel: np.ndarray,
                  K: float, p: Params, spec: GraphSpec,
                  noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """One step of the discrete inertial map (explicit or EC)."""
    n = spec.n
    f = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j: 
                continue
            s += spec.kappa[i,j] * math.sin(theta[j] - theta[i] - spec.alphas[i,j])
        s /= spec.deg_vec[i]
        f[i] = K * s

    if p.scheme.lower() in ("explicit", "e", "exp"):
        theta_new = theta + p.dt * vel
        vel_new   = (1.0 - p.gamma*p.dt) * vel + p.dt * f
    else:  # Euler-Cromer
        vel_new   = (1.0 - p.gamma*p.dt) * vel + p.dt * f
        theta_new = theta + p.dt * vel_new

    if noise > 0.0:
        theta_new = theta_new + np.random.normal(0.0, noise, size=theta.shape)

    return theta_new, vel_new

def triangle_flux(theta: np.ndarray, spec: GraphSpec) -> float:
    """Flux around (0,1,2) triangle; used mainly in K3 demos."""
    if spec.n < 3:
        return 0.0
    i, j, k = 0, 1, 2
    F = (theta[j]-theta[i] - spec.alphas[i,j]) + \
        (theta[k]-theta[j] - spec.alphas[j,k]) + \
        (theta[i]-theta[k] - spec.alphas[k,i])
    return wrap_pi(F)

def post_ns_demo(K_star: float, p: Params, spec: GraphSpec,
                 outdir: str, deltaK: float,
                 T: int, burn: int, noise: float) -> Dict[str, float]:
    n = spec.n
    theta = np.zeros(n, dtype=float)
    vel   = np.zeros(n, dtype=float)
    K_run = K_star + deltaK

    r_series, F_series, xy_series = [], [], []
    for t in range(T):
        theta, vel = step_inertial(theta, vel, K_run, p, spec, noise=noise)
        if t >= burn:
            z = np.exp(1j*theta).mean()
            r_series.append(abs(z))
            F_series.append(triangle_flux(theta, spec))
            xy_series.append([wrap_pi(theta[0]-theta[1]), wrap_pi(theta[1]-theta[2])])

    r_series = np.asarray(r_series) if r_series else np.array([])
    F_series = np.asarray(F_series) if F_series else np.array([])
    xy_series = np.asarray(xy_series) if xy_series else np.zeros((0,2))

    # r(t)
    if r_series.size > 0:
        plt.figure()
        plt.plot(r_series, lw=0.8)
        plt.title(f"r(t) at K={K_run:.6f}")
        plt.xlabel("time (steps)"); plt.ylabel("r")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "ns_confirm_series.png"), dpi=150)
        plt.close()

        # PSD of r(t)
        N = len(r_series)
        r0 = r_series - r_series.mean()
        R = np.fft.rfft(r0)
        freqs = np.fft.rfftfreq(N, d=1.0)
        PSD = (R.conj()*R).real / max(1, N)
        plt.figure()
        plt.plot(freqs, PSD, lw=0.8)
        plt.title(f"PSD of r(t) at K={K_run:.6f}")
        plt.xlabel("frequency (cycles/step)"); plt.ylabel("power")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "ns_confirm_psd.png"), dpi=150)
        plt.close()

        # Phase portrait (first two differences)
        if xy_series.shape[0] > 0:
            plt.figure(figsize=(5,5))
            plt.plot(xy_series[:,0], xy_series[:,1], lw=0.6)
            plt.xlabel("theta1 - theta2"); plt.ylabel("theta2 - theta3")
            plt.title(f"Phase portrait at K={K_run:.6f}")
            plt.tight_layout(); plt.savefig(os.path.join(outdir, "ns_confirm_phase.png"), dpi=150)
            plt.close()

        # flux series
        if F_series.size > 0:
            plt.figure()
            plt.plot(F_series, lw=0.8)
            plt.title("Triangle flux F (principal branch)")
            plt.xlabel("time (steps)"); plt.ylabel("F (rad)")
            plt.tight_layout(); plt.savefig(os.path.join(outdir, "flux_su2_series.png"), dpi=150)
            plt.close()

        k_peak = int(np.argmax(PSD[1:])) + 1 if PSD.size > 1 else 0
        f_meas = float(freqs[k_peak]) if PSD.size > 1 else 0.0
    else:
        f_meas = 0.0

    return {
        "K_run": float(K_run),
        "r_mean": float(r_series.mean() if r_series.size else 0.0),
        "r_std": float(r_series.std() if r_series.size else 0.0),
        "f_peak_per_step": f_meas
    }

# -------------------------------
# Lyapunov exponents
# -------------------------------

def largest_lyap(K: float, p: Params, spec: GraphSpec,
                 T: int = 30000, burn: int = 10000,
                 noise: float = 0.0) -> float:
    return lyap_qr(K, p, spec, T=T, burn=burn, q=1, noise=noise)[0]

def lyap_qr(K: float, p: Params, spec: GraphSpec,
            T: int = 30000, burn: int = 10000,
            q: int = 3, noise: float = 0.0) -> np.ndarray:
    """Top-q Lyapunov exponents (per step) via QR along the nonlinear trajectory."""
    n = spec.n
    theta = np.zeros(n, dtype=float)
    vel   = np.zeros(n, dtype=float)
    D = 2*n

    # random orthonormal frame Q (D x q)
    rng = np.random.default_rng(42)
    Q = rng.normal(size=(D, q))
    Q, _ = np.linalg.qr(Q, mode='reduced')

    sums = np.zeros(q, dtype=float)
    cnt = 0

    for t in range(T):
        # Jacobian at current state
        J = jacobian_inertial(K, p, spec, theta=theta)
        Z = J @ Q
        Q, R = np.linalg.qr(Z, mode='reduced')

        if t >= burn:
            diag = np.abs(np.diag(R))
            diag[diag == 0.0] = 1e-30
            sums += np.log(diag)
            cnt += 1

        theta, vel = step_inertial(theta, vel, K, p, spec, noise=noise)

    if cnt == 0:
        return np.zeros(q, dtype=float)
    return sums / cnt  # per step

# -------------------------------
# Plot helpers for sweep
# -------------------------------

def plot_angle_vs_K(K: np.ndarray, ang: np.ndarray, outdir: str, tag: str) -> None:
    plt.figure()
    plt.plot(K, ang, lw=1.0)
    plt.xlabel("K"); plt.ylabel("angle (rad)")
    plt.title("Leading eigen-angle vs K")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_angle_vs_K.png"), dpi=150)
    plt.close()

def plot_grid_overview(K: np.ndarray, rho: np.ndarray, ang: np.ndarray, lbl: List[str],
                       outdir: str, tag: str) -> None:
    plt.figure(figsize=(6,4))
    plt.plot(K, rho, lw=1.0, label="|lambda|max")
    plt.axhline(1.0, color="k", lw=0.8, ls="--")
    plt.xlabel("K"); plt.ylabel("spectral radius")
    plt.title("Grid overview (leading mode)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_grid_overview.png"), dpi=150)
    plt.close()
