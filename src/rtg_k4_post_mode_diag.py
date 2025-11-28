# rtg_k4_post_mode_diag.py
# Post-run diagnostics for K4: mode projections, frequency geometry,
# return map, and Poincaré section.

from __future__ import annotations
import os, argparse, json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List

from rtg_core import (
    Params, GraphSpec, ensure_dir, now_tag,
    jacobian_inertial, step_inertial,
    gauge_free_basis_theta, block_diag,
)

# -------------------------------
# K4 params and builders (same as scout)
# -------------------------------

@dataclass
class K4Params(Params):
    topology: str = "ringdiag"       # ringdiag | quat_sym
    w_diag: float = 0.15
    sigma2: float = 0.0
    triad_phi: float = 0.0

def build_k4_ringdiag(p: K4Params) -> GraphSpec:
    n = 4
    kappa = np.zeros((n, n), dtype=float)

    # Oriented ring 0->1->2->3->0
    for i in range(n):
        j = (i + 1) % n
        kappa[i, j] = 1.0 + (p.eps_asym if p.directed else 0.0)
        kappa[j, i] = 1.0 - (p.eps_asym if p.directed else 0.0)

    # diagonals (0<->2 and 1<->3)
    d = p.w_diag + p.sigma2
    for (a, b) in [(0, 2), (2, 0), (1, 3), (3, 1)]:
        kappa[a, b] = d

    np.fill_diagonal(kappa, 0.0)

    alphas = np.zeros((n, n), dtype=float)
    if abs(p.triad_phi) > 0:
        alphas[2, 0] = p.triad_phi
        alphas[3, 1] = p.triad_phi

    deg_vec = np.sum(np.abs(kappa), axis=1)
    if not p.deg_norm:
        deg_vec = np.ones_like(deg_vec)

    return GraphSpec(
        n=n, directed=p.directed, eps_asym=p.eps_asym,
        kappa=kappa, alphas=alphas, deg_vec=deg_vec, deg_norm=p.deg_norm
    )

def build_k4_quat_sym(p: K4Params) -> GraphSpec:
    n = 4
    kappa = np.ones((n, n), dtype=float)
    np.fill_diagonal(kappa, 0.0)

    ring_pairs = [(0,1),(1,2),(2,3),(3,0)]
    for (i,j) in ring_pairs:
        kappa[i,j] = 1.0 + (p.eps_asym if p.directed else 0.0)
        kappa[j,i] = 1.0 - (p.eps_asym if p.directed else 0.0)

    diag_w = p.w_diag + p.sigma2
    for (a,b) in [(0,2),(2,0),(1,3),(3,1)]:
        kappa[a,b] = diag_w

    alphas = np.zeros((n,n), dtype=float)
    if abs(p.triad_phi) > 0:
        alphas[2,0] = p.triad_phi
        alphas[3,1] = p.triad_phi

    deg_vec = np.sum(np.abs(kappa), axis=1)
    if not p.deg_norm:
        deg_vec = np.ones_like(deg_vec)

    return GraphSpec(
        n=n, directed=p.directed, eps_asym=p.eps_asym,
        kappa=kappa, alphas=alphas, deg_vec=deg_vec, deg_norm=p.deg_norm
    )

def build_k4_spec(p: K4Params) -> GraphSpec:
    topo = p.topology.lower()
    if topo == "ringdiag":
        return build_k4_ringdiag(p)
    elif topo in ("quat_sym","quat","tetra"):
        return build_k4_quat_sym(p)
    else:
        raise ValueError(f"Unknown topology: {p.topology}")


# -------------------------------
# Eigenmode helper
# -------------------------------

def eig_pairs_with_vecs(J: np.ndarray, n: int):
    """
    Gauge-free eigen decomposition of J.
    Returns (vals, vecs, Q), where:
      vals: list of (n-1) complex eigenvalues (imag >= 0), sorted by |.| descending
      vecs: corresponding eigenvectors in reduced space (2*(n-1)-dim)
      Q   : block-diagonal gauge-free basis (2n x 2*(n-1))
    """
    Qtheta = gauge_free_basis_theta(n)
    Q = block_diag(Qtheta, Qtheta)  # (2n, 2*(n-1))
    Jr = Q.T @ J @ Q
    w, V = np.linalg.eig(Jr)

    keep_idx = [i for i, lam in enumerate(w) if lam.imag >= -1e-12]
    keep_idx.sort(key=lambda i: -abs(w[i]))

    vals = [w[i] for i in keep_idx[:(n-1)]]
    vecs = [V[:, i] for i in keep_idx[:(n-1)]]
    return vals, vecs, Q


# -------------------------------
# Main diagnostic routine
# -------------------------------

def run_mode_diag(K_run: float, p: K4Params, g: GraphSpec,
                  outdir: str, T: int, burn: int, noise: float) -> Dict:
    n = g.n
    ensure_dir(outdir)

    print(f"Mode diagnostics at K_run={K_run:.9f}, T={T}, burn={burn}, noise={noise}")

    # Initial condition: random phases, zero velocities
    rng = np.random.default_rng(12345)
    theta0 = 2*np.pi * rng.random(n)
    omega0 = np.zeros(n, dtype=float)
    x = np.concatenate([theta0, omega0])

    # Allocate storage
    x_series = np.zeros((T, 2*n), dtype=float)
    theta_series = np.zeros((T, n), dtype=float)
    r_series = np.zeros(T, dtype=float)

    # Integrate
    for t in range(T):
        theta = x[:n].copy()
        theta_series[t, :] = theta
        r_series[t] = np.abs(np.mean(np.exp(1j * theta)))
        x_series[t, :] = x
        # NOTE: if your step_inertial signature differs, adapt here.
        x = step_inertial(K_run, p, g, x, noise)

    if burn >= T:
        raise ValueError("burn must be < T")

    theta_eff = theta_series[burn:, :]
    x_eff = x_series[burn:, :]
    r_eff = r_series[burn:]
    T_eff = theta_eff.shape[0]

    # ---------------------------
    # Mode projections via eigenvectors
    # ---------------------------
    J = jacobian_inertial(K_run, p, g, theta=None)
    vals, vecs, Q = eig_pairs_with_vecs(J, n)
    lam1, lam2 = vals[0], vals[1]
    v1, v2 = vecs[0], vecs[1]

    # Project trajectory to gauge-free subspace, centered at 0
    x_red = Q.T @ x_eff.T  # shape (2*(n-1), T_eff)

    z1 = v1.conj().T @ x_red  # shape (T_eff,)
    z2 = v2.conj().T @ x_red
    A1, A2 = np.abs(z1), np.abs(z2)
    phi1, phi2 = np.angle(z1), np.angle(z2)

    # Amplitude plot
    plt.figure()
    plt.plot(np.arange(T_eff), A1, label="A1 (mode 1)")
    plt.plot(np.arange(T_eff), A2, label="A2 (mode 2)")
    plt.xlabel("step (post-burn)")
    plt.ylabel("amplitude")
    plt.title("Mode amplitudes A1, A2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mode_amps.png"), dpi=150)
    plt.close()

    # PSDs of unwrapped phases phi1, phi2
    def phase_psd(phi, name):
        ph = np.unwrap(phi)
        ph -= np.mean(ph)
        F = np.fft.rfft(ph)
        psd = np.abs(F)**2
        freqs = np.fft.rfftfreq(ph.size)  # cycles per step

        plt.figure()
        plt.semilogy(freqs, psd + 1e-20)
        plt.xlabel("frequency (cycles/step)")
        plt.ylabel("PSD")
        plt.title(f"Mode phase PSD: {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"psd_{name}.png"), dpi=150)
        plt.close()
        return freqs, psd

    freqs_phi1, psd_phi1 = phase_psd(phi1, "phi1")
    freqs_phi2, psd_phi2 = phase_psd(phi2, "phi2")

    # ---------------------------
    # Instantaneous frequencies & ratios
    # ---------------------------
    # Differences in theta (post-burn) → ω_i(t)
    omega = np.diff(theta_eff, axis=0)  # Δθ per step
    omega_eff = omega  # already post-burn

    psd_omega_peaks = []
    for i in range(n):
        sig = omega_eff[:, i] - np.mean(omega_eff[:, i])
        F = np.fft.rfft(sig)
        psd = np.abs(F)**2
        freqs = np.fft.rfftfreq(sig.size)  # cycles per step

        plt.figure()
        plt.semilogy(freqs, psd + 1e-20)
        plt.xlabel("frequency (cycles/step)")
        plt.ylabel("PSD")
        plt.title(f"PSD of omega_{i}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"psd_omega{i}.png"), dpi=150)
        plt.close()

        if psd.size > 1:
            k_peak = 1 + np.argmax(psd[1:])
            psd_omega_peaks.append((i, freqs[k_peak]))
        else:
            psd_omega_peaks.append((i, 0.0))

    # Frequency ratio histogram (ω0/ω1)
    if n >= 2:
        denom = omega_eff[:, 1] + 1e-12
        ratios = omega_eff[:, 0] / denom
        winding_rho = float(np.mean(ratios))

        plt.figure()
        plt.hist(ratios, bins=60, density=True)
        plt.xlabel("omega_0 / omega_1")
        plt.ylabel("density")
        plt.title("Frequency ratio histogram (winding estimate)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "freq_ratio_hist_0_1.png"), dpi=150)
        plt.close()
    else:
        ratios = np.array([])
        winding_rho = float("nan")

    # ---------------------------
    # Return map of r(t)
    # ---------------------------
    sig_r = r_eff - np.mean(r_eff)
    F_r = np.fft.rfft(sig_r)
    psd_r = np.abs(F_r)**2
    freqs_r = np.fft.rfftfreq(sig_r.size)  # cycles per step

    if psd_r.size > 1:
        k_peak_r = 1 + np.argmax(psd_r[1:])
        f_peak_per_step = float(freqs_r[k_peak_r])
        # Approximate one-period lag in steps:
        lag = max(1, int(round(1.0 / f_peak_per_step)))
    else:
        f_peak_per_step = 0.0
        lag = 1

    if r_eff.size > lag:
        R0 = r_eff[:-lag]
        R1 = r_eff[lag:]
        plt.figure()
        plt.scatter(R0, R1, s=1, alpha=0.5)
        plt.xlabel("r(t)")
        plt.ylabel(f"r(t+{lag})")
        plt.title("Return map of r(t)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "return_map_r.png"), dpi=150)
        plt.close()
    else:
        R0 = R1 = np.array([])

    # ---------------------------
    # Poincaré section: θ0 - θ1 = 0 (up-crossings)
    # ---------------------------
    if n >= 3:
        # Use the same theta_eff (post-burn)
        phi01 = np.unwrap(theta_eff[:, 0] - theta_eff[:, 1])
        target = 0.0
        cross_up = np.where((phi01[:-1] < target) & (phi01[1:] >= target))[0]

        if cross_up.size > 0:
            # Two section variables: Δθ12 and Δθ23 at crossings
            sec_x = (theta_eff[cross_up, 1] - theta_eff[cross_up, 2]) % (2*np.pi)
            sec_y = (theta_eff[cross_up, 2] - theta_eff[cross_up, 3]) % (2*np.pi)

            plt.figure()
            plt.scatter(sec_x, sec_y, s=3, alpha=0.6)
            plt.xlabel("θ1 - θ2 (mod 2π)")
            plt.ylabel("θ2 - θ3 (mod 2π)")
            plt.title("Poincaré section (θ0 - θ1 = 0 up-crossings)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "poincare_section.png"), dpi=150)
            plt.close()
        else:
            sec_x = sec_y = np.array([])
    else:
        cross_up = np.array([], dtype=int)
        sec_x = sec_y = np.array([])

    # ---------------------------
    # Collect diagnostics
    # ---------------------------
    diag: Dict = {
        "K_run": float(K_run),
        "T": int(T),
        "burn": int(burn),
        "r_mean": float(r_eff.mean()),
        "r_std": float(r_eff.std()),
        "f_peak_per_step": float(f_peak_per_step),
        "lambda1": complex(vals[0]),
        "lambda2": complex(vals[1]),
        "A1_mean": float(A1.mean()),
        "A2_mean": float(A2.mean()),
        "winding_rho": float(winding_rho),
        "poincare_samples": int(cross_up.size),
        "psd_omega_peaks": [
            {"osc_idx": int(i), "f_peak_per_step": float(f)} for (i, f) in psd_omega_peaks
        ],
    }

    with open(os.path.join(outdir, "k4_mode_diag.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print("Mode diagnostics written to", outdir)
    return diag


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="K4 post-run mode diagnostics")
    ap.add_argument("--topology", type=str, default="quat_sym",
                    choices=["ringdiag", "quat_sym"])
    ap.add_argument("--scheme", type=str, default="explicit",
                    choices=["explicit", "ec"])
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--eps_asym", type=float, default=0.25)
    ap.add_argument("--deg_norm", action="store_true")

    ap.add_argument("--w_diag", type=float, default=0.15)
    ap.add_argument("--sigma2", type=float, default=0.0)
    ap.add_argument("--triad_phi", type=float, default=0.0)

    ap.add_argument("--K_run", type=float, required=True,
                    help="Coupling K at which to run diagnostics "
                         "(e.g. K_run from k4_post_diag.json)")
    ap.add_argument("--T", type=int, default=300000)
    ap.add_argument("--burn", type=int, default=150000)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--outdir", type=str, default=None)

    args = ap.parse_args()

    p = K4Params(
        scheme=args.scheme,
        dt=args.dt,
        gamma=args.gamma,
        directed=args.directed,
        eps_asym=args.eps_asym,
        deg_norm=args.deg_norm,
        topology=args.topology,
        w_diag=args.w_diag,
        sigma2=args.sigma2,
        triad_phi=args.triad_phi,
        # The remaining Params fields are unused by this script,
        # but we set sensible defaults:
        K_min=0.0,
        K_max=0.0,
        K_pts=1,
        ang_tol=0.0,
        post_ns=False,
        deltaK=0.0,
        T_diag=args.T,
        burn_diag=args.burn,
        noise=args.noise,
        lyap_demo=False,
        le_q=0,
    )

    g = build_k4_spec(p)
    outdir = args.outdir or os.path.join(
        f"figs_k4_mode_diag_{args.scheme}_{now_tag()}"
    )
    run_mode_diag(args.K_run, p, g, outdir, args.T, args.burn, args.noise)


if __name__ == "__main__":
    main()
