# rtg_k4_quat_scout_v5.py
# K4 scout: scan all complex pairs, detect per-pair NS, test for double-NS
# proximity, optional post-NS demo & Lyapunov, plus extra observables
# (mode projections, frequency geometry, Poincaré section).
#
# Compatible with the rtg_core.py you pasted.

from __future__ import annotations
import os, argparse, json, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from rtg_core import (
    Params, GraphSpec, ensure_dir, now_tag, wrap_pi,
    jacobian_inertial, step_inertial, triangle_flux,
    gauge_free_basis_theta, block_diag, lyap_qr, post_ns_demo
)

# ============================================================
# K4-specific params
# ============================================================

@dataclass
class K4Params(Params):
    # graph / topology
    topology: str = "ringdiag"       # "ringdiag" | "quat_sym"
    w_diag: float = 0.15             # diagonal weight (0<->2, 1<->3)
    sigma2: float = 0.0              # secondary scaling for diagonals
    triad_phi: float = 0.0           # frustration on selected edges
    near2: float = 0.05              # max |ΔK| to call a double-NS

    # extra diagnostics
    mode_proj: bool = False          # project trajectory onto NS eigenmodes
    freq_geo: bool = False           # ω-PSDs, freq ratios, return map
    poincare_sec: bool = False       # Poincaré section on θ0-θ1 = 0


# ============================================================
# K4 builders
# ============================================================

def build_k4_ringdiag(p: K4Params) -> GraphSpec:
    """
    K4 as ring + diagonals:
      - ring 0-1-2-3-0 with asymmetry eps_asym
      - diagonals 0<->2, 1<->3 with weight w_diag + sigma2
    """
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
        # Put phi on two closing edges to seed frustration in two triangles
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
    """
    Symmetric K4 suitable for quaternion rung:
      - complete graph
      - ring 0-1-2-3-0 distinguished from diagonals
      - diagonals with separate weight w_diag + sigma2
    """
    n = 4
    kappa = np.ones((n, n), dtype=float)
    np.fill_diagonal(kappa, 0.0)

    # distinguish nearest neighbors (ring) from diagonals
    ring_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for (i, j) in ring_pairs:
        kappa[i, j] = 1.0 + (p.eps_asym if p.directed else 0.0)
        kappa[j, i] = 1.0 - (p.eps_asym if p.directed else 0.0)

    # diagonals
    diag_w = p.w_diag + p.sigma2
    for (a, b) in [(0, 2), (2, 0), (1, 3), (3, 1)]:
        kappa[a, b] = diag_w

    alphas = np.zeros((n, n), dtype=float)
    if abs(p.triad_phi) > 0:
        # symmetric frustration on two opposite closings
        alphas[2, 0] = p.triad_phi
        alphas[3, 1] = p.triad_phi

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
    elif topo in ("quat_sym", "quat"):
        return build_k4_quat_sym(p)
    else:
        raise ValueError(f"Unknown topology: {p.topology}")


# ============================================================
# Pair-by-pair eigen tracking (gauge-free)
# ============================================================

def eig_pairs(J: np.ndarray, n: int) -> List[complex]:
    """
    Return positive-imaginary eigenvalues on the gauge-free subspace,
    sorted by |lambda| (descending). One representative per complex pair.
    """
    Qtheta = gauge_free_basis_theta(n)
    Q = block_diag(Qtheta, Qtheta)
    Jr = Q.T @ J @ Q
    w = np.linalg.eigvals(Jr)
    keep = [lam for lam in w if lam.imag >= -1e-12]
    keep.sort(key=lambda z: -abs(z))
    # there are (n-1) complex pairs
    return keep[:(n - 1)]


def scan_k4(p: K4Params, g: GraphSpec) -> Dict[str, np.ndarray]:
    K = np.linspace(p.K_min, p.K_max, p.K_pts)
    m = g.n - 1  # number of complex pairs
    rhos = np.zeros((m, K.size))
    angs = np.zeros((m, K.size))
    for i, k in enumerate(K):
        J = jacobian_inertial(k, p, g, theta=None)
        pairs = eig_pairs(J, g.n)
        for j in range(len(pairs)):
            lam = pairs[j]
            rhos[j, i] = abs(lam)
            ang = abs(np.angle(lam))
            ang = min(ang, 2 * np.pi - ang)
            angs[j, i] = ang
    return {"K": K, "rhos": rhos, "angs": angs}


def find_pair_bracket(
    K: np.ndarray, rho: np.ndarray, ang: np.ndarray, ang_tol: float
) -> Optional[Tuple[float, float]]:
    """
    Find K interval where rho crosses 1 with angle away from 0 or pi.
    """
    for i in range(K.size - 1):
        ok_i = (ang[i] > ang_tol) and (abs(ang[i] - np.pi) > ang_tol)
        ok_j = (ang[i + 1] > ang_tol) and (abs(ang[i + 1] - np.pi) > ang_tol)
        if ok_i and ok_j:
            if (rho[i] - 1.0) * (rho[i + 1] - 1.0) <= 0.0:
                return (K[i], K[i + 1])
    return None


def refine_pair(
    K_lo: float, K_hi: float, pair_idx: int, p: K4Params, g: GraphSpec,
    iters: int = 30
) -> Tuple[float, float, float]:
    """Bisection refinement to |lambda| ~ 1 for selected pair."""
    bestK, bestr, besta = None, None, None
    for _ in range(iters):
        km = 0.5 * (K_lo + K_hi)
        J = jacobian_inertial(km, p, g, theta=None)
        lam = eig_pairs(J, g.n)[pair_idx]
        r = abs(lam)
        a = abs(np.angle(lam))
        a = min(a, 2 * np.pi - a)
        bestK, bestr, besta = km, r, a
        if r >= 1.0:
            K_hi = km
        else:
            K_lo = km
    return bestK, bestr, besta


# ============================================================
# Extra observables: simulate trajectory & compute diagnostics
# ============================================================

def simulate_traj(
    K_run: float,
    p: K4Params,
    g: GraphSpec,
    T: int,
    burn: int,
    noise: float = 0.0,
    rng_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the inertial map at K_run for T steps, discarding 'burn',
    and return:
      theta_traj: (T_eff, n)
      omega_traj: (T_eff-1, n)  approximate instantaneous frequencies
      r_series  : (T_eff,)
      vel_traj  : (T_eff, n)
    """
    n = g.n
    theta = np.zeros(n, dtype=float)
    vel   = np.zeros(n, dtype=float)

    if rng_seed is not None:
        np.random.seed(rng_seed)

    T_eff = max(0, T - burn)
    theta_traj = np.zeros((T_eff, n), dtype=float)
    vel_traj   = np.zeros((T_eff, n), dtype=float)
    r_series   = np.zeros(T_eff, dtype=float)

    idx = 0
    for t in range(T):
        theta, vel = step_inertial(theta, vel, K_run, p, g, noise=noise)
        if t >= burn:
            if idx >= T_eff:
                break
            theta_traj[idx, :] = theta
            vel_traj[idx, :] = vel
            z = np.exp(1j * theta).mean()
            r_series[idx] = abs(z)
            idx += 1

    if T_eff >= 2:
        theta_unwrap = np.unwrap(theta_traj, axis=0)
        omega_traj = np.diff(theta_unwrap, axis=0) / p.dt
    else:
        omega_traj = np.zeros((max(T_eff - 1, 0), n), dtype=float)

    return theta_traj, omega_traj, r_series, vel_traj


def extra_observables(
    K_run: float,
    p: K4Params,
    g: GraphSpec,
    outdir: str,
    diag: Dict[str, float],
    rng_seed: int = 0,
) -> Dict[str, float]:
    """
    Compute additional observables on top of post_ns_demo:
      - mode projections (--mode_proj)
      - frequency geometry (--freq_geo)
      - Poincaré section (--poincare_sec)
    Results are merged into 'diag' and returned.
    """
    if not (p.mode_proj or p.freq_geo or p.poincare_sec):
        return diag

    T = p.T_diag
    burn = p.burn_diag
    noise = p.noise

    theta_traj, omega_traj, r_series, vel_traj = simulate_traj(
        K_run, p, g, T, burn, noise=noise, rng_seed=rng_seed
    )
    n = g.n
    T_eff = theta_traj.shape[0]
    if T_eff == 0:
        print("[extra] trajectory too short; skipping extra observables.")
        return diag

    # ---------------------------
    # 1) Mode projections
    # ---------------------------
    if p.mode_proj:
        J = jacobian_inertial(K_run, p, g, theta=None)
        Qtheta = gauge_free_basis_theta(n)
        Q = block_diag(Qtheta, Qtheta)
        Jr = Q.T @ J @ Q
        eigvals, eigvecs = np.linalg.eig(Jr)

        # choose two dominant complex eigenmodes
        idx_complex = [i for i, lam in enumerate(eigvals) if lam.imag > 0]
        if len(idx_complex) >= 2:
            idx_complex.sort(key=lambda i: abs(eigvals[i]), reverse=True)
            i1, i2 = idx_complex[0], idx_complex[1]
        elif len(idx_complex) == 1:
            i1 = idx_complex[0]
            order = np.argsort(np.abs(eigvals))[::-1]
            i2 = order[1]
        else:
            order = np.argsort(np.abs(eigvals))[::-1]
            i1, i2 = order[0], order[1]

        v1 = eigvecs[:, i1]
        v2 = eigvecs[:, i2]

        # full state trajectory: [theta, vel]
        x_traj = np.concatenate([theta_traj, vel_traj], axis=1)  # (T_eff, 2n)
        x_red = Q.T @ x_traj.T                                   # (2n-2, T_eff)

        z1 = v1.conj().T @ x_red
        z2 = v2.conj().T @ x_red
        z1 = np.asarray(z1).ravel()
        z2 = np.asarray(z2).ravel()

        A1 = np.abs(z1)
        A2 = np.abs(z2)
        phi1 = np.angle(z1)
        phi2 = np.angle(z2)

        # amplitudes
        plt.figure()
        plt.plot(A1, lw=0.6, label="mode1")
        plt.plot(A2, lw=0.6, label="mode2")
        plt.xlabel("time (steps after burn)")
        plt.ylabel("amplitude")
        plt.title("Mode amplitudes on NS torus")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "mode_amps.png"), dpi=150)
        plt.close()

        # helper: phase PSD
        def _phase_psd(phi: np.ndarray, tag: str) -> float:
            if phi.size < 8:
                return 0.0
            phi_un = np.unwrap(phi)
            N = phi_un.size
            P = np.fft.rfft(phi_un - phi_un.mean())
            freqs = np.fft.rfftfreq(N, d=1.0)
            PSD = (P.conj() * P).real / max(1, N)
            if PSD.size <= 1:
                f_peak = 0.0
            else:
                k_peak = int(np.argmax(PSD[1:])) + 1
                f_peak = float(freqs[k_peak])
            plt.figure()
            plt.plot(freqs, PSD, lw=0.8)
            plt.xlabel("frequency (cycles/step)")
            plt.ylabel("power")
            plt.title(f"PSD of {tag}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"psd_{tag}.png"), dpi=150)
            plt.close()
            return f_peak

        f_phi1 = _phase_psd(phi1, "phi1")
        f_phi2 = _phase_psd(phi2, "phi2")

        diag["mode_A1_mean"] = float(A1.mean())
        diag["mode_A2_mean"] = float(A2.mean())
        diag["mode_A1_std"] = float(A1.std())
        diag["mode_A2_std"] = float(A2.std())
        diag["mode_f_phi1"] = f_phi1
        diag["mode_f_phi2"] = f_phi2

    # ---------------------------
    # 2) Frequency geometry
    # ---------------------------
    if p.freq_geo:
        if omega_traj.shape[0] >= 8:
            Nw = omega_traj.shape[0]
            peak_freqs: List[float] = []
            for i in range(n):
                w_i = omega_traj[:, i]
                W = np.fft.rfft(w_i - w_i.mean())
                freqs_w = np.fft.rfftfreq(Nw, d=1.0)
                PSD_w = (W.conj() * W).real / max(1, Nw)

                if PSD_w.size <= 1:
                    f_peak = 0.0
                else:
                    k_peak = int(np.argmax(PSD_w[1:])) + 1
                    f_peak = float(freqs_w[k_peak])
                peak_freqs.append(f_peak)

                plt.figure()
                plt.plot(freqs_w, PSD_w, lw=0.8)
                plt.xlabel("frequency (cycles/step)")
                plt.ylabel("power")
                plt.title(f"PSD of omega_{i}")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, f"psd_omega{i}.png"), dpi=150)
                plt.close()

            diag["omega_peak_freqs"] = [float(f) for f in peak_freqs]

            # instantaneous frequency ratio histogram (omega0 / omega1)
            if n >= 2:
                denom = omega_traj[:, 1]
                mask = np.abs(denom) > 1e-8
                ratios = omega_traj[mask, 0] / denom[mask]
                if ratios.size > 0:
                    rho = float(np.mean(ratios))
                    diag["winding_rho_inst"] = rho
                    plt.figure()
                    plt.hist(ratios, bins=50, density=True)
                    plt.xlabel("omega0 / omega1")
                    plt.ylabel("density")
                    plt.title("Instantaneous freq ratio histogram")
                    plt.tight_layout()
                    plt.savefig(os.path.join(outdir, "freq_ratio_hist.png"), dpi=150)
                    plt.close()

        # Return map for r(t)
        if r_series.size >= 8 and diag.get("f_peak_per_step", 0.0) > 0:
            lag = int(round(1.0 / diag["f_peak_per_step"]))
            lag = max(1, min(lag, r_series.size - 1))
            R0 = r_series[:-lag]
            R1 = r_series[lag:]
            plt.figure()
            plt.scatter(R0, R1, s=1, alpha=0.5)
            plt.xlabel("r(t)")
            plt.ylabel(f"r(t+{lag})")
            plt.title("Return map of r(t)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "return_map_r.png"), dpi=150)
            plt.close()

    # ---------------------------
    # 3) Poincaré section
    # ---------------------------
    if p.poincare_sec:
        phi12 = wrap_pi(theta_traj[:, 0] - theta_traj[:, 1])
        cross_idx = np.where((phi12[:-1] < 0.0) & (phi12[1:] >= 0.0))[0]
        num = int(cross_idx.size)
        diag["poincare_samples"] = num
        if num >= 10:
            if n >= 3:
                phi23 = wrap_pi(theta_traj[:, 1] - theta_traj[:, 2])
            else:
                phi23 = phi12.copy()
            if n >= 4:
                phi03 = wrap_pi(theta_traj[:, 0] - theta_traj[:, 3])
            else:
                phi03 = r_series

            x_sec = phi23[cross_idx]
            y_sec = phi03[cross_idx]
            plt.figure(figsize=(5, 5))
            plt.scatter(x_sec, y_sec, s=2, alpha=0.5)
            plt.xlabel("theta2 - theta3 (mod 2π)")
            plt.ylabel("theta0 - theta3 (mod 2π)")
            plt.title("Poincaré section (theta0-theta1 = 0 crossings)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "poincare_section.png"), dpi=150)
            plt.close()

    return diag


# ============================================================
# Core runner for a single sigma2
# ============================================================

def run_single_sigma2(p: K4Params, outdir: str) -> None:
    g = build_k4_spec(p)
    ensure_dir(outdir)

    print("Explicit: coupling + triads can push complex pairs to |lambda|=1, enabling multi-pair NS.")
    print(
        f"Target K4 | scheme={p.scheme} | dt={p.dt} | gamma={p.gamma} | "
        f"directed={p.directed} eps={p.eps_asym} | deg_norm={p.deg_norm}"
    )
    print(
        f"K range [{p.K_min}, {p.K_max}] with {p.K_pts} points; "
        f"ang_tol={p.ang_tol:.3f}, near2={p.near2:.3f}, "
        f"sigma2={p.sigma2:.3f}, phi={p.triad_phi:.3f}\n"
    )

    sweep = scan_k4(p, g)
    K = sweep["K"]
    rhos = sweep["rhos"]
    angs = sweep["angs"]
    m = g.n - 1

    # Save sweep CSV
    csv_path = os.path.join(outdir, "k4_sweep.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ["K"]
        for j in range(m):
            header += [f"rho{j+1}", f"ang{j+1}"]
        f.write(",".join(header) + "\n")
        for i in range(K.size):
            row = [f"{K[i]:.9f}"]
            for j in range(m):
                row.append(f"{rhos[j, i]:.9f}")
                row.append(f"{angs[j, i]:.9f}")
            f.write(",".join(row) + "\n")
    print(f"[save] sweep -> {csv_path}")

    # Per-pair brackets
    brackets: Dict[int, Tuple[float, float]] = {}
    for j in range(m):
        br = find_pair_bracket(K, rhos[j], angs[j], p.ang_tol)
        if br is not None:
            brackets[j] = br

    if not brackets:
        print("No single-pair crossing detected on this grid.")
        return

    # Refine per-pair K* and report
    Kstars: Dict[int, Tuple[float, float, float]] = {}
    for j, (klo, khi) in brackets.items():
        kstar, rstar, astar = refine_pair(klo, khi, j, p, g, iters=32)
        Kstars[j] = (kstar, rstar, astar)
        print(f"pair{j+1}: K*={kstar:.9f} | rho=1.000000 | ang={astar:+.6f}")

    # Check double-NS proximity
    keys = sorted(Kstars.keys())
    best_dK: Optional[float] = None
    best_pair: Optional[Tuple[int, int]] = None
    for ia in range(len(keys)):
        for ib in range(ia + 1, len(keys)):
            j1, j2 = keys[ia], keys[ib]
            dK = abs(Kstars[j1][0] - Kstars[j2][0])
            if (best_dK is None) or (dK < best_dK):
                best_dK = dK
                best_pair = (j1, j2)

    if best_pair is not None and best_dK is not None:
        j1, j2 = best_pair
        kA, rA, aA = Kstars[j1]
        kB, rB, aB = Kstars[j2]
        if best_dK <= p.near2:
            print(
                f"Double-NS candidate: pair{j1+1}@K≈{kA:.6f} and "
                f"pair{j2+1}@K≈{kB:.6f} (|ΔK|={best_dK:.6f} ≤ near2)"
            )
        else:
            print(
                f"Two per-pair crossings found, but |ΔK| = {best_dK:.6f} > near2."
            )
        winding_lin = float(aA / aB) if aB != 0 else float("nan")
    else:
        winding_lin = float("nan")
        if len(Kstars) >= 2:
            j1, j2 = keys[0], keys[1]
            print(
                f"Two per-pair crossings found, but |ΔK| = "
                f"{abs(Kstars[j1][0] - Kstars[j2][0]):.6f} > near2."
            )
        else:
            print("Single-pair NS found.")

    # Build K* summary payload
    kstars_payload: Dict[str, object] = {"pairs": {}}
    for j, (kstar, rstar, astar) in Kstars.items():
        name = f"pair{j+1}"
        freq = astar / (2.0 * math.pi)
        kstars_payload["pairs"][name] = {
            "K_star": float(kstar),
            "rho": float(rstar),
            "angle_rad": float(astar),
            "freq_cycles_per_step": float(freq),
        }

    if best_pair is not None and best_dK is not None:
        j1, j2 = best_pair
        kstars_payload["closest_pair"] = {
            "pairA": f"pair{j1+1}",
            "pairB": f"pair{j2+1}",
            "deltaK": float(best_dK),
            "winding_ratio_lin": winding_lin,
        }

    # Save K* summary JSON
    kstars_path = os.path.join(outdir, "k4_kstars.json")
    with open(kstars_path, "w", encoding="utf-8") as f:
        json.dump(kstars_payload, f, indent=2)
    print(f"[save] K* summary -> {kstars_path}")

    # Optional post run, extra diagnostics, and Lyapunov
    diag: Optional[Dict[str, float]] = None
    if p.post_ns:
        k_pick = max(v[0] for v in Kstars.values())
        diag = post_ns_demo(
            k_pick, p, g, outdir, p.deltaK, p.T_diag, p.burn_diag, p.noise
        )
        print(
            f"[post] run at K={diag['K_run']:.6f} | "
            f"r_mean={diag['r_mean']:.6f} | r_std={diag['r_std']:.3e} | "
            f"f_peak_per_step={diag['f_peak_per_step']:.6f}"
        )

        # attach K* info
        diag["kstars_path"] = kstars_path
        diag["Kstars"] = kstars_payload
        diag["winding_lin"] = winding_lin

        # extra observables
        if p.mode_proj or p.freq_geo or p.poincare_sec:
            print("[extra] running additional observables ...")
            diag = extra_observables(
                diag["K_run"], p, g, outdir, diag, rng_seed=0
            )

        with open(
            os.path.join(outdir, "k4_post_diag.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(diag, f, indent=2)

    if p.lyap_demo:
        if diag is not None:
            K_lyap = diag["K_run"]
        else:
            K_lyap = max(v[0] for v in Kstars.values()) + p.deltaK
        les = lyap_qr(
            K_lyap,
            p,
            g,
            T=max(20000, p.T_diag // 2),
            burn=max(5000, p.burn_diag // 2),
            q=max(1, p.le_q),
            noise=p.noise,
        )
        with open(
            os.path.join(outdir, "lyap_summary.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("Largest Lyapunov exponents (per step):\n")
            for i, le in enumerate(les):
                f.write(f"  LE_{i+1} ~= {le:.6e}\n")
        print("Lyapunov summary written.")


# ============================================================
# Main CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="K4 double-NS scout")

    ap.add_argument("--topology", type=str, default="ringdiag",
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

    ap.add_argument("--K_min", type=float, default=0.5)
    ap.add_argument("--K_max", type=float, default=8.0)
    ap.add_argument("--K_pts", type=int, default=120)
    ap.add_argument("--ang_tol", type=float, default=0.10)
    ap.add_argument("--near2", type=float, default=0.05)
    ap.add_argument("--outdir", type=str, default=None)

    # sigma2 sweep options
    ap.add_argument("--sigma2_min", type=float, default=None)
    ap.add_argument("--sigma2_max", type=float, default=None)
    ap.add_argument("--sigma2_pts", type=int, default=0)

    ap.add_argument("--post_ns", action="store_true")
    ap.add_argument("--deltaK", type=float, default=0.03)
    ap.add_argument("--T", type=int, default=60000)
    ap.add_argument("--burn", type=int, default=30000)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--lyap_demo", action="store_true")
    ap.add_argument("--le_q", type=int, default=4)

    # new flags for extra observables
    ap.add_argument("--mode_proj", action="store_true",
                    help="project trajectory onto NS eigenmodes")
    ap.add_argument("--freq_geo", action="store_true",
                    help="instantaneous frequency diagnostics")
    ap.add_argument("--poincare_sec", action="store_true",
                    help="Poincaré section on theta0-theta1=0")

    args = ap.parse_args()

    # Base params (sigma2 will be set per-run if sweeping)
    base_p = K4Params(
        scheme=args.scheme,
        dt=args.dt,
        gamma=args.gamma,
        directed=args.directed,
        eps_asym=args.eps_asym,
        deg_norm=args.deg_norm,
        K_min=args.K_min,
        K_max=args.K_max,
        K_pts=args.K_pts,
        ang_tol=args.ang_tol,
        post_ns=args.post_ns,
        deltaK=args.deltaK,
        T_diag=args.T,
        burn_diag=args.burn,
        noise=args.noise,
        lyap_demo=args.lyap_demo,
        le_q=args.le_q,
        topology=args.topology,
        w_diag=args.w_diag,
        sigma2=args.sigma2,
        triad_phi=args.triad_phi,
        near2=args.near2,
        mode_proj=args.mode_proj,
        freq_geo=args.freq_geo,
        poincare_sec=args.poincare_sec,
    )

    # Determine whether we sweep sigma2 or use a single value
    do_sigma2_sweep = (
        args.sigma2_min is not None
        and args.sigma2_max is not None
        and args.sigma2_pts is not None
        and args.sigma2_pts > 0
    )

    if do_sigma2_sweep:
        base_outdir = args.outdir or os.path.join(
            f"figs_k4_quat_{base_p.scheme}_{now_tag()}"
        )
        ensure_dir(base_outdir)

        sigma2_grid = np.linspace(
            args.sigma2_min, args.sigma2_max, args.sigma2_pts
        )
        print(
            f"Sigma2 sweep: from {args.sigma2_min:.6f} to "
            f"{args.sigma2_max:.6f} in {args.sigma2_pts} steps.\n"
        )

        for s2 in sigma2_grid:
            p_s2 = K4Params(**{**base_p.__dict__})
            p_s2.sigma2 = float(s2)
            subdir = os.path.join(base_outdir, f"sigma2_{s2:+.6f}")
            print("\n" + "=" * 72)
            print(f"Running sigma2 = {s2:+.6f}")
            print("=" * 72 + "\n")
            run_single_sigma2(p_s2, subdir)
    else:
        outdir = args.outdir or os.path.join(
            f"figs_k4_quat_{base_p.scheme}_{now_tag()}"
        )
        run_single_sigma2(base_p, outdir)


if __name__ == "__main__":
    main()
