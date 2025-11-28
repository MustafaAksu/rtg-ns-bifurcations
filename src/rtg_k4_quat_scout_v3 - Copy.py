# rtg_k4_quat_scout_v3.py
# K4 scout: scan all complex pairs, detect per-pair NS, test for double-NS proximity,
# optional post-NS demo & Lyapunov. Topologies: ringdiag, quat_sym.

from __future__ import annotations
import os, argparse, json, math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from rtg_core import (
    Params, GraphSpec, ensure_dir, now_tag, wrap_pi,
    jacobian_inertial, step_inertial, triangle_flux,
    gauge_free_basis_theta, block_diag, lyap_qr, post_ns_demo
)

# -------------------------------
# Params (K4-specific fields)
# -------------------------------
@dataclass
class K4Params(Params):
    topology: str = "ringdiag"       # ringdiag | quat_sym
    w_diag: float = 0.15             # diagonal weight (0<->2, 1<->3)
    sigma2: float = 0.0              # secondary scaling for diagonals (added)
    triad_phi: float = 0.0           # frustration put on selected edges
    near2: float = 0.05              # max |ΔK| to call a double-NS

    # reuse: post_ns, deltaK, T_diag->T, burn_diag->burn, noise, lyap_demo, le_q

# -------------------------------
# K4 builders
# -------------------------------

def build_k4_ringdiag(p: K4Params) -> GraphSpec:
    n = 4
    kappa = np.zeros((n,n), dtype=float)

    # Oriented ring 0->1->2->3->0
    for i in range(n):
        j = (i+1) % n
        kappa[i, j] = 1.0 + (p.eps_asym if p.directed else 0.0)
        kappa[j, i] = 1.0 - (p.eps_asym if p.directed else 0.0)

    # diagonals (0<->2 and 1<->3)
    d = p.w_diag + p.sigma2
    for (a,b) in [(0,2), (2,0), (1,3), (3,1)]:
        kappa[a,b] = d

    np.fill_diagonal(kappa, 0.0)

    alphas = np.zeros((n,n), dtype=float)
    if abs(p.triad_phi) > 0:
        # Put phi on two closing edges to seed frustration in two triangles
        alphas[2,0] = p.triad_phi
        alphas[3,1] = p.triad_phi

    deg_vec = np.sum(np.abs(kappa), axis=1)
    if not p.deg_norm:
        deg_vec = np.ones_like(deg_vec)
    return GraphSpec(n=n, directed=p.directed, eps_asym=p.eps_asym,
                     kappa=kappa, alphas=alphas, deg_vec=deg_vec, deg_norm=p.deg_norm)

def build_k4_quat_sym(p: K4Params) -> GraphSpec:
    """
    Symmetric K4: complete graph with ring orientation on nearest neighbors
    and separate weight on diagonals. Small eps_asym breaks degeneracies.
    """
    n = 4
    kappa = np.ones((n,n), dtype=float)
    np.fill_diagonal(kappa, 0.0)
    # distinguish nearest neighbors (ring) from diagonals
    ring_pairs = [(0,1),(1,2),(2,3),(3,0)]
    for (i,j) in ring_pairs:
        kappa[i,j] = 1.0 + (p.eps_asym if p.directed else 0.0)
        kappa[j,i] = 1.0 - (p.eps_asym if p.directed else 0.0)
    # diagonals
    diag_w = p.w_diag + p.sigma2
    for (a,b) in [(0,2),(2,0),(1,3),(3,1)]:
        kappa[a,b] = diag_w

    alphas = np.zeros((n,n), dtype=float)
    if abs(p.triad_phi) > 0:
        # symmetric frustration on two opposite closings
        alphas[2,0] = p.triad_phi
        alphas[3,1] = p.triad_phi

    deg_vec = np.sum(np.abs(kappa), axis=1)
    if not p.deg_norm:
        deg_vec = np.ones_like(deg_vec)
    return GraphSpec(n=n, directed=p.directed, eps_asym=p.eps_asym,
                     kappa=kappa, alphas=alphas, deg_vec=deg_vec, deg_norm=p.deg_norm)

def build_k4_spec(p: K4Params) -> GraphSpec:
    topo = p.topology.lower()
    if topo == "ringdiag":
        return build_k4_ringdiag(p)
    elif topo in ("quat_sym", "quat", "tetra"):
        return build_k4_quat_sym(p)
    else:
        raise ValueError(f"Unknown topology: {p.topology}")

# -------------------------------
# Pair-by-pair eigen tracking (gauge-free)
# -------------------------------

def eig_pairs(J: np.ndarray, n: int) -> List[complex]:
    """Return positive-imag eigenvalues on the gauge-free subspace, sorted by |.|."""
    Qtheta = gauge_free_basis_theta(n)
    Q = block_diag(Qtheta, Qtheta)
    Jr = Q.T @ J @ Q
    w = np.linalg.eigvals(Jr)
    # keep lambda with imag >= 0 (one from each complex pair)
    keep = [lam for lam in w if lam.imag >= -1e-12]
    keep.sort(key=lambda z: -abs(z))
    return keep[:(n-1)]  # there are (n-1) pairs

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
            ang = abs(np.angle(lam)); ang = min(ang, 2*np.pi - ang)
            angs[j, i] = ang
    return {"K": K, "rhos": rhos, "angs": angs}

def find_pair_bracket(K: np.ndarray, rho: np.ndarray, ang: np.ndarray, ang_tol: float) -> Optional[Tuple[float,float]]:
    for i in range(K.size-1):
        ok_i = (ang[i] > ang_tol) and (abs(ang[i]-np.pi) > ang_tol)
        ok_j = (ang[i+1] > ang_tol) and (abs(ang[i+1]-np.pi) > ang_tol)
        if ok_i and ok_j:
            if (rho[i]-1.0)*(rho[i+1]-1.0) <= 0.0:
                return (K[i], K[i+1])
    return None

def refine_pair(K_lo: float, K_hi: float, pair_idx: int, p: K4Params, g: GraphSpec,
                iters: int = 30) -> Tuple[float, float, float]:
    bestK, bestr, besta = None, None, None
    for _ in range(iters):
        km = 0.5*(K_lo + K_hi)
        J = jacobian_inertial(km, p, g, theta=None)
        lam = eig_pairs(J, g.n)[pair_idx]
        r = abs(lam)
        a = abs(np.angle(lam)); a = min(a, 2*np.pi - a)
        bestK, bestr, besta = km, r, a
        if r >= 1.0: K_hi = km
        else:        K_lo = km
    return bestK, bestr, besta

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="K4 double-NS scout")
    ap.add_argument("--topology", type=str, default="ringdiag",
                    choices=["ringdiag", "quat_sym"])
    ap.add_argument("--scheme", type=str, default="explicit", choices=["explicit","ec"])
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

    ap.add_argument("--post_ns", action="store_true")
    ap.add_argument("--deltaK", type=float, default=0.03)
    ap.add_argument("--T", type=int, default=60000)
    ap.add_argument("--burn", type=int, default=30000)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--lyap_demo", action="store_true")
    ap.add_argument("--le_q", type=int, default=4)

    args = ap.parse_args()
    p = K4Params(
        scheme=args.scheme, dt=args.dt, gamma=args.gamma,
        directed=args.directed, eps_asym=args.eps_asym, deg_norm=args.deg_norm,
        K_min=args.K_min, K_max=args.K_max, K_pts=args.K_pts, ang_tol=args.ang_tol,
        post_ns=args.post_ns, deltaK=args.deltaK, T_diag=args.T, burn_diag=args.burn,
        noise=args.noise, lyap_demo=args.lyap_demo, le_q=args.le_q,
        topology=args.topology, w_diag=args.w_diag, sigma2=args.sigma2,
        triad_phi=args.triad_phi, near2=args.near2
    )

    g = build_k4_spec(p)
    outdir = args.outdir or os.path.join(f"figs_k4_quat_{p.scheme}_{now_tag()}")
    ensure_dir(outdir)

    print("Explicit: coupling + triads can push complex pairs to |lambda|=1, enabling multi-pair NS.")
    print(f"Target K4 | scheme={p.scheme} | dt={p.dt} | gamma={p.gamma} | directed={p.directed} eps={p.eps_asym} | deg_norm={p.deg_norm}")
    print(f"K range [{p.K_min}, {p.K_max}] with {p.K_pts} points; ang_tol={p.ang_tol:.3f}, near2={p.near2:.3f}, sigma2={p.sigma2:.3f}, phi={p.triad_phi:.3f}\n")

    sweep = scan_k4(p, g)
    K = sweep["K"]; rhos = sweep["rhos"]; angs = sweep["angs"]
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
                row.append(f"{rhos[j,i]:.9f}")
                row.append(f"{angs[j,i]:.9f}")
            f.write(",".join(row) + "\n")
    print(f"[save] sweep -> {csv_path}")

    # Per-pair brackets
    brackets: Dict[int, Tuple[float,float]] = {}
    for j in range(m):
        br = find_pair_bracket(K, rhos[j], angs[j], p.ang_tol)
        if br is not None:
            brackets[j] = br

    if not brackets:
        print("No single-pair crossing detected on this grid.")
        return

    # Refine per-pair K* and report
    Kstars: Dict[int, Tuple[float,float,float]] = {}
    for j, (klo, khi) in brackets.items():
        kstar, rstar, astar = refine_pair(klo, khi, j, p, g, iters=32)
        Kstars[j] = (kstar, rstar, astar)
        print(f"pair{j+1}: K*={kstar:.9f} | rho=1.000000 | ang={astar:+.6f}")

    # Check double-NS proximity
    keys = sorted(Kstars.keys())
    best = None
    for a in range(len(keys)):
        for b in range(a+1, len(keys)):
            da = abs(Kstars[keys[a]][0] - Kstars[keys[b]][0])
            pairnames = (keys[a]+1, keys[b]+1)
            if (best is None) or (da < best[0]):
                best = (da, pairnames)

    if best is not None and best[0] <= p.near2:
        kA = Kstars[best[1][0]-1][0]
        kB = Kstars[best[1][1]-1][0]
        print(f"Double-NS candidate: pair{best[1][0]}@K≈{kA:.6f} and pair{best[1][1]}@K≈{kB:.6f} (|ΔK|={abs(kA-kB):.6f} ≤ near2)")
    else:
        if len(Kstars) >= 2:
            a, b = keys[0], keys[1]
            print(f"Two per-pair crossings found, but |ΔK| = {abs(Kstars[a][0]-Kstars[b][0]):.6f} > near2.")
        else:
            print("Single-pair NS found.")

    # Optional post run: choose the max K* then +deltaK
    if p.post_ns:
        k_pick = max(v[0] for v in Kstars.values())
        diag = post_ns_demo(k_pick, p, g, outdir, p.deltaK, p.T_diag, p.burn_diag, p.noise)
        print(f"[post] run at K={diag['K_run']:.6f} | r_mean={diag['r_mean']:.6f} | r_std={diag['r_std']:.3e} | f_peak_per_step={diag['f_peak_per_step']:.6f}")
        with open(os.path.join(outdir, "k4_post_diag.json"), "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)

    if p.lyap_demo:
        k_pick = max(v[0] for v in Kstars.values())
        les = lyap_qr(k_pick + p.deltaK, p, g, T=max(20000,p.T_diag//2), burn=max(5000,p.burn_diag//2),
                      q=max(1,p.le_q), noise=p.noise)
        with open(os.path.join(outdir, "lyap_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Largest Lyapunov exponents (per step):\n")
            for i, le in enumerate(les):
                f.write(f"  LE_{i+1} ~= {le:.6e}\n")
        print("Lyapunov summary written.")

if __name__ == "__main__":
    main()
