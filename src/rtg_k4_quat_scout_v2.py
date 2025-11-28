# rtg_k4_quat_scout_v2.py
# Robust K4 double–NS scout with pair-tracking and pair-aware bracketing.

from __future__ import annotations
import os, json, math, argparse
from typing import Optional, List, Tuple, Dict
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

from rtg_core import (
    now_tag, ensure_dir, wrap_pi,
    gauge_free_basis_theta, block_diag,
    M_coupling_linear  # we'll assemble J here to allow a triad term
)

# ---------- K4 builder ----------
class K4Spec:
    def __init__(self, eps_asym: float = 0.25, directed: bool = True, deg_norm: bool = False):
        self.n = 4
        self.directed = directed
        self.eps_asym = eps_asym
        self.deg_norm = deg_norm

        # Base magnitudes = 1; add direction-asymmetry along a 0-1-2-3-0 cycle
        kappa = np.ones((4,4), dtype=float)
        np.fill_diagonal(kappa, 0.0)
        if directed:
            def oriented(i,j):
                return (j - i) % 4 == 1  # "forward" edges in 0->1->2->3->0
            for i in range(4):
                for j in range(4):
                    if i==j: continue
                    s = +1.0 if oriented(i,j) else -1.0
                    kappa[i,j] = 1.0 + s * self.eps_asym
        self.kappa = kappa

        # Phase shifts alpha_ij (single edge bank; keep simple = 0)
        alphas = np.zeros((4,4), dtype=float)
        self.alphas = alphas

        deg_vec = np.sum(np.abs(self.kappa), axis=1)
        if not deg_norm:
            deg_vec = np.ones_like(deg_vec)
        self.deg_vec = deg_vec

# Simple triad Laplacian (sum of triangle Laplacians over all 4 faces)
def k4_triad_laplacian() -> np.ndarray:
    triads = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    T = np.zeros((4,4), dtype=float)
    for (i,j,k) in triads:
        for a,b in [(i,j),(j,i),(i,k),(k,i),(j,k),(k,j)]:
            T[a,b] += 1.0
    # row-sum zero:
    for i in range(4):
        T[i,i] = -np.sum(T[i,:]) + T[i,i]
    return T

# ---------- Inertial Jacobian with optional triad term ----------
def jacobian_inertial_k4(K: float, dt: float, gamma: float,
                         spec: K4Spec,
                         sigma2: float = 0.0,
                         scheme: str = "explicit") -> np.ndarray:
    """
    We re-use rtg_core.M_coupling_linear to build the linear edges-only part,
    then add a small triad Laplacian 'sigma2 * T' as a scout knob.
    """
    n = spec.n
    A = 1.0 - gamma * dt

    # Edges-only linearization from rtg_core (already scaled by K)
    # We emulate the K3 function's signature by creating a fake GraphSpec-like object.
    class FakeSpec:
        pass
    fake = FakeSpec()
    fake.n = n
    fake.kappa = spec.kappa
    fake.deg_vec = spec.deg_vec
    fake.alphas = spec.alphas

    M_edges = M_coupling_linear(theta=None, K=K, spec=fake)

    # Optional small triad term (row-sum zero)
    if sigma2 != 0.0:
        T = k4_triad_laplacian()
        M = M_edges + sigma2 * T
    else:
        M = M_edges

    I = np.eye(n)
    if scheme.lower() in ("explicit","exp","e"):
        J = np.block([[I,           dt*I],
                      [dt*M,        A*I  ]])
    elif scheme.lower() in ("ec","euler_cromer","euler-cromer"):
        J = np.block([[I + (dt*dt)*M, dt*A*I],
                      [dt*M,          A*I  ]])
    else:
        raise ValueError(f"unknown scheme {scheme}")
    return J

# ---------- Pair extraction on the gauge-reduced Jacobian ----------
def jr_from_J(J: np.ndarray, n: int) -> np.ndarray:
    Qt = gauge_free_basis_theta(n)        # (n, n-1)
    Q  = block_diag(Qt, Qt)               # (2n, 2(n-1))
    return Q.T @ J @ Q                    # (2(n-1), 2(n-1))

def complex_pairs(Jr: np.ndarray) -> List[Tuple[complex, np.ndarray]]:
    w, V = npl.eig(Jr)  # V columns are eigenvectors
    pairs = []
    for idx, lam in enumerate(w):
        if lam.imag > 0.0:
            v = V[:, idx]
            v = v / (npl.norm(v) + 1e-30)
            pairs.append((lam, v))
    # sort by angle (away from 0)
    pairs.sort(key=lambda t: abs(np.angle(t[0])))
    return pairs

def track_pairs_over_grid(Kgrid: np.ndarray,
                          Jbuilder,
                          n: int,
                          ang_tol: float) -> Dict[str, np.ndarray]:
    """Return per-pair arrays: rho_j(K), ang_j(K), vec_j(K). Greedy match by vector overlap."""
    rho_list, ang_list, vec_list = [], [], []
    # first K
    Jr0 = jr_from_J(Jbuilder(Kgrid[0]), n)
    P0 = complex_pairs(Jr0)
    if len(P0) == 0:
        return {"rho": np.zeros((0,len(Kgrid))), "ang": np.zeros((0,len(Kgrid))), "vec": []}

    m = len(P0)  # how many pairs we'll try to track
    rho = np.full((m, len(Kgrid)), np.nan, dtype=float)
    ang = np.full((m, len(Kgrid)), np.nan, dtype=float)
    vec = [[None for _ in range(len(Kgrid))] for __ in range(m)]

    for j,(lam,v) in enumerate(P0):
        rho[j,0] = abs(lam)
        a = abs(np.angle(lam));  a = 2*np.pi - a if a>np.pi else a
        ang[j,0] = a
        vec[j][0] = v

    # follow
    for t in range(1, len(Kgrid)):
        Jr = jr_from_J(Jbuilder(Kgrid[t]), n)
        P  = complex_pairs(Jr)
        if len(P) == 0:
            continue
        used = set()
        # greedy match: for each previous j, pick current idx with max |<v_prev, v_curr>|
        for j in range(m):
            v_prev = vec[j][t-1]
            if v_prev is None:
                continue
            best_idx, best_ov = None, -1.0
            for i,(lam,v) in enumerate(P):
                if i in used: continue
                ov = abs(np.vdot(v_prev, v))
                if ov > best_ov:
                    best_ov, best_idx = ov, i
            if best_idx is not None:
                used.add(best_idx)
                lam, v = P[best_idx]
                rho[j,t] = abs(lam)
                a = abs(np.angle(lam));  a = 2*np.pi - a if a>np.pi else a
                ang[j,t] = a
                vec[j][t] = v
    return {"rho": rho, "ang": ang, "vec": vec}

# ---------- Per-pair bracketing + refinement ----------
def find_brackets_per_pair(K: np.ndarray, rho: np.ndarray, ang: np.ndarray, ang_tol: float):
    """For each pair j, return list of (i_lo, i_hi) indices where rho-1 changes sign and angle is OK."""
    brackets = [[] for _ in range(rho.shape[0])]
    for j in range(rho.shape[0]):
        for i in range(len(K)-1):
            r1, r2 = rho[j,i], rho[j,i+1]
            a1, a2 = ang[j,i], ang[j,i+1]
            if np.isnan(r1) or np.isnan(r2): continue
            if (r1-1.0)*(r2-1.0) <= 0.0:
                if (a1>ang_tol and abs(a1-np.pi)>ang_tol and
                    a2>ang_tol and abs(a2-np.pi)>ang_tol):
                    brackets[j].append((i, i+1))
    return brackets

def refine_pair_zero(Klo, Khi, v_lo, v_hi, Jbuilder, n, maxit=32):
    """Pair-aware bisection on rho-1==0 using eigenvector overlap to pick the same pair at midpoints."""
    def pair_rho_at(K, v_ref):
        Jr = jr_from_J(Jbuilder(K), n)
        P  = complex_pairs(Jr)
        if len(P)==0: return np.nan, None
        # choose the one aligned with v_ref
        best, best_ov = None, -1.0
        for lam,v in P:
            ov = abs(np.vdot(v_ref, v))
            if ov>best_ov:
                best_ov, best = ov, (lam,v)
        lam,v = best
        return abs(lam), v

    r_lo, _ = pair_rho_at(Klo, v_lo)
    r_hi, _ = pair_rho_at(Khi, v_hi)
    if np.isnan(r_lo) or np.isnan(r_hi): return (0.5*(Klo+Khi), np.nan)

    v_ref = (v_lo + v_hi) / (npl.norm(v_lo + v_hi) + 1e-30)
    K1, K2 = Klo, Khi
    for _ in range(maxit):
        Km = 0.5*(K1+K2)
        r_m, v_m = pair_rho_at(Km, v_ref)
        if np.isnan(r_m): break
        if (r_lo-1.0)*(r_m-1.0) <= 0.0:
            K2 = Km; r_hi = r_m; v_ref = (v_lo + v_m); v_ref /= (npl.norm(v_ref)+1e-30)
        else:
            K1 = Km; r_lo = r_m; v_ref = (v_m + v_hi); v_ref /= (npl.norm(v_ref)+1e-30)
    return 0.5*(K1+K2), r_m

# ---------- Plot helpers ----------
def plot_pairs(K, rho, ang, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    # spectral radii
    plt.figure(figsize=(6,4))
    for j in range(rho.shape[0]):
        plt.plot(K, rho[j,:], lw=0.9, label=f"pair{j+1}")
    plt.axhline(1.0, color="k", lw=0.8, ls="--")
    plt.xlabel("K"); plt.ylabel("rho (|lambda|)")
    plt.title("K4: spectral radii per complex pair")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_pairs_rho.png"), dpi=150)
    plt.close()

    # angles
    plt.figure(figsize=(6,4))
    for j in range(ang.shape[0]):
        plt.plot(K, ang[j,:], lw=0.9, label=f"pair{j+1}")
    plt.axhline(0.0, color="k", lw=0.5); plt.axhline(np.pi, color="k", lw=0.5)
    plt.xlabel("K"); plt.ylabel("angle (rad)")
    plt.title("K4: angles per complex pair")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_pairs_ang.png"), dpi=150)
    plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="K4 quaternion scout (robust pair tracking)")
    ap.add_argument("--scheme", type=str, default="explicit", choices=["explicit","ec"])
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--eps_asym", type=float, default=0.25)
    ap.add_argument("--deg_norm", action="store_true")

    ap.add_argument("--sigma2", type=float, default=0.0, help="small triad Laplacian weight")
    ap.add_argument("--triad_phi", type=float, default=0.0, help="(reserved)")

    ap.add_argument("--K_min", type=float, default=0.8)
    ap.add_argument("--K_max", type=float, default=2.0)
    ap.add_argument("--K_pts", type=int, default=300)
    ap.add_argument("--ang_tol", type=float, default=0.08)
    ap.add_argument("--near2", type=float, default=0.05, help="max |K1-K2| to call double-NS")

    ap.add_argument("--outdir", type=str, default=None)

    # Optional post run
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--deltaK", type=float, default=0.03)
    ap.add_argument("--T", type=int, default=60000)
    ap.add_argument("--burn", type=int, default=30000)
    ap.add_argument("--noise", type=float, default=1e-6)

    args = ap.parse_args()

    spec = K4Spec(eps_asym=args.eps_asym, directed=args.directed, deg_norm=args.deg_norm)

    tag = f"k4_quat_{args.scheme}_{now_tag()}"
    outdir = args.outdir or f"figs_{tag}"
    ensure_dir(outdir)

    print("Explicit: coupling + triads can push complex pairs to |lambda|=1, enabling multi-pair NS.")
    print(f"Target K4 | scheme={args.scheme} | dt={args.dt} | gamma={args.gamma} | directed={args.directed} eps={args.eps_asym} | deg_norm={args.deg_norm}")
    print(f"K range [{args.K_min}, {args.K_max}] with {args.K_pts} points; ang_tol={args.ang_tol:.3f}, near2={args.near2:.3f}, sigma2={args.sigma2:.3f}, phi={args.triad_phi:.3f}\n")

    Kgrid = np.linspace(args.K_min, args.K_max, args.K_pts)

    def Jbuilder(K):
        return jacobian_inertial_k4(K, args.dt, args.gamma, spec,
                                    sigma2=args.sigma2, scheme=args.scheme)

    # track pairs
    out = track_pairs_over_grid(Kgrid, Jbuilder, n=4, ang_tol=args.ang_tol)
    rho, ang, vec = out["rho"], out["ang"], out["vec"]
    if rho.size == 0:
        print("No complex pair found on this grid.")
        return

    # save sweep (pair-wise)
    with open(os.path.join(outdir, "k4_sweep.csv"), "w", encoding="utf-8") as f:
        f.write("K")
        for j in range(rho.shape[0]): f.write(f",rho{j+1},ang{j+1}")
        f.write("\n")
        for i,K in enumerate(Kgrid):
            f.write(f"{K:.9f}")
            for j in range(rho.shape[0]):
                rj = rho[j,i]; aj = ang[j,i]
                f.write(f",{rj:.9f},{aj:.9f}")
            f.write("\n")
    print(f"[save] sweep -> {os.path.join(outdir, 'k4_sweep.csv')}")

    plot_pairs(Kgrid, rho, ang, outdir, tag)

    # brackets per pair
    brackets = find_brackets_per_pair(Kgrid, rho, ang, args.ang_tol)
    # refine each pair's first bracket, if any
    roots = []
    for j, brs in enumerate(brackets):
        if not brs: continue
        i_lo,i_hi = brs[0]
        Klo, Khi  = Kgrid[i_lo], Kgrid[i_hi]
        Kstar, _  = refine_pair_zero(Klo, Khi, vec[j][i_lo], vec[j][i_hi], Jbuilder, n=4)
        roots.append((j, Kstar))

    if len(roots) < 2:
        print("No double-NS bracket found for two distinct pairs on this grid.")
        # Also report if a single crossing exists:
        if len(roots) == 1:
            j, K1 = roots[0]
            print(f"Single-pair crossing at K≈{K1:.6f} (pair {j+1}).")
        return

    # choose two closest roots
    roots.sort(key=lambda x: x[1])
    best = (None, None, 1e9)
    for a in range(len(roots)):
        for b in range(a+1, len(roots)):
            j1, K1 = roots[a]; j2, K2 = roots[b]
            d = abs(K2 - K1)
            if d < best[2]: best = (a, b, d)
    a,b,d = best
    j1,K1 = roots[a]; j2,K2 = roots[b]

    if d <= args.near2:
        print(f"Double-NS candidate: pair{j1+1}@K≈{K1:.6f} and pair{j2+1}@K≈{K2:.6f} (|ΔK|={d:.6f} ≤ near2)")
        with open(os.path.join(outdir, "k4_double_ns.json"), "w", encoding="utf-8") as f:
            json.dump({"pair1": int(j1+1), "K1": float(K1),
                       "pair2": int(j2+1), "K2": float(K2),
                       "dK": float(d)}, f, indent=2)
        # optional post run at mean K + delta
        if args.post:
            try:
                from rtg_core import step_inertial  # generic n, OK for K4
                theta = np.zeros(4, dtype=float)
                vel   = np.zeros(4, dtype=float)
                K_run = 0.5*(K1+K2) + args.deltaK
                r_series, x_series = [], []
                for t in range(args.T):
                    theta, vel = step_inertial(theta, vel, K_run,
                                               # wrap a minimal Params-like bundle:
                                               type("P",(object,),{"dt":args.dt,"gamma":args.gamma,"scheme":args.scheme})(),
                                               # wrap a minimal GraphSpec-like bundle:
                                               type("G",(object,),{"n":4,"kappa":spec.kappa,"deg_vec":spec.deg_vec,"alphas":spec.alphas})(),
                                               noise=args.noise)
                    if t >= args.burn:
                        z = np.exp(1j*theta).mean(); r_series.append(abs(z))
                        x_series.append(wrap_pi(theta[0]-theta[1]))
                r = np.asarray(r_series); x = np.asarray(x_series)
                # PSDs
                def psd(y):
                    Y = np.fft.rfft(y - y.mean()); f = np.fft.rfftfreq(len(y), d=1.0)
                    P = (Y.conj()*Y).real / max(1,len(y))
                    pk = int(np.argmax(P[1:]))+1 if len(P)>1 else 0
                    return f, P, (f[pk] if len(f)>pk else 0.0)
                fr, Pr, fpr = psd(r)
                fx, Px, fpx = psd(x)

                # plots
                plt.figure(figsize=(6,3.8)); plt.plot(r, lw=0.7)
                plt.title(f"r(t) @ K={K_run:.6f}"); plt.xlabel("step"); plt.ylabel("r")
                plt.tight_layout(); plt.savefig(os.path.join(outdir,"k4_ns_series_r.png"), dpi=150); plt.close()

                plt.figure(figsize=(6,3.8)); plt.plot(fr, Pr, lw=0.9)
                plt.title(f"PSD r(t) @ K={K_run:.6f}"); plt.xlabel("cycles/step"); plt.ylabel("power")
                plt.tight_layout(); plt.savefig(os.path.join(outdir,"k4_ns_psd_r.png"), dpi=150); plt.close()

                plt.figure(figsize=(6,3.8)); plt.plot(fx, Px, lw=0.9)
                plt.title(f"PSD (theta1-theta2) @ K={K_run:.6f}"); plt.xlabel("cycles/step"); plt.ylabel("power")
                plt.tight_layout(); plt.savefig(os.path.join(outdir,"k4_ns_psd_phase.png"), dpi=150); plt.close()

                with open(os.path.join(outdir,"k4_post_diag.json"),"w",encoding="utf-8") as f:
                    json.dump({"K_run": float(K_run), "f_peak_r_per_step": float(fpr),
                               "f_peak_phase_per_step": float(fpx), "r_mean": float(r.mean()),
                               "r_std": float(r.std())}, f, indent=2)
            except Exception as e:
                print(f"[post] skipped ({e})")
    else:
        print("Two per-pair crossings found, but |ΔK| > near2. Try lowering gamma, raising |sigma2|, or tightening eps_asym.")

if __name__ == "__main__":
    main()
