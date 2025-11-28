# rtg_k4_quat_scout.py
# K4 quaternion scout: double-NS detection with optional post-run+LEs
from __future__ import annotations
import os, math, time, json, argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

# -------- utils --------
def now_tag() -> str: return time.strftime("%Y%m%d_%H%M%S")
def ensure_dir(p: str) -> None: os.makedirs(p, exist_ok=True)
def wrap_pi(x): return (x + np.pi) % (2*np.pi) - np.pi
def block_diag(A,B):
    Z1 = np.zeros((A.shape[0], B.shape[1])); Z2 = np.zeros((B.shape[0], A.shape[1]))
    return np.block([[A,Z1],[Z2,B]])
def gauge_free_basis_theta(n: int) -> np.ndarray:
    e = np.eye(n); one = np.ones((n,1))/np.sqrt(n); cols=[]
    for k in range(n):
        v = e[:,k:k+1]; v = v - one @ (one.T @ v)
        for c in cols: v = v - c @ (c.T @ v)
        nv = npl.norm(v); 
        if nv>1e-12: cols.append(v/nv)
        if len(cols)==n-1: break
    return np.hstack(cols)

# -------- model spec (K4) --------
@dataclass
class K4Spec:
    n: int = 4
    directed: bool = True
    eps_asym: float = 0.25
    deg_norm: bool = False
    kappa: np.ndarray = None      # (4x4)
    alphas: np.ndarray = None     # (4x4)
    deg_vec: np.ndarray = None    # (4,)

def build_k4_spec(directed=True, eps_asym=0.25, deg_norm=False,
                  Ftri: float = -2.0*np.pi/3.0) -> K4Spec:
    n = 4
    kappa = np.zeros((n,n), dtype=float)
    # choose a consistent orientation for all 6 undirected edges:
    # forward edges: (1->2), (2->3), (3->4), (4->1), (1->3), (2->4)
    fw = [(0,1),(1,2),(2,3),(3,0),(0,2),(1,3)]
    for (i,j) in fw:
        kappa[i,j] = 1.0 + (eps_asym if directed else 0.0)
        kappa[j,i] = 1.0 - (eps_asym if directed else 0.0)
    np.fill_diagonal(kappa, 0.0)

    # alphas: antisymmetric; impose identical triangle flux Ftri on all four faces.
    # A consistent solution is: alpha_12 = alpha_34 = -Ftri; others = 0.
    alphas = np.zeros((n,n), dtype=float)
    alphas[0,1] = -Ftri; alphas[1,0] = +Ftri
    alphas[2,3] = -Ftri; alphas[3,2] = +Ftri

    deg_vec = np.sum(np.abs(kappa), axis=1)
    if not deg_norm: deg_vec = np.ones_like(deg_vec)
    return K4Spec(directed=directed, eps_asym=eps_asym, deg_norm=deg_norm,
                  kappa=kappa, alphas=alphas, deg_vec=deg_vec)

# -------- linearization (pairwise + triads) --------
def M_linear_K4(K: float, spec: K4Spec, sigma2: float = 0.0, phi: float = 0.0) -> np.ndarray:
    n = spec.n
    M = np.zeros((n,n), dtype=float)
    # pairwise (derivative of sin is cos)
    for i in range(n):
        rowsum = 0.0
        for j in range(n):
            if i==j: continue
            c = (spec.kappa[i,j]/spec.deg_vec[i]) * np.cos(-spec.alphas[i,j])
            M[i,j] += K*c
            rowsum += K*c
        # triads: for each unordered (j,k) != i
        if sigma2 != 0.0:
            nbrs = [u for u in range(n) if u!=i]
            for a in range(len(nbrs)):
                for b in range(a+1, len(nbrs)):
                    j, k = nbrs[a], nbrs[b]
                    # linearization of sin( (dj)+(dk) - phi ) at theta=0
                    tri_c = (sigma2/spec.deg_vec[i]) * np.cos(-(spec.alphas[i,j]+spec.alphas[i,k]) - phi)
                    M[i,j] += K*tri_c
                    M[i,k] += K*tri_c
                    rowsum += K*(tri_c + tri_c)
        M[i,i] = -rowsum
    return M

def jacobian_inertial_explicit(K: float, dt: float, gamma: float, M: np.ndarray) -> np.ndarray:
    n = M.shape[0]; A = 1.0 - gamma*dt; I = np.eye(n)
    return np.block([[I, dt*I],[dt*M, A*I]])

# -------- eigen-pair extraction on gauge-reduced space --------
def top_two_complex_pairs(J: np.ndarray, n: int, ang_tol: float) -> Tuple[List[complex], List[float], List[float]]:
    Qth = gauge_free_basis_theta(n); Q = block_diag(Qth, Qth)
    Jr = Q.T @ J @ Q
    w = npl.eigvals(Jr)
    # take conjugate-positive representatives
    vec = [(lam, abs(lam), abs(np.angle(lam))) for lam in w if abs(np.imag(lam))>1e-10]
    vec.sort(key=lambda t: -t[1])  # by radius
    pairs, rads, angs = [], [], []
    used = np.zeros(len(vec), dtype=bool)
    for i,(lami,ri,ai) in enumerate(vec):
        if used[i]: continue
        if not (ang_tol < ai < np.pi-ang_tol): 
            continue
        # choose representative with positive imag
        if np.imag(lami) < 0: lami = np.conj(lami); ri = abs(lami); ai = abs(np.angle(lami))
        pairs.append(lami); rads.append(ri); angs.append(ai)
        if len(pairs)==2: break
    return pairs, rads, angs

# -------- sweep / bracket / refine --------
def sweep_K4(Kmin,Kmax,Kpts,dt,gamma,spec,sigma2,phi,ang_tol):
    Ks = np.linspace(Kmin,Kmax,Kpts)
    rho1 = np.zeros_like(Ks); ang1 = np.zeros_like(Ks)
    rho2 = np.zeros_like(Ks); ang2 = np.zeros_like(Ks)
    for i,K in enumerate(Ks):
        M = M_linear_K4(K, spec, sigma2, phi)
        J = jacobian_inertial_explicit(K, dt, gamma, M)
        _, r, a = top_two_complex_pairs(J, spec.n, ang_tol)
        if len(r)>=1: rho1[i], ang1[i] = r[0], a[0]
        if len(r)>=2: rho2[i], ang2[i] = r[1], a[1]
    return Ks, rho1, ang1, rho2, ang2

def find_double_bracket(Ks,r1,r2,a1,a2,ang_tol):
    for i in range(len(Ks)-1):
        ok_i = (a1[i]>ang_tol and a2[i]>ang_tol and a1[i]<np.pi-ang_tol and a2[i]<np.pi-ang_tol)
        ok_j = (a1[i+1]>ang_tol and a2[i+1]>ang_tol and a1[i+1]<np.pi-ang_tol and a2[i+1]<np.pi-ang_tol)
        if not (ok_i and ok_j): continue
        d11 = r1[i]-1.0; d12 = r1[i+1]-1.0
        d21 = r2[i]-1.0; d22 = r2[i+1]-1.0
        if d11*d12 <= 0.0 and d21*d22 <= 0.0:
            return Ks[i], Ks[i+1]
    return None

def refine_double(Klo,Khi,dt,gamma,spec,sigma2,phi,ang_tol,iters=40):
    best = (None, 1e9, 0.0, 0.0)
    lo,hi = Klo,Khi
    for _ in range(iters):
        Km = 0.5*(lo+hi)
        M = M_linear_K4(Km, spec, sigma2, phi)
        J = jacobian_inertial_explicit(Km, dt, gamma, M)
        _, r, a = top_two_complex_pairs(J, spec.n, ang_tol)
        if len(r)<2: break
        dev = max(abs(r[0]-1.0), abs(r[1]-1.0))
        if dev < best[1]: best = (Km, dev, r[0], r[1])
        # choose side by which pair is further outside
        s1 = r[0]-1.0; s2 = r[1]-1.0
        # heuristic: move toward balancing the signs
        if (s1>0 and s2>0): hi = Km
        elif (s1<0 and s2<0): lo = Km
        else:
            # opposite signs: split the one farther from zero
            if abs(s1) > abs(s2):
                if s1>0: hi = Km
                else: lo = Km
            else:
                if s2>0: hi = Km
                else: lo = Km
    Km, dev, r1, r2 = best
    return Km, r1, r2

# -------- nonlinear stepper (pairwise + triads) --------
def step_K4(theta, vel, K, dt, gamma, spec, sigma2=0.0, phi=0.0, noise=0.0):
    n = 4
    f = np.zeros(n, dtype=float)
    for i in range(n):
        # pairwise
        ps = 0.0
        for j in range(n):
            if i==j: continue
            ps += spec.kappa[i,j] * math.sin(theta[j]-theta[i]-spec.alphas[i,j])
        ps /= spec.deg_vec[i]
        # triads
        ts = 0.0
        if sigma2 != 0.0:
            nbrs = [u for u in range(n) if u!=i]
            for a in range(len(nbrs)):
                for b in range(a+1,len(nbrs)):
                    j,k = nbrs[a], nbrs[b]
                    ts += math.sin((theta[j]-theta[i]-spec.alphas[i,j]) +
                                   (theta[k]-theta[i]-spec.alphas[i,k]) - phi)
            ts /= spec.deg_vec[i]
        f[i] = K*(ps + sigma2*ts)
    theta_new = theta + dt*vel
    vel_new   = (1.0 - gamma*dt)*vel + dt*f
    if noise>0.0:
        theta_new = theta_new + np.random.normal(0.0, noise, size=theta.shape)
    return theta_new, vel_new

def largest_q_lyap(K,dt,gamma,spec,sigma2,phi,q,T,burn,noise=0.0):
    n = 4; D = 2*n
    theta = np.zeros(n); vel = np.zeros(n)
    Q = np.linalg.qr(np.random.randn(D, q))[0]
    sums = np.zeros(q); cnt=0
    for t in range(T):
        # Jacobian at current state
        M = M_linear_K4(K, spec, sigma2, phi)  # linearization at theta~0 is a reasonable proxy
        J = jacobian_inertial_explicit(K, dt, gamma, M)
        Z = J @ Q
        Q, R = np.linalg.qr(Z)
        if t>=burn:
            d = np.abs(np.diag(R)); d[d==0]=1e-30
            sums += np.log(d); cnt += 1
        theta, vel = step_K4(theta, vel, K, dt, gamma, spec, sigma2, phi, noise=noise)
    if cnt==0: return np.zeros(q)
    return sums / cnt

# -------- plots --------
def plot_pairs(Ks,r1,a1,r2,a2,outdir,tag):
    plt.figure(figsize=(6,4))
    plt.plot(Ks, r1, lw=1.0, label="|lambda| pair1")
    plt.plot(Ks, r2, lw=1.0, label="|lambda| pair2")
    plt.axhline(1.0, color='k', ls='--', lw=0.8)
    plt.xlabel("K"); plt.ylabel("spectral radius"); plt.title("K4: leading two complex pairs")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_rho_pairs.png"), dpi=160); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(Ks, a1, lw=1.0, label="angle pair1")
    plt.plot(Ks, a2, lw=1.0, label="angle pair2")
    plt.xlabel("K"); plt.ylabel("angle (rad)"); plt.title("K4: eigen-angles of the two pairs")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_ang_pairs.png"), dpi=160); plt.close()

def psd(y):
    y = np.asarray(y); N=len(y)
    if N==0: return np.array([0.0]), np.array([0.0])
    z = y - y.mean(); R = np.fft.rfft(z); f = np.fft.rfftfreq(N, d=1.0)
    P = (R.conj()*R).real / max(1,N); return f,P

def post_run(Kstar, deltaK, dt, gamma, spec, sigma2, phi, T, burn, noise, outdir):
    K = Kstar + deltaK
    n = 4; theta = np.zeros(n); vel = np.zeros(n)
    r_series = []; ph12 = []
    F123 = []
    for t in range(T):
        theta, vel = step_K4(theta, vel, K, dt, gamma, spec, sigma2, phi, noise)
        if t>=burn:
            z = np.exp(1j*theta).mean(); r_series.append(abs(z))
            ph12.append(wrap_pi(theta[0]-theta[1]))
            # one triangle flux F(1,2,3)
            F = wrap_pi((theta[1]-theta[0]-spec.alphas[0,1]) +
                        (theta[2]-theta[1]-spec.alphas[1,2]) +
                        (theta[0]-theta[2]-spec.alphas[2,0]))
            F123.append(F)

    r_series = np.asarray(r_series); ph12=np.asarray(ph12); F123=np.asarray(F123)
    fr,Pr = psd(r_series); fp,Pp = psd(ph12)
    # plots
    plt.figure(figsize=(7,4)); plt.plot(r_series, lw=0.8)
    plt.title(f"r(t) @ K={K:.6f}"); plt.xlabel("steps"); plt.ylabel("r")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"k4_post_series.png"), dpi=160); plt.close()

    plt.figure(figsize=(7,4)); plt.plot(fr,Pr,lw=1.0)
    plt.title(f"PSD of r(t) @ K={K:.6f}"); plt.xlabel("cycles/step"); plt.ylabel("power")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"k4_post_psd_r.png"), dpi=160); plt.close()

    plt.figure(figsize=(7,4)); plt.plot(fp,Pp,lw=1.0)
    plt.title(f"PSD of (theta1-theta2) @ K={K:.6f}"); plt.xlabel("cycles/step"); plt.ylabel("power")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"k4_post_psd_phase.png"), dpi=160); plt.close()

    plt.figure(figsize=(7,4)); plt.plot(F123,lw=0.8)
    plt.title("Triangle flux F(1-2-3) (principal branch)"); plt.xlabel("steps"); plt.ylabel("F (rad)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"k4_post_flux.png"), dpi=160); plt.close()

    out = {
        "K_run": float(K),
        "r_mean": float(r_series.mean()) if r_series.size else 0.0,
        "r_std": float(r_series.std()) if r_series.size else 0.0,
        "f_peak_r": float(fr[np.argmax(Pr)]) if fr.size else 0.0,
        "f_peak_phase": float(fp[np.argmax(Pp)]) if fp.size else 0.0
    }
    with open(os.path.join(outdir,"k4_post_diag.json"),"w",encoding="utf-8") as f: json.dump(out,f,indent=2)
    return out

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="K4 quaternion scout (double-NS)")
    ap.add_argument("--scheme", type=str, default="explicit", choices=["explicit"])
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--eps_asym", type=float, default=0.25)
    ap.add_argument("--deg_norm", action="store_true")
    ap.add_argument("--sigma2", type=float, default=0.0)
    ap.add_argument("--triad_phi", type=float, default=0.0)
    ap.add_argument("--K_min", type=float, default=0.8)
    ap.add_argument("--K_max", type=float, default=2.0)
    ap.add_argument("--K_pts", type=int, default=240)
    ap.add_argument("--ang_tol", type=float, default=0.08)
    ap.add_argument("--near2", type=float, default=0.05)  # not used in this version but kept for compat
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--deltaK", type=float, default=0.03)
    ap.add_argument("--T", type=int, default=60000)
    ap.add_argument("--burn", type=int, default=30000)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--lyap_demo", action="store_true")
    ap.add_argument("--le_q", type=int, default=4)
    args = ap.parse_args()

    print("Explicit: coupling + triads can push complex pairs to |lambda|=1, enabling multi-pair NS.")
    spec = build_k4_spec(directed=args.directed, eps_asym=args.eps_asym, deg_norm=args.deg_norm)
    tag = f"k4_quat_{args.scheme}_{now_tag()}"; outdir = f"figs_{tag}"; ensure_dir(outdir)
    print(f"Target K4 | scheme={args.scheme} | dt={args.dt} | gamma={args.gamma} | directed={args.directed} eps={args.eps_asym} | deg_norm={args.deg_norm}")
    print(f"K range [{args.K_min}, {args.K_max}] with {args.K_pts} points; ang_tol={args.ang_tol:.3f}, near2={args.near2:.3f}, sigma2={args.sigma2:.3f}, phi={args.triad_phi:.3f}\n")

    Ks,r1,a1,r2,a2 = sweep_K4(args.K_min,args.K_max,args.K_pts,args.dt,args.gamma,spec,args.sigma2,args.triad_phi,args.ang_tol)
    # save CSV + plots
    csv = os.path.join(outdir,"k4_sweep.csv")
    with open(csv,"w",encoding="utf-8") as f:
        f.write("K,rho1,ang1,rho2,ang2\n")
        for i in range(len(Ks)): f.write(f"{Ks[i]:.9f},{r1[i]:.9f},{a1[i]:.9f},{r2[i]:.9f},{a2[i]:.9f}\n")
    print(f"[save] sweep -> {csv}")
    plot_pairs(Ks,r1,a1,r2,a2,outdir,tag)

    br = find_double_bracket(Ks,r1,r2,a1,a2,args.ang_tol)
    if br is None:
        print("No double-NS bracket found. You can:\n  - decrease near2 (e.g., --near2 0.03)\n  - try sigma2<0 (e.g., --sigma2 -0.22)\n  - reduce gamma or dt slightly\nNo single-pair crossing detected on this grid either.")
        return

    Klo,Khi = br
    print(f"Double-NS candidate bracket in [{Klo:.6f}, {Khi:.6f}] ... refining")
    Kstar, R1, R2 = refine_double(Klo,Khi,args.dt,args.gamma,spec,args.sigma2,args.triad_phi,args.ang_tol)
    print(f"Refine result: K*={Kstar:.9f} | pair1: rho={R1:.6f} | pair2: rho={R2:.6f}")

    if args.post:
        diag = post_run(Kstar, args.deltaK, args.dt, args.gamma, spec, args.sigma2, args.triad_phi,
                        args.T, args.burn, args.noise, outdir)
        print(f"[post] run at K={diag['K_run']:.6f} | r_mean={diag['r_mean']:.6f} | r_std={diag['r_std']:.3e} | f_peak_r={diag['f_peak_r']:.6f} | f_peak_phase={diag['f_peak_phase']:.6f}")
    if args.lyap_demo:
        les = largest_q_lyap(Kstar+args.deltaK, args.dt, args.gamma, spec, args.sigma2, args.triad_phi,
                             q=args.le_q, T=max(args.T//2,20000), burn=max(args.burn//2,5000), noise=args.noise)
        print("Largest Lyapunov exponents (per step):")
        for i,le in enumerate(les): print(f"  LE_{i+1} ~= {le:.6e}")

if __name__=="__main__":
    main()
