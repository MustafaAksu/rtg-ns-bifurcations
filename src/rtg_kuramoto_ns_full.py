#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rtg_kuramoto_ns_full.py

K3 with frustrated flux; NS-enabled probes for
 - 'memory' (1st-order Kuramoto map) and
 - 'inertial' (2nd-order Kuramoto map),

with selectable inertial time-stepping scheme (Euler–Cromer or Explicit),
directed asymmetry, gauge projection (θ and v zero-sum for inertial),
NS bracket + bisection, and headless plotting & diagnostics (PSD, phase portrait, Lyapunov).

USAGE EXAMPLES
--------------
# 1) Inertial, EC scheme (no true NS for γ>0; useful as a control)
python rtg_kuramoto_ns_full.py --family inertial --scheme ec --dt 0.1 --gamma 0.3 \
  --directed --eps_asym 0.2 --K_min 5.5 --K_max 6.1 --K_pts 121

# 2) Inertial, Explicit scheme (NS expected near μ(K) = -γ/dt)
python rtg_kuramoto_ns_full.py --family inertial --scheme explicit --dt 0.1 --gamma 0.3 \
  --directed --eps_asym 0.2 --K_min 1.3 --K_max 2.3 --K_pts 200 \
  --log_mu --post_ns --deltaK 0.03 --T_diag 60000 --burn_diag 30000

# 3) Memory family, small dt (helps complex angles appear)
python rtg_kuramoto_ns_full.py --family memory --dt 0.20 \
  --Delta 0.01 0.0 -0.01 --K_min 1.1 --K_max 2.2 --K_pts 61

NOTES
-----
- For EC inertial: complex pair radius is sqrt(1 - γ·dt) (K-independent), so NS crossing cannot occur for γ>0.
- For Explicit inertial: |λ|^2 = A - dt^2 μ, A=1-γ·dt, NS occurs at μ = -γ/dt.
- We now project to the invariant subspace {sum θ_i = 0, sum v_i = 0} for inertial eigen-analysis.
"""

import os, csv, math, time, argparse
import numpy as np
import numpy.linalg as LA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU = 2*np.pi

# ---------------- helpers ----------------
def wrap_pi(x): return (x + np.pi) % (2*np.pi) - np.pi

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def basis_gauge_free(n):
    """Orthonormal QR basis for the subspace {sum theta_i = 0}."""
    B = np.eye(n)[:, :-1] - np.eye(n)[:, [-1]]
    Q, _ = np.linalg.qr(B)
    return Q  # shape (n, n-1)

def write_csv(path, rows, header=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        w.writerows(rows)

# ---------------- graph + flux ----------------
def build_k3(F_target=2*np.pi/3, directed=False, eps=0.0, degree_normalize=True):
    n = 3
    # undirected template
    kappa = np.ones((n,n)) - np.eye(n)
    if directed:
        kappa[:] = 0.0
        Kf, Kr = 1.0 + eps, 1.0 - eps
        # 0->1->2->0 forward; reverse edges slightly weaker (nonreciprocal)
        kappa[0,1]=Kf; kappa[1,2]=Kf; kappa[2,0]=Kf
        kappa[1,0]=Kr; kappa[2,1]=Kr; kappa[0,2]=Kr
    alphas = np.zeros_like(kappa)
    a = F_target/3.0
    alphas[0,1]=a; alphas[1,2]=a; alphas[2,0]=a
    alphas[1,0]=-a; alphas[2,1]=-a; alphas[0,2]=-a
    deg_vec = kappa.sum(axis=1)
    return kappa, alphas, deg_vec, dict(F_target=F_target, directed=directed,
                                        eps=eps, degree_normalize=degree_normalize)

def triangle_flux(theta, alphas, cycle=(0,1,2,0)):
    F = 0.0
    for i,j in zip(cycle[:-1], cycle[1:]):
        F += wrap_pi(theta[j] - theta[i] - alphas[i,j])
    return wrap_pi(F)

def su2_half_trace_from_flux(F): return float(np.cos(0.5*F))

# ---------------- one-step maps ----------------
def step_memory(theta, K, Delta, kappa, alphas, deg_vec, dt, degree_normalize=True):
    n = len(theta)
    w = deg_vec if degree_normalize else np.ones(n)
    g = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i==j: continue
            s += kappa[i,j]*np.sin(theta[j] - theta[i] - alphas[i,j])
        g[i] = Delta[i] + (K/w[i])*s
    return theta + dt*g

def step_inertial(theta, vel, K, Delta, kappa, alphas, deg_vec, dt, gamma,
                  scheme="ec", degree_normalize=True):
    """Inertial update. 'ec' = Euler–Cromer (semi-implicit); 'explicit' = Forward Euler."""
    n = len(theta)
    w = deg_vec if degree_normalize else np.ones(n)
    # compute g(theta)
    g = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i==j: continue
            s += kappa[i,j]*np.sin(theta[j] - theta[i] - alphas[i,j])
        g[i] = Delta[i] + (K/w[i])*s

    if scheme == "ec":
        v_new = vel + dt*( -gamma*vel + g )
        theta_new = theta + dt*v_new
    elif scheme == "explicit":
        theta_new = theta + dt*vel
        v_new     = vel + dt*( -gamma*vel + g )
    else:
        raise ValueError("scheme must be 'ec' or 'explicit'")
    return theta_new, v_new

# ---------------- Jacobians (with dt) ----------------
def M_coupling(theta, K, kappa, alphas, deg_vec, degree_normalize=True):
    """Derivative of g(theta)=Delta + (K/w_i)*sum sin(θ_j-θ_i-α_ij) wrt theta."""
    n = len(theta)
    w = deg_vec if degree_normalize else np.ones(n)
    M = np.zeros((n,n))
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i==j: continue
            c = kappa[i,j]*np.cos(theta[j] - theta[i] - alphas[i,j])
            M[i,j] = (K/w[i])*c
            row_sum += c
        M[i,i] = -(K/w[i])*row_sum
    return M

def jacobian_memory(theta, K, kappa, alphas, deg_vec, dt, degree_normalize=True):
    n = len(theta)
    M = M_coupling(theta, K, kappa, alphas, deg_vec, degree_normalize)
    return np.eye(n) + dt*M

def jacobian_inertial(theta, K, kappa, alphas, deg_vec, dt, gamma,
                      scheme="ec", degree_normalize=True):
    n = len(theta)
    M = M_coupling(theta, K, kappa, alphas, deg_vec, degree_normalize)
    A = 1.0 - gamma*dt
    J = np.zeros((2*n, 2*n))
    if scheme == "ec":
        # Euler–Cromer:
        # v'  = v + dt*(-γ v + g(θ))
        # θ'  = θ + dt v'
        # => dθ'/dθ = I + dt^2 M,  dθ'/dv = dt*A I,  dv'/dθ = dt M,  dv'/dv = A I
        J[:n, :n] = np.eye(n) + (dt**2)*M
        J[:n, n:] = dt*A*np.eye(n)
        J[n:, :n] = dt*M
        J[n:, n:] = A*np.eye(n)
    elif scheme == "explicit":
        # Forward Euler:
        # θ' = θ + dt v
        # v' = v + dt*(-γ v + g(θ))
        # => dθ'/dθ = I,  dθ'/dv = dt I,  dv'/dθ = dt M,  dv'/dv = A I
        J[:n, :n] = np.eye(n)
        J[:n, n:] = dt*np.eye(n)
        J[n:, :n] = dt*M
        J[n:, n:] = A*np.eye(n)
    else:
        raise ValueError("scheme must be 'ec' or 'explicit'")
    return J

def gauge_project_eigs_memory(J):
    Q = basis_gauge_free(J.shape[0])     # (3,2)
    return LA.eigvals(Q.T @ J @ Q)

def gauge_project_eigs_inertial(J):
    """Project to invariant subspace {sum θ_i = 0, sum v_i = 0}."""
    n2 = J.shape[0] // 2
    Q = basis_gauge_free(n2)            # (n2, n2-1)
    Qfull = np.zeros((2*n2, 2*(n2-1)))
    Qfull[:n2, :n2-1] = Q               # theta block (zero-sum)
    Qfull[n2:, n2-1:] = Q               # velocity block (zero-sum)
    return LA.eigvals(Qfull.T @ J @ Qfull)

def mu_gauge_projected(theta, K, kappa, alphas, deg_vec, degree_normalize=True):
    """Gauge-free eigenvalues of M(θ*;K) on {sum θ=0}."""
    M = M_coupling(theta, K, kappa, alphas, deg_vec, degree_normalize)
    Q = basis_gauge_free(M.shape[0])    # (n, n-1)
    Mg = Q.T @ M @ Q
    return LA.eigvals(Mg)

# ---------------- Newton solve for locked state ----------------
def newton_locked(K, Delta, kappa, alphas, deg_vec, degree_normalize=True,
                  theta0=None, Omega0=0.0, tol=1e-12, maxit=80):
    """Solve g_i(theta) = Omega (common frequency), with gauge sum(theta)=0."""
    n = kappa.shape[0]
    theta = np.zeros(n) if theta0 is None else theta0.astype(float).copy()
    Omega = float(Omega0)

    def residual(theta, Omega):
        w = deg_vec if degree_normalize else np.ones(n)
        g = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i==j: continue
                s += kappa[i,j]*np.sin(theta[j] - theta[i] - alphas[i,j])
            g[i] = Delta[i] + (K/w[i])*s - Omega
        r = np.zeros(n+1)
        r[:n] = g
        r[n]  = theta.sum()  # gauge
        return r

    def jacobian(theta):
        M = M_coupling(theta, K, kappa, alphas, deg_vec, degree_normalize)
        n = M.shape[0]
        J = np.zeros((n+1, n+1))
        J[:n, :n] = M
        J[:n, n]  = -1.0
        J[n, :n]  =  1.0
        return J

    for _ in range(maxit):
        r = residual(theta, Omega)
        if LA.norm(r, ord=np.inf) < tol: return theta, Omega, True
        J = jacobian(theta)
        try:
            step = LA.solve(J, -r)
        except LA.LinAlgError:
            return theta, Omega, False
        theta += step[:-1]; Omega += step[-1]
        theta = wrap_pi(theta)
    return theta, Omega, False

# ---------------- plotting ----------------
def plot_flux_su2(outdir, F_series, tag=""):
    t = np.arange(len(F_series))
    plt.figure(figsize=(7.6,3.2))
    plt.subplot(1,2,1); plt.plot(t, F_series)
    plt.title(f"U(1) flux during run {tag}"); plt.xlabel("steps"); plt.ylabel("F(t) [rad]")
    plt.subplot(1,2,2); plt.plot(t, [np.cos(0.5*F) for F in F_series])
    plt.title("SU(2) half-trace = cos(F/2)"); plt.xlabel("steps"); plt.ylabel("½Tr(H)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "flux_su2_series.png"), dpi=170); plt.close()

def plot_ns_confirm_series(outdir, r_series, F_series, tag=""):
    t = np.arange(len(r_series))
    plt.figure(figsize=(8.8,3.4))
    plt.subplot(1,2,1); plt.plot(t, r_series); plt.ylim(0,1.05)
    plt.title(f"r(t) post-NS {tag}"); plt.xlabel("steps"); plt.ylabel("r")
    t2 = np.arange(len(F_series))
    plt.subplot(1,2,2); plt.plot(t2, [np.cos(0.5*F) for F in F_series])
    plt.title("SU(2) half-trace"); plt.xlabel("steps"); plt.ylabel("½Tr(H)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "ns_confirm_series.png"), dpi=170); plt.close()

def plot_fft(outdir, series, tag="", dt=1.0):
    y = np.asarray(series, float)
    y = y - y.mean()
    N = len(y)
    if N < 8:
        return None, None
    Y = np.fft.rfft(y)
    freqs_step = np.fft.rfftfreq(N, d=1.0)  # cycles per step
    psd = (np.abs(Y)**2)/N
    # ignore DC
    if len(psd) > 1:
        peak_idx = np.argmax(psd[1:]) + 1
    else:
        peak_idx = 0
    f_step = freqs_step[peak_idx]
    f_time = f_step / dt
    plt.figure(figsize=(6.0,3.2))
    plt.plot(freqs_step, psd)
    if peak_idx > 0:
        plt.axvline(f_step, ls='--', alpha=0.6)
    plt.title(f"PSD of r(t) {tag}")
    plt.xlabel("frequency (cycles / step)"); plt.ylabel("power")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "ns_confirm_psd.png"), dpi=170); plt.close()
    return float(f_step), float(f_time)

def plot_phase_portrait(outdir, xy_series, tag=""):
    XY = np.asarray(xy_series, float)
    if XY.size == 0: return
    plt.figure(figsize=(4.4,4.4))
    plt.plot(XY[:,0], XY[:,1], lw=0.8)
    plt.title(f"Phase portrait {tag}")
    plt.xlabel("θ1-θ2 [rad]"); plt.ylabel("θ2-θ3 [rad]")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "ns_confirm_phase.png"), dpi=170); plt.close()

def quick_grid_plots(outdir, rows, tag):
    rows = [r for r in rows if not np.isnan(r[1])]
    if not rows: return
    K  = np.array([r[0] for r in rows])
    rho= np.array([r[1] for r in rows])
    ang= np.array([r[2] for r in rows])
    F  = np.array([r[3] for r in rows])
    r0 = np.array([r[4] for r in rows])

    plt.figure(figsize=(10,3.6))
    plt.subplot(1,3,1); plt.plot(K, r0, 'o-', ms=3); plt.title("Coherence vs K")
    plt.xlabel("K"); plt.ylabel("r(θ*)")
    plt.subplot(1,3,2); plt.plot(K, F, 'o-', ms=3); plt.title("Flux vs K (invariant)")
    plt.xlabel("K"); plt.ylabel("F_triangle [rad]")
    plt.subplot(1,3,3); plt.plot(K, rho, 'o-', ms=3); plt.axhline(1.0, ls='--', c='gray')
    plt.title("Spectral radius vs K"); plt.xlabel("K"); plt.ylabel("max |λ|")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{tag}_grid_overview.png"), dpi=180); plt.close()

    plt.figure(figsize=(6.2,3.2))
    plt.plot(K, ang, 'o-', ms=3); plt.axhline(0, ls='--', c='gray'); plt.axhline(np.pi, ls='--', c='gray')
    plt.title("Eigenangle vs K"); plt.xlabel("K"); plt.ylabel("arg λ_max [rad]")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{tag}_angle_vs_K.png"), dpi=180); plt.close()

# ---------------- detect / refine ----------------
def detect_ns_bracket(rows, ang_tol=0.10):
    """Find K interval [Klo, Khi] where |λ| crosses 1 and angle not ~0 or π."""
    rows = [r for r in rows if not np.isnan(r[1])]
    for a, b in zip(rows[:-1], rows[1:]):
        K1, ρ1, φ1 = a[0], a[1], a[2]
        K2, ρ2, φ2 = b[0], b[1], b[2]
        if (ρ1-1.0)*(ρ2-1.0) <= 0.0:
            phi1 = np.mod(np.abs(φ1), np.pi)
            phi2 = np.mod(np.abs(φ2), np.pi)
            ok1 = (phi1 > ang_tol) and (phi1 < np.pi - ang_tol)
            ok2 = (phi2 > ang_tol) and (phi2 < np.pi - ang_tol)
            if ok1 and ok2:
                return K1, K2
    return None

def bisection_refine(Klo, Khi, eval_fun, iters=26, ang_tol=0.10):
    thL, ρL, φL = eval_fun(Klo, None)
    thH, ρH, φH = eval_fun(Khi, thL)
    seed = thH
    for _ in range(iters):
        Km = 0.5*(Klo+Khi)
        thM, ρM, φM = eval_fun(Km, seed)
        if thM is None: break
        seed = thM
        phiM = np.mod(np.abs(φM), np.pi)
        if (ρL-1.0)*(ρM-1.0) <= 0.0 and (phiM > ang_tol) and (phiM < np.pi - ang_tol):
            Khi, thH, ρH, φH = Km, thM, ρM, φM
        else:
            Klo, thL, ρL, φL = Km, thM, ρM, φM
    return 0.5*(Klo+Khi), max(ρL, ρH), 0.5*(φL+φH)

# ---------------- Lyapunov (simple Benettin) ----------------
def max_lyap_inertial(theta0, v0, step_fun, steps=20000, burn=5000, eps0=1e-7,
                      enforce_zero_sum=True):
    """
    Largest Lyapunov exponent per map step for inertial system by shadowing.
    step_fun: function (θ, v) -> (θ', v').
    """
    theta = theta0.copy(); v = v0.copy()
    thp = theta + np.random.normal(0, eps0, size=theta.shape)
    vp  = v + np.random.normal(0, eps0, size=v.shape)
    acc = 0.0; count = 0

    def project_zero_sum(th, vv):
        if not enforce_zero_sum: return th, vv
        th = th - th.mean()
        vv = vv - vv.mean()
        return th, vv

    theta, v   = project_zero_sum(theta, v)
    thp,  vp   = project_zero_sum(thp, vp)

    for t in range(steps):
        theta, v = step_fun(theta, v)
        thp,  vp = step_fun(thp, vp)
        dth = thp - theta; dv = vp - v
        if enforce_zero_sum:
            dth = dth - dth.mean(); dv = dv - dv.mean()
        d = np.sqrt(np.sum(dth**2) + np.sum(dv**2))
        if d == 0.0:
            # re-seed tiny separation
            thp = theta + np.random.normal(0, eps0, size=theta.shape)
            vp  = v + np.random.normal(0, eps0, size=v.shape)
            continue
        if t >= burn:
            acc += np.log(d/eps0); count += 1
        # renormalize
        thp = theta + (eps0/d)*dth
        vp  = v     + (eps0/d)*dv

    if count == 0: return 0.0
    return acc / count

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["memory","inertial"], default="memory")
    ap.add_argument("--scheme", choices=["ec","explicit"], default="ec",
                    help="inertial scheme: ec=Euler–Cromer (semi-implicit), explicit=Forward Euler")
    ap.add_argument("--dt", type=float, default=0.20, help="map time-step")
    ap.add_argument("--gamma", type=float, default=0.55, help="damping for inertial")
    ap.add_argument("--Delta", nargs=3, type=float, default=[0.01, 0.0, -0.01])
    ap.add_argument("--K_min", type=float, default=1.1)
    ap.add_argument("--K_max", type=float, default=2.1)
    ap.add_argument("--K_pts", type=int,   default=26)
    ap.add_argument("--F_target", type=float, default=2*np.pi/3)
    ap.add_argument("--directed", action="store_true", help="nonreciprocal edges")
    ap.add_argument("--eps_asym", type=float, default=0.0, help="fractional asymmetry if --directed")
    ap.add_argument("--deg_norm", action="store_true", help="degree-normalize coupling (default ON)")
    ap.add_argument("--no-deg_norm", dest="deg_norm", action="store_false")
    ap.set_defaults(deg_norm=True)

    # NS detection knobs
    ap.add_argument("--ang_tol", type=float, default=0.10, help="NS angle tolerance away from 0,π [rad]")

    # Logging / extras
    ap.add_argument("--log_mu", action="store_true", help="log μ(K) from gauge-free M and μ_eff from |λ| (explicit)")
    ap.add_argument("--debug", action="store_true")

    # Optional short demo at a single K (memory/inertial)
    ap.add_argument("--K_demo", type=float, default=None)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--burn", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="additive Gaussian noise (demo only)")

    # Post-NS diagnostics around K*+deltaK
    ap.add_argument("--post_ns", action="store_true", help="if NS found, run post-NS time-series diagnostics")
    ap.add_argument("--deltaK", type=float, default=0.03, help="offset above K* for post-NS run")
    ap.add_argument("--T_diag", type=int, default=60000)
    ap.add_argument("--burn_diag", type=int, default=30000)
    ap.add_argument("--noise_diag", type=float, default=0.0, help="noise for post-NS diagnostic run")

    # Lyapunov (alias lyap_demo for backwards compat)
    ap.add_argument("--lyap", action="store_true")
    ap.add_argument("--lyap_demo", dest="lyap", action="store_true")

    args = ap.parse_args()

    outdir = ensure_dir(f"figs_ns_full_{time.strftime('%Y%m%d_%H%M%S')}")
    kappa, alphas, deg_vec, meta = build_k3(F_target=args.F_target,
                                            directed=args.directed,
                                            eps=args.eps_asym,
                                            degree_normalize=args.deg_norm)
    Delta = np.array(args.Delta, dtype=float)
    Ks = np.linspace(args.K_min, args.K_max, args.K_pts)

    # Header
    header = (f"Target F={args.F_target:.6f} rad | family={args.family} "
              f"| scheme={args.scheme if args.family=='inertial' else 'n/a'} "
              f"| dt={args.dt} | gamma={args.gamma if args.family=='inertial' else '–'} "
              f"| Delta={Delta} | directed={args.directed} eps={args.eps_asym} | deg_norm={args.deg_norm}")
    print(header)

    # Theory banners
    if args.family == "inertial":
        A = 1.0 - args.gamma*args.dt
        if args.scheme == "ec":
            print(f"[theory] Inertial (Euler–Cromer): complex-pair modulus |λ|=sqrt(1-γ·dt)={math.sqrt(max(0.0,A)):.6f}; "
                  f"NS crossing is impossible for γ>0.")
        else:
            print(f"[theory] Inertial (Explicit): |λ|^2 = A - dt^2 μ,  A=1-γ·dt={A:.6f};  "
                  f"NS occurs at μ(K*) = -γ/dt = {-args.gamma/args.dt:.6f}.")

    rows = []
    mu_rows = []
    seed = None

    # Sweep over K
    for K in Ks:
        th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec, args.deg_norm,
                                   theta0=seed, Omega0=0.0)
        if not ok:
            rows.append([K, np.nan, np.nan, np.nan, np.nan, "fail"])
            continue
        seed = th

        if args.family == "memory":
            J = jacobian_memory(th, K, kappa, alphas, deg_vec, args.dt, args.deg_norm)
            lam = gauge_project_eigs_memory(J)
        else:
            J = jacobian_inertial(th, K, kappa, alphas, deg_vec, args.dt, args.gamma,
                                  scheme=args.scheme, degree_normalize=args.deg_norm)
            lam = gauge_project_eigs_inertial(J)

        # leading eigenvalue by modulus
        idx = np.argmax(np.abs(lam))
        lam1 = lam[idx]
        rho  = float(np.abs(lam1))
        ang  = float(np.angle(lam1))
        Ftri = float(triangle_flux(th, alphas))
        rbar = float(np.abs(np.exp(1j*th).mean()))

        # classify (rough)
        phi_mod = np.mod(np.abs(ang), np.pi)
        if abs(rho-1.0) > 1e-8 and (phi_mod > args.ang_tol) and (phi_mod < np.pi - args.ang_tol) and (rho > 1.0):
            kind = "NS-candidate"
        elif phi_mod < args.ang_tol:
            kind = "real+"
        elif (np.pi - phi_mod) < args.ang_tol:
            kind = "flip"
        else:
            kind = "complex"

        rows.append([K, rho, ang, Ftri, rbar, kind])

        if args.debug:
            fro = LA.norm(J, ord="fro")
            lam_sorted = sorted(lam, key=lambda z: -abs(z))[:3]
            print(f"K={K:6.3f} | |λ|max={rho:.6f}, arg={ang:+.3f} rad | r={rbar:.3f} | F={Ftri:.6f} | {kind} | ||J||_F={fro:.6f}")
            for z in lam_sorted:
                print(f"   top: {z.real:+.6f}{z.imag:+.6f}i  | |λ|={abs(z):.6f}")

        # log μ: gauge-free M eigs, and μ_eff from |λ| (explicit)
        if args.log_mu:
            mu = mu_gauge_projected(th, K, kappa, alphas, deg_vec, args.deg_norm)
            mu_sorted = sorted(mu, key=lambda z: (z.real, z.imag), reverse=True)
            mu0 = complex(mu_sorted[0]) if len(mu_sorted)>0 else complex(np.nan)
            mu1 = complex(mu_sorted[1]) if len(mu_sorted)>1 else complex(np.nan)
            mu_eff = np.nan
            if args.family == "inertial" and args.scheme == "explicit":
                A = 1.0 - args.gamma*args.dt
                mu_eff = (A - rho**2) / (args.dt**2)  # meaningful for complex pair
            mu_rows.append([K, mu_eff, mu0.real, mu0.imag, mu1.real, mu1.imag])

    # Write sweep
    tag = f"{args.family}_dt{args.dt:.3f}" + (f"_{args.scheme}" if args.family=="inertial" else "") + ("_dir" if args.directed else "")
    write_csv(os.path.join(outdir,"k_sweep.csv"),
              rows, header=["K","rho","angle","F_triangle","r(theta*)","type"])
    if args.log_mu and len(mu_rows)>0:
        write_csv(os.path.join(outdir,"modes_vs_K.csv"),
                  mu_rows, header=["K","mu_eff_from_lambda","mu0_real","mu0_imag","mu1_real","mu1_imag"])

    quick_grid_plots(outdir, rows, tag)

    # Predict K* (explicit only) from μ_eff(K) linear fit if available
    if args.family=="inertial" and args.scheme=="explicit" and args.log_mu and len(mu_rows)>=3:
        K_arr  = np.array([r[0] for r in mu_rows])
        mu_eff_arr = np.array([r[1] for r in mu_rows], float)
        mask = np.isfinite(mu_eff_arr)
        if mask.sum() >= 3:
            b, a = np.polyfit(K_arr[mask], mu_eff_arr[mask], 1)  # mu ≈ a + b K
            K_pred = ( -args.gamma/args.dt - a ) / b
            print(f"[predict] Based on μ_eff(K) fit, K* ≈ {K_pred:.6f}")

    # detect and refine NS bracket if present
    br = detect_ns_bracket(rows, ang_tol=args.ang_tol)
    if br:
        Klo, Khi = br
        print(f"\nNS-like crossing bracketed in [{Klo:.6f}, {Khi:.6f}] … refining")

        def _eval(K, seed_theta):
            th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec, args.deg_norm,
                                       theta0=seed_theta, Omega0=0.0)
            if not ok: return None, None, None
            if args.family == "memory":
                J = jacobian_memory(th, K, kappa, alphas, deg_vec, args.dt, args.deg_norm)
                lam = gauge_project_eigs_memory(J)
            else:
                J = jacobian_inertial(th, K, kappa, alphas, deg_vec, args.dt, args.gamma,
                                      scheme=args.scheme, degree_normalize=args.deg_norm)
                lam = gauge_project_eigs_inertial(J)
            lam1 = lam[np.argmax(np.abs(lam))]
            return th, float(np.abs(lam1)), float(np.angle(lam1))

        Kc, rhoc, phic = bisection_refine(Klo, Khi, _eval, iters=26, ang_tol=args.ang_tol)
        print(f"\nRefine result: K*={Kc:.9f} | rho={rhoc:.6f} | angle={phic:+.6f} rad  [NS-candidate]\n")

        # slope d|λ|/dK at K*
        dK = 1e-4*max(1.0, abs(Kc))
        _, rL, _ = _eval(Kc - dK, None)
        _, rH, _ = _eval(Kc + dK, None)
        slope = (rH - rL)/(2*dK)
        f_step_pred = abs(phic)/TAU
        f_time_pred = f_step_pred/args.dt if args.family=="inertial" else f_step_pred
        print(f"[check] d|λ|/dK at K* ≈ {slope:.3e}  |  f_step≈{f_step_pred:.5f}  f_time≈{f_time_pred:.5f}")

        # diagnostics post-NS (time series, PSD, phase)
        if args.post_ns and args.family=="inertial":
            K_post = Kc + args.deltaK
            th, Om, ok = newton_locked(K_post, Delta, kappa, alphas, deg_vec, args.deg_norm)
            if ok:
                theta = th.copy(); vel = np.zeros_like(theta)
                F_series, r_series, xy_series = [], [], []
                for t in range(args.T_diag):
                    if args.noise_diag>0:
                        theta += np.random.normal(0.0, args.noise_diag, size=theta.shape)
                        vel   += np.random.normal(0.0, args.noise_diag, size=vel.shape)
                    theta, vel = step_inertial(theta, vel, K_post, Delta, kappa, alphas, deg_vec,
                                               args.dt, args.gamma, scheme=args.scheme, degree_normalize=args.deg_norm)
                    if t >= args.burn_diag:
                        F_series.append(triangle_flux(theta, alphas))
                        r_series.append(np.abs(np.exp(1j*theta).mean()))
                        xy_series.append([wrap_pi(theta[0]-theta[1]), wrap_pi(theta[1]-theta[2])])

                # plots + frequency check
                plot_ns_confirm_series(outdir, r_series, F_series, tag=f"@K={K_post:.4f}")
                plot_flux_su2(outdir, F_series, tag=f"@K={K_post:.4f}")
                f_step_meas, f_time_meas = plot_fft(outdir, np.array(r_series), tag=f"@K={K_post:.4f}", dt=args.dt)
                plot_phase_portrait(outdir, xy_series, tag=f"@K={K_post:.4f}")

                if f_step_meas is not None:
                    print(f"[confirm] measured peak above K*: f_step≈{f_step_meas:.5f}  vs linear {f_step_pred:.5f}  "
                          f"(cycles/step); per-time {f_time_meas:.5f}")

                # Lyapunov (optional)
                if args.lyap:
                    step_fun = lambda th_, v_: step_inertial(th_, v_, K_post, Delta, kappa, alphas, deg_vec,
                                                             args.dt, args.gamma, scheme=args.scheme, degree_normalize=args.deg_norm)
                    LE1 = max_lyap_inertial(th.copy(), np.zeros_like(th), step_fun,
                                            steps=max(args.T_diag//2, 20000),
                                            burn=min(args.burn_diag, max(args.T_diag//4, 5000)),
                                            eps0=1e-7, enforce_zero_sum=True)
                    with open(os.path.join(outdir, "lyap_summary.txt"), "w", encoding="utf-8") as f:
                        f.write("Largest Lyapunov exponent (per step):\n")
                        f.write(f"  LE_1 ≈ {LE1:.6e}\n")

    else:
        # no bracket found; helpful hints
        rhos = np.array([r[1] for r in rows if np.isfinite(r[1])], float)
        if rhos.size > 0:
            if np.nanmin(rhos) > 1.0:
                print("\n[detect] min(|λ|) > 1 on this grid — you are already above K*. Lower K_min or widen the window.\n")
            elif np.nanmax(rhos) < 1.0:
                print("\n[detect] max(|λ|) < 1 on this grid — you are below K*. Raise K_max or widen the window.\n")
        print("No NS bracket found on this grid. Either flip precedes NS or the leading complex pair stays within |λ|<1 across the window.\n")

    # Optional single-K demo (quick)
    if args.K_demo is not None:
        K = float(args.K_demo)
        th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec, args.deg_norm)
        if ok:
            if args.family == "memory":
                theta = th.copy()
                F_series = []
                for t in range(args.T):
                    if args.noise>0: theta += np.random.normal(0.0, args.noise, size=theta.shape)
                    theta = step_memory(theta, K, Delta, kappa, alphas, deg_vec, args.dt, args.deg_norm)
                    if t>=args.burn: F_series.append(triangle_flux(theta, alphas))
                plot_flux_su2(outdir, F_series, tag=f"@K={K:.4f}")
            else:
                theta = th.copy(); vel = np.zeros_like(theta)
                F_series = []
                for t in range(args.T):
                    if args.noise>0:
                        theta += np.random.normal(0.0, args.noise, size=theta.shape)
                        vel   += np.random.normal(0.0, args.noise, size=vel.shape)
                    theta, vel = step_inertial(theta, vel, K, Delta, kappa, alphas, deg_vec,
                                               args.dt, args.gamma, scheme=args.scheme, degree_normalize=args.deg_norm)
                    if t>=args.burn: F_series.append(triangle_flux(theta, alphas))
                plot_flux_su2(outdir, F_series, tag=f"@K={K:.4f}")

    print(f"\nAll outputs saved in: {outdir}")
    print("(PNGs + CSV; nothing is shown interactively.)")

if __name__ == "__main__":
    main()
