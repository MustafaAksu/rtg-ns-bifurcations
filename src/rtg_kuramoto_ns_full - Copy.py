#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rtg_kuramoto_ns_full.py

K3 with frustrated flux; Neimark–Sacker (NS) probes for
 - 'memory' (first-order Kuramoto map) and
 - 'inertial' (second-order Kuramoto map, Euler–Cromer update),

with explicit time-step dt, directed asymmetry, gauge projection,
NS bracket + bisection, and optional post-NS confirmation
(phase portrait + FFT + transversality slope).

Examples
--------
# Inertial NS case (your successful regime)
python rtg_kuramoto_ns_full.py --family inertial --dt 0.1 --gamma 0.3 \
  --directed --eps_asym 0.2 --K_min 5 --K_max 6 --K_pts 60 \
  --post_ns --deltaK 0.03 --T_diag 60000 --burn_diag 30000

# Short post-NS time-series demo (adds tiny noise just for the demo)
python rtg_kuramoto_ns_full.py --family inertial --dt 0.1 --gamma 0.3 \
  --directed --eps_asym 0.2 --K_demo 5.90 --T 10000 --burn 5000 --noise 1e-6
"""

import os, csv, time, argparse
import numpy as np
import numpy.linalg as LA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU = 2.0*np.pi

# ---------------- helpers ----------------
def wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def basis_gauge_free(n: int) -> np.ndarray:
    """Orthonormal basis Q for {sum_i theta_i = 0} in R^n, shape (n, n-1)."""
    B = np.eye(n)[:, :-1] - np.eye(n)[:, [-1]]
    Q, _ = np.linalg.qr(B)
    return Q

def write_csv(path, rows, header=None):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        w.writerows(rows)

# ---------------- graph + flux ----------------
def build_k3(F_target=2*np.pi/3, directed=False, eps=0.0, degree_normalize=True):
    """
    K3 with per-edge phase shifts α_ij summing to F_target on the 0→1→2→0 triangle.
    If directed=True, forward edges have (1+eps), backward (1-eps) coupling.
    """
    n = 3
    kappa = np.ones((n,n)) - np.eye(n)
    if directed:
        kappa[:] = 0.0
        Kf, Kr = 1.0 + eps, 1.0 - eps
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
    """Gauge-invariant U(1) flux accumulated around the triangle."""
    F = 0.0
    for i, j in zip(cycle[:-1], cycle[1:]):
        F += wrap_pi(theta[j] - theta[i] - alphas[i,j])
    return wrap_pi(F)

# ---------------- one-step maps (with dt) ----------------
def _w_vec(deg_vec, degree_normalize: bool):
    return deg_vec if degree_normalize else np.ones_like(deg_vec)

def step_memory(theta, K, Delta, kappa, alphas, deg_vec, dt, degree_normalize=True):
    n = len(theta)
    w = _w_vec(deg_vec, degree_normalize)
    g = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j: continue
            s += kappa[i,j]*np.sin(theta[j] - theta[i] - alphas[i,j])
        g[i] = Delta[i] + (K/w[i])*s
    return theta + dt*g

def step_inertial(theta, vel, K, Delta, kappa, alphas, deg_vec, dt, gamma, degree_normalize=True):
    """
    Euler–Cromer update:
      v_{t+1} = v_t + dt*(-γ v_t + g(theta_t))
      θ_{t+1} = θ_t + dt * v_{t+1}
    """
    n = len(theta)
    w = _w_vec(deg_vec, degree_normalize)
    g = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j: continue
            s += kappa[i,j]*np.sin(theta[j] - theta[i] - alphas[i,j])
        g[i] = Delta[i] + (K/w[i])*s
    v_new = vel + dt*(-gamma*vel + g)
    th_new = theta + dt*v_new
    return th_new, v_new

# ---------------- Jacobians (with dt) ----------------
def M_coupling(theta, K, kappa, alphas, deg_vec, degree_normalize=True):
    """
    Jacobian of g(theta) = Delta + (K/w_i) * sum_j kappa_ij sin(θ_j - θ_i - α_ij)
    with respect to theta.
    """
    n = len(theta)
    w = _w_vec(deg_vec, degree_normalize)
    M = np.zeros((n,n))
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i == j: continue
            c = kappa[i,j]*np.cos(theta[j] - theta[i] - alphas[i,j])
            M[i,j] = (K/w[i])*c
            row_sum += c
        M[i,i] = -(K/w[i])*row_sum
    return M

def jacobian_memory(theta, K, kappa, alphas, deg_vec, dt, degree_normalize=True):
    M = M_coupling(theta, K, kappa, alphas, deg_vec, degree_normalize)
    return np.eye(len(theta)) + dt*M

def jacobian_inertial(theta, K, kappa, alphas, deg_vec, dt, gamma, degree_normalize=True):
    """
    Linearization of Euler–Cromer step about (theta*, v* = 0):
        dθ' = dθ + dt dv'
        dv' = (1 - γ dt) dv + dt M dθ
    ⇒ J = [[I + dt^2 M,   dt(1 - γ dt) I],
           [dt M,         (1 - γ dt) I ]]
    """
    n = len(theta)
    M = M_coupling(theta, K, kappa, alphas, deg_vec, degree_normalize)
    A = 1.0 - gamma*dt
    J = np.zeros((2*n, 2*n))
    J[:n, :n] = np.eye(n) + (dt**2)*M
    J[:n, n:] = dt*A*np.eye(n)
    J[n:, :n] = dt*M
    J[n:, n:] = A*np.eye(n)
    return J

def gauge_project_eigs_memory(J):
    Q = basis_gauge_free(J.shape[0])   # (n, n-1)
    return LA.eigvals(Q.T @ J @ Q)

def gauge_project_eigs_inertial(J):
    n2 = J.shape[0] // 2
    Qθ = basis_gauge_free(n2)          # (n, n-1)
    Q2 = np.zeros((2*n2, (n2-1)+n2))
    Q2[:n2, :n2-1] = Qθ
    Q2[n2:, n2-1:] = np.eye(n2)
    return LA.eigvals(Q2.T @ J @ Q2)

# ---------------- Newton solve for locked state ----------------
def newton_locked(K, Delta, kappa, alphas, deg_vec, degree_normalize=True,
                  theta0=None, Omega0=0.0, tol=1e-12, maxit=80):
    """
    Solve g_i(theta) = Omega (common frequency), with gauge sum_i theta_i = 0.
    Returns (theta*, Omega, success)
    """
    n = kappa.shape[0]
    theta = np.zeros(n) if theta0 is None else np.array(theta0, dtype=float).copy()
    Omega = float(Omega0)

    def residual(theta, Omega):
        w = _w_vec(deg_vec, degree_normalize)
        g = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i == j: continue
                s += kappa[i,j]*np.sin(theta[j] - theta[i] - alphas[i,j])
            g[i] = Delta[i] + (K/w[i])*s - Omega
        r = np.zeros(n+1)
        r[:n] = g
        r[n]  = theta.sum()
        return r

    def jac(theta):
        M = M_coupling(theta, K, kappa, alphas, deg_vec, degree_normalize)
        n = M.shape[0]
        J = np.zeros((n+1, n+1))
        J[:n, :n] = M
        J[:n, n]  = -1.0
        J[n, :n]  =  1.0
        return J

    for _ in range(maxit):
        r = residual(theta, Omega)
        if LA.norm(r, ord=np.inf) < tol:
            return wrap_pi(theta), Omega, True
        J = jac(theta)
        try:
            step = LA.solve(J, -r)
        except LA.LinAlgError:
            return wrap_pi(theta), Omega, False
        theta += step[:-1]; Omega += step[-1]
        theta = wrap_pi(theta)
    return wrap_pi(theta), Omega, False

# ---------------- plots ----------------
def plot_flux_su2(outdir, F_series):
    t = np.arange(len(F_series))
    plt.figure(figsize=(9.6,4.2))
    plt.subplot(1,2,1); plt.plot(t, F_series)
    plt.title("U(1) flux during run"); plt.xlabel("steps"); plt.ylabel("F(t) (rad)")
    plt.subplot(1,2,2); plt.plot(t, [np.cos(0.5*F) for F in F_series], label="cos(F/2)")
    plt.title("SU(2) trace–angle check"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "flux_su2_series.png"), dpi=180); plt.close()

def quick_grid_plots(outdir, rows, tag):
    rows = [r for r in rows if not np.isnan(r[1])]
    if not rows: return
    K  = np.array([r[0] for r in rows])
    rho= np.array([r[1] for r in rows])
    ang= np.array([r[2] for r in rows])
    F  = np.array([r[3] for r in rows])
    r0 = np.array([r[4] for r in rows])

    plt.figure(figsize=(13.5,4.2))
    plt.subplot(1,3,1); plt.plot(K, r0, 'o-'); plt.title("Coherence vs K")
    plt.xlabel("K"); plt.ylabel("mean r (post-burn)")
    plt.subplot(1,3,2); plt.plot(K, F, 'o-'); plt.title("Flux vs K (invariant)")
    plt.xlabel("K"); plt.ylabel("F_triangle (rad)")
    plt.subplot(1,3,3); plt.plot(K, rho, 'o-'); plt.axhline(1.0, ls='--', c='gray')
    plt.title("Spectral radius vs K"); plt.xlabel("K"); plt.ylabel("max |λ|")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{tag}_grid_overview.png"), dpi=180); plt.close()

    plt.figure(figsize=(8.0,4.6))
    plt.plot(K, ang, 'o-'); plt.axhline(0, ls='--', c='gray'); plt.axhline(np.pi, ls='--', c='gray')
    plt.title("Eigenangle vs K"); plt.xlabel("K"); plt.ylabel("arg λ_max (rad)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{tag}_angle_vs_K.png"), dpi=180); plt.close()

# ---------------- NS detect & refine ----------------
def detect_ns_bracket(rows, ang_tol=0.15):
    """
    Find adjacent K points with |λ|-1 changing sign AND both angles away from 0 or π.
    Returns (K_lo, K_hi) or None.
    """
    rows = [r for r in rows if not np.isnan(r[1])]
    for a, b in zip(rows[:-1], rows[1:]):
        K1, ρ1, φ1 = a[0], a[1], a[2]
        K2, ρ2, φ2 = b[0], b[1], b[2]
        if (ρ1-1.0)*(ρ2-1.0) <= 0.0:
            ok1 = (abs((abs(φ1) % np.pi)) > ang_tol) and (abs((abs(φ1) % np.pi)) < np.pi-ang_tol)
            ok2 = (abs((abs(φ2) % np.pi)) > ang_tol) and (abs((abs(φ2) % np.pi)) < np.pi-ang_tol)
            if ok1 and ok2:
                return K1, K2
    return None

def bisection_refine(Klo, Khi, eval_fun, iters=24, ang_tol=0.15):
    """
    Bisect on K to keep a complex crossing (|λ|→1 with angle away from 0,π).
    eval_fun(K, seed_theta) -> (theta, rho, angle).
    """
    thL, ρL, φL = eval_fun(Klo, None)
    thH, ρH, φH = eval_fun(Khi, thL)
    seed = thH
    for _ in range(iters):
        Km = 0.5*(Klo+Khi)
        thM, ρM, φM = eval_fun(Km, seed)
        if thM is None: break
        seed = thM
        NS_ok = (abs((abs(φM) % np.pi)) > ang_tol) and (abs((abs(φM) % np.pi)) < np.pi-ang_tol)
        if (ρL-1.0)*(ρM-1.0) <= 0.0 and NS_ok:
            Khi, thH, ρH, φH = Km, thM, ρM, φM
        else:
            Klo, thL, ρL, φL = Km, thM, ρM, φM
    return 0.5*(Klo+Khi), max(ρL, ρH), 0.5*(φL+φH)

# ---------------- post-NS confirmation ----------------
def confirm_ns(outdir, Kc, family, Delta, kappa, alphas, deg_vec,
               dt, gamma, deg_norm=True,
               deltaK=0.03, T=20000, burn=15000,
               eps_init=1e-7, noise=0.0, seed=1):
    """
    Two-sided confirmation around an NS candidate:
      - Below K*: contraction back to fixed point.
      - Above K*: a thin invariant circle (phase portrait), persistent oscillations, FFT peak.
    """
    rng = np.random.default_rng(seed)
    n = 3

    def run_once(K, tag):
        th_star, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec, deg_norm)
        if not ok:
            print(f"[confirm] Newton failed at K={K:.6f} ({tag}).")
            return None
        th = th_star + eps_init * rng.standard_normal(n)
        if family == "inertial":
            vel = eps_init * rng.standard_normal(n)
        F_series, r_series, xy = [], [], []
        for t in range(T):
            if noise>0:
                dth = rng.normal(0.0, noise, size=n); th += dth
                if family == "inertial":
                    dvl = rng.normal(0.0, noise, size=n); vel += dvl
            if family == "memory":
                th = step_memory(th, K, Delta, kappa, alphas, deg_vec, dt, deg_norm)
            else:
                th, vel = step_inertial(th, vel, K, Delta, kappa, alphas, deg_vec, dt, gamma, deg_norm)
            if t >= burn:
                F_series.append(triangle_flux(th, alphas))
                r_series.append(np.abs(np.exp(1j*th).mean()))
                xy.append([wrap_pi(th[0]-th[1]), wrap_pi(th[1]-th[2])])
        return np.array(F_series), np.array(r_series), np.array(xy)

    below = run_once(Kc - deltaK, "below")
    above = run_once(Kc + deltaK, "above")
    if below is None or above is None: return

    F_L, r_L, xy_L = below
    F_R, r_R, xy_R = above

    # time series below/above
    t = np.arange(len(r_L))
    plt.figure(figsize=(11.5,4.5))
    plt.subplot(1,2,1); plt.plot(t, r_L); plt.title(f"Below: K={Kc-deltaK:.4f} (decay)")
    plt.xlabel("steps after burn"); plt.ylabel("r(t)")
    plt.subplot(1,2,2); plt.plot(t, r_R); plt.title(f"Above: K={Kc+deltaK:.4f} (torus)")
    plt.xlabel("steps after burn"); plt.ylabel("r(t)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "ns_confirm_series.png"), dpi=170); plt.close()

    # phase portrait above
    plt.figure(figsize=(5.3,5.0))
    plt.plot(xy_R[:,0], xy_R[:,1], lw=0.8)
    plt.title("Phase portrait above K*"); plt.xlabel("θ1−θ2"); plt.ylabel("θ2−θ3")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "ns_confirm_phase.png"), dpi=170); plt.close()

    # PSD of r(t) above
    rR = r_R - r_R.mean()
    R = np.fft.rfft(rR)
    freqs = np.fft.rfftfreq(len(rR), d=1.0)  # per step
    amp = np.abs(R)
    pk = np.argmax(amp[1:]) + 1
    f_meas_steps = freqs[pk]              # cycles/step
    f_meas = f_meas_steps / dt            # cycles per unit time

    # linear prediction at Kc
    thc, Om, ok = newton_locked(Kc, Delta, kappa, alphas, deg_vec, deg_norm)
    if ok:
        if family == "memory":
            lam = gauge_project_eigs_memory(jacobian_memory(thc, Kc, kappa, alphas, deg_vec, dt, deg_norm))
        else:
            lam = gauge_project_eigs_inertial(jacobian_inertial(thc, Kc, kappa, alphas, deg_vec, dt, gamma, deg_norm))
        lam1 = lam[np.argmax(np.abs(lam))]
        ang = float(np.angle(lam1))     # radians per step
        f_pred = (ang/(2*np.pi)) / dt   # cycles per time
        print(f"[confirm] Measured peak above K*: f≈{f_meas:.4f} vs linear prediction f≈{f_pred:.4f} (cycles/time)")

    # save PSD (per-step frequency on x-axis)
    plt.figure(figsize=(6.8,4.0))
    f_step = np.fft.rfftfreq(len(rR), d=1.0)
    plt.plot(f_step, amp)
    plt.title("PSD of r(t) above K*"); plt.xlabel("frequency [cycles/step]"); plt.ylabel("|FFT|")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "ns_confirm_psd.png"), dpi=170); plt.close()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["memory","inertial"], default="memory")
    ap.add_argument("--dt", type=float, default=0.20, help="map time-step")
    ap.add_argument("--gamma", type=float, default=0.55, help="damping for inertial family")
    ap.add_argument("--Delta", nargs=3, type=float, default=[0.01, 0.0, -0.01])
    ap.add_argument("--K_min", type=float, default=1.1)
    ap.add_argument("--K_max", type=float, default=2.1)
    ap.add_argument("--K_pts", type=int,   default=26)
    ap.add_argument("--F_target", type=float, default=2*np.pi/3)
    ap.add_argument("--directed", action="store_true", help="nonreciprocal edges (forward/backward 1±eps)")
    ap.add_argument("--eps_asym", type=float, default=0.0, help="fractional asymmetry if --directed")
    ap.add_argument("--deg_norm", action="store_true", help="degree-normalize coupling (default ON)")
    ap.add_argument("--no-deg_norm", dest="deg_norm", action="store_false")
    ap.set_defaults(deg_norm=True)

    # optional time-series demo at a single K
    ap.add_argument("--K_demo", type=float, default=None, help="if set, run a short trajectory at this K")
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--burn", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="additive Gaussian noise std (demo/confirm runs)")

    # post-NS confirmation tooling
    ap.add_argument("--post_ns", action="store_true",
                    help="after refine, run two-sided confirmation & diagnostics")
    ap.add_argument("--deltaK", type=float, default=0.03)
    ap.add_argument("--T_diag", type=int, default=40000)
    ap.add_argument("--burn_diag", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    outdir = ensure_dir(f"figs_ns_full_{time.strftime('%Y%m%d_%H%M%S')}")
    kappa, alphas, deg_vec, meta = build_k3(F_target=args.F_target,
                                            directed=args.directed,
                                            eps=args.eps_asym,
                                            degree_normalize=args.deg_norm)
    Delta = np.array(args.Delta, dtype=float)
    Ks = np.linspace(args.K_min, args.K_max, args.K_pts)

    print(f"Target F={args.F_target:.6f} rad, family={args.family}, dt={args.dt}, "
          f"gamma={args.gamma if args.family=='inertial' else '–'}, Delta={Delta}, "
          f"directed={args.directed}, eps={args.eps_asym}, deg_norm={args.deg_norm}")

    rows = []
    seed_theta = None

    # ----- sweep -----
    for K in Ks:
        th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec, args.deg_norm, theta0=seed_theta, Omega0=0.0)
        if not ok:
            rows.append([K, np.nan, np.nan, np.nan, np.nan, "newton-fail"])
            continue
        seed_theta = th

        if args.family == "memory":
            J = jacobian_memory(th, K, kappa, alphas, deg_vec, args.dt, args.deg_norm)
            lam = gauge_project_eigs_memory(J)
        else:
            J = jacobian_inertial(th, K, kappa, alphas, deg_vec, args.dt, args.gamma, args.deg_norm)
            lam = gauge_project_eigs_inertial(J)

        # leading eigenvalue (nongauge subspace)
        idx = np.argmax(np.abs(lam))
        lam1 = lam[idx]
        rho  = float(np.abs(lam1))
        ang  = float(np.angle(lam1))
        Ftri = float(triangle_flux(th, alphas))
        rbar = float(np.abs(np.exp(1j*th).mean()))

        # light-touch classification
        ang_mod = abs((abs(ang) % np.pi))
        if abs(ang_mod - np.pi) < 1e-2 or ang_mod < 1e-2:
            kind = "real"
        else:
            kind = "complex" if rho <= 1.0 else "NS-candidate"

        rows.append([K, rho, ang, Ftri, rbar, kind])

        if args.debug:
            top3 = ", ".join([f"{x.real:+.6f}{x.imag:+.6f}i" for x in lam[np.argsort(-np.abs(lam))[:3]]])
            print(f"K={K:7.3f} | |λ|max={rho:.6f}, arg={ang:+.3f} rad | {kind:12s} | "
                  f"r={rbar:.3f} | F={Ftri:.6f} | top={top3}")
        else:
            print(f"K={K:6.3f} | |λ|max={rho:.6f}, arg={ang:+.3f} rad | r_mean={rbar:.3f} | F={Ftri:.6f} | {kind}")

    tag = f"{args.family}_dt{args.dt:.3f}{'_dir' if args.directed else ''}"
    write_csv(os.path.join(outdir,"k_sweep.csv"),
              rows, header=["K","rho","angle","F_triangle","r(theta*)","type"])
    quick_grid_plots(outdir, rows, tag)

    # ----- detect & refine -----
    br = detect_ns_bracket(rows, ang_tol=0.15)
    if br:
        Klo, Khi = br
        print(f"\nNS-like crossing bracketed in [{Klo:.6f}, {Khi:.6f}] … refining")

        def _eval(K, seed_theta_in):
            th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec, args.deg_norm,
                                       theta0=seed_theta_in, Omega0=0.0)
            if not ok: return None, None, None
            if args.family == "memory":
                lam = gauge_project_eigs_memory(jacobian_memory(th, K, kappa, alphas, deg_vec, args.dt, args.deg_norm))
            else:
                lam = gauge_project_eigs_inertial(jacobian_inertial(th, K, kappa, alphas, deg_vec, args.dt, args.gamma, args.deg_norm))
            lam1 = lam[np.argmax(np.abs(lam))]
            return th, float(np.abs(lam1)), float(np.angle(lam1))

        Kc, rhoc, phic = bisection_refine(Klo, Khi, _eval, iters=24, ang_tol=0.15)
        print(f"\nRefine result: K*={Kc:.9f} | rho={rhoc:.6f} | angle={phic:+.6f} rad  [NS-candidate]\n")

        # optional post-NS confirmation (two-sided)
        if args.post_ns:
            confirm_ns(outdir, Kc, args.family, Delta, kappa, alphas, deg_vec,
                       args.dt, args.gamma, args.deg_norm,
                       deltaK=args.deltaK, T=args.T_diag, burn=args.burn_diag,
                       eps_init=1e-7, noise=args.noise, seed=args.seed)

            # transversality slope d|λ|max/dK at K*
            def leading_rho(K):
                th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec, args.deg_norm)
                if not ok: return np.nan
                if args.family == "memory":
                    lam = gauge_project_eigs_memory(jacobian_memory(th, K, kappa, alphas, deg_vec, args.dt, args.deg_norm))
                else:
                    lam = gauge_project_eigs_inertial(jacobian_inertial(th, K, kappa, alphas, deg_vec, args.dt, args.gamma, args.deg_norm))
                return float(np.max(np.abs(lam)))

            epsK = 5e-3
            slope = (leading_rho(Kc+epsK) - leading_rho(Kc-epsK)) / (2*epsK)
            print(f"[check] d|λ|max/dK at K* ≈ {slope:.3e} (nonzero supports transversality)")

    else:
        print("\nNo NS bracket found on this grid. "
              "Either a flip (real at ±1) precedes NS for this family/discretization, "
              "or the leading complex pair stays within |λ|<1 across the window.\n")

    # ----- optional single-K demo -----
    if args.K_demo is not None:
        K = float(args.K_demo)
        th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec, args.deg_norm)
        if ok:
            T = args.T; burn = args.burn
            if args.family == "memory":
                theta = th.copy()
                F_series = []
                for t in range(T):
                    if args.noise>0: theta += np.random.normal(0.0, args.noise, size=theta.shape)
                    theta = step_memory(theta, K, Delta, kappa, alphas, deg_vec, args.dt, args.deg_norm)
                    if t>=burn: F_series.append(triangle_flux(theta, alphas))
                plot_flux_su2(outdir, F_series)
            else:
                theta = th.copy(); vel = np.zeros_like(theta)
                F_series = []
                for t in range(T):
                    if args.noise>0:
                        theta += np.random.normal(0.0, args.noise, size=theta.shape)
                        vel   += np.random.normal(0.0, args.noise, size=vel.shape)
                    theta, vel = step_inertial(theta, vel, K, Delta, kappa, alphas, deg_vec,
                                               args.dt, args.gamma, args.deg_norm)
                    if t>=burn: F_series.append(triangle_flux(theta, alphas))
                plot_flux_su2(outdir, F_series)

    print(f"\nAll outputs saved in: {outdir}")
    print("(PNGs + CSV; nothing is shown interactively.)")

if __name__ == "__main__":
    main()
