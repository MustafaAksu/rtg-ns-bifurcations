#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rtg_kuramoto_ns_probes.py
=========================
K3 with frustrated flux; NS-enabled probes for both first-order ('memory')
and second-order ('inertial') families.

Features
--------
• Phase-lags α with sum F_target (U(1) flux baked into the couplings)
• 'memory' model: Newton solve for (θ*,Ω), 3×3 one-step Jacobian, gauge projection
• 'inertial' model: Newton solve for (θ*,Ω) (same equations), 6×6 one-step Jacobian,
  gauge projection (remove the global phase mode)
• K-sweep, automatic NS bracket detection (|λ|max crosses 1 with angle ∉ {0,π}),
  and bisection refinement
• Headless plots + CSV logs
• Optional tiny noise for long time-series demos (not needed for eigen probes)

Run examples
------------
# Memory (first-order) – reproduces NS-candidate near K≈2.02 on your machine
python rtg_kuramoto_ns_probes.py --model memory \
  --Delta 0.01 0.0 -0.01 --K_min 1.1 --K_max 2.1 --K_pts 26

# Inertial (second-order) – proper 6×6 Jacobian with damping gamma
python rtg_kuramoto_ns_probes.py --model inertial --gamma 0.55 \
  --Delta 0.01 0.0 -0.01 --K_min 1.6 --K_max 2.2 --K_pts 41
"""

import os, csv, math, time, argparse
import numpy as np
import numpy.linalg as LA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU = 2*np.pi

# -------------------- utilities --------------------
def wrap_pi(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def basis_gauge_free(n):
    """Columns span the subspace orthogonal to the all-ones vector in R^n."""
    B = np.eye(n)[:, :-1] - np.eye(n)[:, [-1]]  # e_k - e_n
    Q, _ = np.linalg.qr(B)
    return Q  # (n, n-1)

def write_csv(path, rows, header=None):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        w.writerows(rows)

# -------------------- graph + flux --------------------
def build_k3(F_target=2*np.pi/3, directed=False, eps=0.0, degree_normalize=True):
    """
    K3 with (optionally) directed asymmetry 'eps' on cycle 0→1→2→0.
    The phase-lag matrix 'alphas' satisfies sum of lags on that oriented triangle = F_target.
    """
    n = 3
    A = np.ones((n,n)) - np.eye(n)

    # Coupling matrix
    kappa = A.copy()
    if directed:
        # forward edges get 1+eps, reverse 1-eps
        Kf, Kr = 1.0+eps, 1.0-eps
        kappa[:] = 0.0
        kappa[0,1]=Kf; kappa[1,2]=Kf; kappa[2,0]=Kf
        kappa[1,0]=Kr; kappa[2,1]=Kr; kappa[0,2]=Kr

    # Phase-lags α – equal share of F_target on the forward cycle, opposite on reverse
    alphas = np.zeros_like(kappa)
    a = F_target/3.0
    alphas[0,1]=a; alphas[1,2]=a; alphas[2,0]=a
    alphas[1,0]=-a; alphas[2,1]=-a; alphas[0,2]=-a

    deg_vec = kappa.sum(axis=1)  # for degree normalization (optional)
    meta = dict(F_target=F_target, directed=directed, eps=eps,
                degree_normalize=degree_normalize)
    return kappa, alphas, deg_vec, meta

def triangle_flux(theta, alphas, cycle=(0,1,2,0)):
    """Gauge-invariant U(1) flux: sum of (θ_j-θ_i-α_ij) on an oriented cycle."""
    F = 0.0
    for i,j in zip(cycle[:-1], cycle[1:]):
        F += wrap_pi(theta[j] - theta[i] - alphas[i,j])
    return wrap_pi(F)

def su2_half_trace_from_flux(F):
    """0.5 * Tr(H_su2) = cos(F/2) – SU(2) trace–angle identity."""
    return float(np.cos(0.5*F))

# -------------------- models: step maps --------------------
def step_memory(theta, K, Delta, kappa, alphas, deg_vec, degree_normalize=True,
                delay_state=None, use_delay=False, noise_sigma=0.0, rng=None):
    """First-order Kuramoto map with phase lags."""
    n = len(theta)
    w = deg_vec if degree_normalize else np.ones(n)
    thetaj = delay_state if use_delay else theta
    d = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i==j: continue
            s += kappa[i,j]*np.sin(thetaj[j] - theta[i] - alphas[i,j])
        d[i] = Delta[i] + (K/w[i])*s
    new = theta + d
    if (noise_sigma>0) and (rng is not None):
        new = new + rng.normal(scale=noise_sigma, size=n)
    return new

def step_inertial(theta, vel, K, Delta, kappa, alphas, deg_vec, gamma,
                  degree_normalize=True, noise_sigma=0.0, rng=None):
    """
    Second-order (position–velocity) update:
      v' = (1-γ) v + Δ + (K/deg_i) Σ κ_ij sin(θ_j - θ_i - α_ij)
      θ' = θ + v'
    Use in a rotating frame (Ω removed by Newton) so the locked state has v=0.
    """
    n = len(theta)
    w = deg_vec if degree_normalize else np.ones(n)
    acc = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i==j: continue
            s += kappa[i,j]*np.sin(theta[j] - theta[i] - alphas[i,j])
        acc[i] = Delta[i] + (K/w[i])*s
    vnew = (1.0 - gamma)*vel + acc
    thetanew = theta + vnew
    if (noise_sigma>0) and (rng is not None):
        thetanew = thetanew + rng.normal(scale=noise_sigma, size=n)
    return thetanew, vnew

# -------------------- Jacobians (one-step) --------------------
def jacobian_memory(theta, K, kappa, alphas, deg_vec, degree_normalize=True):
    """3×3 Jacobian of the one-step map for the memory model at theta."""
    n = len(theta)
    w = deg_vec if degree_normalize else np.ones(n)
    J = np.zeros((n,n))
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i==j: continue
            c = kappa[i,j]*np.cos(theta[j] - theta[i] - alphas[i,j])
            row_sum += c
            J[i,j] = (K/w[i])*c
        J[i,i] = 1.0 - (K/w[i])*row_sum
    return J

def jacobian_inertial(theta, K, kappa, alphas, deg_vec, gamma, degree_normalize=True):
    """
    6×6 Jacobian of the inertial map at state (θ*, v*=0):
      v' = a v + M δθ
      θ' = θ + v' = (I+M) δθ + a δv
    where a=(1-γ), and M is the n×n linearization of the coupling term.
    """
    n = len(theta)
    a = 1.0 - gamma
    w = deg_vec if degree_normalize else np.ones(n)

    # Build M
    M = np.zeros((n,n))
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i==j: continue
                # cosine gain at equilibrium
            c = kappa[i,j]*np.cos(theta[j] - theta[i] - alphas[i,j])
            M[i,j] = (K/w[i])*c
            row_sum += c
        M[i,i] = -(K/w[i])*row_sum

    # Assemble block Jacobian on [δθ; δv]
    J = np.zeros((2*n, 2*n))
    # δθ' = (I+M) δθ + a δv
    J[:n, :n] = np.eye(n) + M
    J[:n, n:] = a * np.eye(n)
    # δv' = M δθ + a δv
    J[n:, :n] = M
    J[n:, n:] = a * np.eye(n)
    return J

def gauge_project_eigs_memory(J):
    """Remove the neutral global-phase mode (3×3 → 2×2)."""
    Q = basis_gauge_free(J.shape[0])      # (3,2)
    Jg = Q.T @ J @ Q
    return LA.eigvals(Jg)

def gauge_project_eigs_inertial(J):
    """
    Remove the neutral phase mode only. Build block‑diagonal projector:
      Q2 = diag(Qθ, I_n), where Qθ is (n, n-1).
    """
    n2 = J.shape[0] // 2
    Qθ = basis_gauge_free(n2)             # (3,2)
    Q2 = np.zeros((2*n2, (n2-1)+n2))
    # top-left block (θ): Qθ
    Q2[:n2, :n2-1] = Qθ
    # bottom-right block (v): identity
    Q2[n2:, n2-1:] = np.eye(n2)
    Jg = Q2.T @ J @ Q2
    return LA.eigvals(Jg)

# -------------------- Newton for the locked state --------------------
def locked_residual(theta, Omega, K, Delta, kappa, alphas, deg_vec, degree_normalize=True):
    """
    Equations for a phase‑locked solution with common frequency Ω:
      Δ_i + (K/deg_i) Σ κ_ij sin(θ_j-θ_i-α_ij) - Ω = 0,  i=1..n
      Σ_i θ_i = 0  (gauge)
    """
    n = len(theta)
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
    r[n]  = theta.sum()     # gauge fix
    return r

def locked_jacobian(theta, K, kappa, alphas, deg_vec, degree_normalize=True):
    """Jacobian of the residual system (n+1)×(n+1)."""
    n = len(theta)
    w = deg_vec if degree_normalize else np.ones(n)
    G = np.zeros((n,n))
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i==j: continue
            c = kappa[i,j]*np.cos(theta[j] - theta[i] - alphas[i,j])
            G[i,j] = (K/w[i])*c
            row_sum += c
        G[i,i] = -(K/w[i])*row_sum
    J = np.zeros((n+1, n+1))
    J[:n, :n] = G
    J[:n, n]  = -1.0      # ∂g/∂Ω
    J[n, :n]  =  1.0      # Σθ_i
    J[n, n]   =  0.0
    return J

def newton_locked(K, Delta, kappa, alphas, deg_vec, degree_normalize=True,
                  theta0=None, Omega0=0.0, tol=1e-12, maxit=80):
    """Newton solve for (θ*,Ω)."""
    n = kappa.shape[0]
    theta = np.zeros(n) if theta0 is None else theta0.astype(float).copy()
    Omega = float(Omega0)
    for _ in range(maxit):
        r = locked_residual(theta, Omega, K, Delta, kappa, alphas, deg_vec, degree_normalize)
        if LA.norm(r, ord=np.inf) < tol:
            return theta, Omega, True
        J = locked_jacobian(theta, K, kappa, alphas, deg_vec, degree_normalize)
        try:
            step = LA.solve(J, -r)
        except LA.LinAlgError:
            return theta, Omega, False
        theta += step[:-1]; Omega += step[-1]
        theta = wrap_pi(theta)
    return theta, Omega, False

# -------------------- sweeps, detection, refinement --------------------
def sweep_memory(outdir, K_vals, Delta, kappa, alphas, deg_vec, tag="memory"):
    rows = []
    seed = None
    for K in K_vals:
        th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec,
                                   degree_normalize=True, theta0=seed, Omega0=0.0)
        if not ok:
            rows.append([K, np.nan, np.nan, np.nan, np.nan, "fail"])
            continue
        seed = th
        J = jacobian_memory(th, K, kappa, alphas, deg_vec, True)
        lam = gauge_project_eigs_memory(J)
        lam1 = lam[np.argmax(np.abs(lam))]
        rho  = float(np.abs(lam1))
        ang  = float(np.angle(lam1))
        F    = float(triangle_flux(th, alphas))
        rbar = float(np.abs(np.exp(1j*th).mean()))
        label = "real-neg" if np.isclose(abs(abs(ang)%np.pi), np.pi, atol=1e-2) else "complex"
        rows.append([K, rho, ang, F, rbar, label])

    write_csv(os.path.join(outdir, f"{tag}_k_sweep.csv"),
              rows, header=["K","rho","angle","F_triangle","r(theta*)","type"])

    _quick_plots(outdir, rows, tag)
    return rows

def sweep_inertial(outdir, K_vals, Delta, kappa, alphas, deg_vec, gamma, tag="inertial"):
    """
    Linear stability of the inertial one-step map AT the locked configuration (θ*,Ω).
    Note: Newton for (θ*,Ω) is the same as in the memory case (model-independent).
    """
    rows = []
    seed = None
    for K in K_vals:
        th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec,
                                   degree_normalize=True, theta0=seed, Omega0=0.0)
        if not ok:
            rows.append([K, np.nan, np.nan, np.nan, np.nan, "fail"])
            continue
        seed = th
        J = jacobian_inertial(th, K, kappa, alphas, deg_vec, gamma, True)
        lam = gauge_project_eigs_inertial(J)
        lam1 = lam[np.argmax(np.abs(lam))]
        rho  = float(np.abs(lam1))
        ang  = float(np.angle(lam1))
        F    = float(triangle_flux(th, alphas))
        rbar = float(np.abs(np.exp(1j*th).mean()))
        label = "real-neg" if np.isclose(abs(abs(ang)%np.pi), np.pi, atol=1e-2) else "complex"
        rows.append([K, rho, ang, F, rbar, label])

    write_csv(os.path.join(outdir, f"{tag}_k_sweep.csv"),
              rows, header=["K","rho","angle","F_triangle","r(theta*)","type"])

    _quick_plots(outdir, rows, tag)
    return rows

def _quick_plots(outdir, rows, tag):
    rows = [r for r in rows if not np.isnan(r[1])]
    if not rows: return
    K  = np.array([r[0] for r in rows])
    ρ  = np.array([r[1] for r in rows])
    φ  = np.array([r[2] for r in rows])
    F  = np.array([r[3] for r in rows])
    r̄  = np.array([r[4] for r in rows])

    plt.figure(figsize=(5,4))
    plt.plot(K, φ, 'o-'); plt.axhline(0, ls='--', c='gray'); plt.axhline(np.pi, ls='--', c='gray')
    plt.title(f"{tag}: arg λ_max vs K"); plt.xlabel("K"); plt.ylabel("arg λ_max (rad)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{tag}_angle_vs_K.png"), dpi=160); plt.close()

    plt.figure(figsize=(5,4))
    plt.plot(K, ρ, 'o-'); plt.axhline(1.0, ls='--', c='gray')
    plt.title(f"{tag}: max |λ| vs K"); plt.xlabel("K"); plt.ylabel("max |λ|")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{tag}_radius_vs_K.png"), dpi=160); plt.close()

    plt.figure(figsize=(5,4))
    plt.plot(K, F, 'o-')
    plt.title(f"{tag}: flux vs K (should be invariant)"); plt.xlabel("K"); plt.ylabel("F_triangle (rad)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{tag}_flux_vs_K.png"), dpi=160); plt.close()

    plt.figure(figsize=(5,4))
    plt.plot(K, r̄, 'o-')
    plt.title(f"{tag}: r(theta*) vs K"); plt.xlabel("K"); plt.ylabel("r")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{tag}_r_vs_K.png"), dpi=160); plt.close()

def detect_ns_bracket(rows, ang_tol=0.2):
    rows = [r for r in rows if not np.isnan(r[1])]
    for a, b in zip(rows[:-1], rows[1:]):
        K1, ρ1, φ1 = a[0], a[1], a[2]
        K2, ρ2, φ2 = b[0], b[1], b[2]
        ang_ok_1 = (abs((abs(φ1)%np.pi)) > ang_tol)
        ang_ok_2 = (abs((abs(φ2)%np.pi)) > ang_tol)
        if ang_ok_1 and ang_ok_2 and (ρ1-1.0)*(ρ2-1.0) <= 0.0:
            return K1, K2
    return None

def bisection_refine(Klo, Khi, eval_fun, iters=24, ang_tol=0.2):
    """
    eval_fun(K, seed_state) → (theta*, rho, angle)
    Uses Newton inside eval_fun. Keeps the “complex-ang” bracket while |λ|-crosses.
    """
    seed = None
    thL, ρL, φL = eval_fun(Klo, seed); seed = thL
    thH, ρH, φH = eval_fun(Khi, seed); seed = thH
    for _ in range(iters):
        Km = 0.5*(Klo+Khi)
        thM, ρM, φM = eval_fun(Km, seed)
        if thM is None: break
        seed = thM
        if (ρL-1.0)*(ρM-1.0) <= 0.0 and (abs((abs(φM)%np.pi))>ang_tol):
            Khi, thH, ρH, φH = Km, thM, ρM, φM
        else:
            Klo, thL, ρL, φL = Km, thM, ρM, φM
    return 0.5*(Klo+Khi), max(ρL, ρH), 0.5*(φL+φH)

# -------------------- CLI / main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["memory","inertial"], default="memory")
    ap.add_argument("--gamma", type=float, default=0.55, help="damping for inertial model")
    ap.add_argument("--Delta", nargs=3, type=float, default=[0.01, 0.0, -0.01])
    ap.add_argument("--K_min", type=float, default=1.1)
    ap.add_argument("--K_max", type=float, default=2.1)
    ap.add_argument("--K_pts", type=int,   default=26)
    ap.add_argument("--F_target", type=float, default=2*np.pi/3)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--eps_asym", type=float, default=0.0, help="directed asymmetry if --directed")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = ensure_dir(f"figs_ns_probes_{stamp}")

    kappa, alphas, deg_vec, meta = build_k3(F_target=args.F_target,
                                            directed=args.directed,
                                            eps=args.eps_asym,
                                            degree_normalize=True)
    Delta = np.array(args.Delta, dtype=float)
    Ks = np.linspace(args.K_min, args.K_max, args.K_pts)

    print(f"Model: {args.model} | gamma={args.gamma if args.model=='inertial' else '–'} "
          f"| Delta={Delta} | target F={args.F_target:.6f}")

    if args.model == "memory":
        rows = sweep_memory(outdir, Ks, Delta, kappa, alphas, deg_vec, tag="memory")
        bracket = detect_ns_bracket(rows)
        if bracket:
            Klo, Khi = bracket
            # refinement callback
            def _eval(K, seed):
                th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec, True,
                                           theta0=seed, Omega0=0.0)
                if not ok: return None, None, None
                J = jacobian_memory(th, K, kappa, alphas, deg_vec, True)
                lam = gauge_project_eigs_memory(J)
                lam1 = lam[np.argmax(np.abs(lam))]
                return th, float(np.abs(lam1)), float(np.angle(lam1))
            Kc, ρc, φc = bisection_refine(Klo, Khi, _eval, iters=24)
            print(f"\nRefine result (memory): K*={Kc:.6f} | rho={ρc:.6f} | angle={φc:+.4f} rad  [NS-candidate]\n")
        else:
            print("\nNo NS bracket found on this grid (memory). "
                  "This still classifies the first instability in this family.\n")

    else:  # inertial
        rows = sweep_inertial(outdir, Ks, Delta, kappa, alphas, deg_vec, gamma=args.gamma, tag="inertial")
        bracket = detect_ns_bracket(rows)
        if bracket:
            Klo, Khi = bracket
            def _eval(K, seed):
                th, Om, ok = newton_locked(K, Delta, kappa, alphas, deg_vec, True,
                                           theta0=seed, Omega0=0.0)
                if not ok: return None, None, None
                J = jacobian_inertial(th, K, kappa, alphas, deg_vec, args.gamma, True)
                lam = gauge_project_eigs_inertial(J)
                lam1 = lam[np.argmax(np.abs(lam))]
                return th, float(np.abs(lam1)), float(np.angle(lam1))
            Kc, ρc, φc = bisection_refine(Klo, Khi, _eval, iters=24)
            print(f"\nRefine result (inertial): K*={Kc:.6f} | rho={ρc:.6f} | angle={φc:+.4f} rad  [NS-candidate]\n")
        else:
            print("\nNo NS bracket found on this grid (inertial). "
                  "Either flip occurs first or the leading complex pair stays inside |λ|<1.\n")

    print(f"All outputs saved in: {outdir}")

if __name__ == "__main__":
    main()
