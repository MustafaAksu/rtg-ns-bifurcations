# rtg_kuramoto_ns_enabled.py
# Discrete Kuramoto/Sakaguchi on K3 with directed couplings (eps) -> NS bifurcation.
# Saves plots and CSV; no interactive windows.

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import eig

TAU = 2*np.pi

# ---------- utilities ----------
def mod2pi(x):
    y = (x + np.pi) % (2*np.pi) - np.pi
    if isinstance(y, np.ndarray):
        y[y == -np.pi] = np.pi
    elif y == -np.pi:
        y = np.pi
    return y

def order_param(theta):
    return np.abs(np.mean(np.exp(1j*theta)))

def make_directed_k3(eps=0.60):
    """Nonreciprocal weights: along 0→1→2→0 is 1+eps, reverse is 1−eps."""
    K = np.zeros((3,3))
    K[0,1] = 1+eps; K[1,0] = 1-eps
    K[1,2] = 1+eps; K[2,1] = 1-eps
    K[2,0] = 1+eps; K[0,2] = 1-eps
    np.fill_diagonal(K, 0.0)
    deg_out = np.sum(np.abs(K), axis=1)
    return K, deg_out

def make_alpha(phi=2*np.pi/9):
    """Phase‑lag matrix with net flux 3φ on 0→1→2→0; reverse gets −φ."""
    A = np.zeros((3,3))
    A[0,1] = +phi; A[1,0] = -phi
    A[1,2] = +phi; A[2,1] = -phi
    A[2,0] = +phi; A[0,2] = -phi
    return A

def triangle_flux(theta, A):
    a01 = mod2pi(theta[1] - theta[0] - A[0,1])
    a12 = mod2pi(theta[2] - theta[1] - A[1,2])
    a20 = mod2pi(theta[0] - theta[2] - A[2,0])
    return mod2pi(a01 + a12 + a20)

# ---------- model (map) ----------
def step_map(theta, Kc, W, deg_out, A, Delta):
    """θᵢ(t+1) = θᵢ + Δᵢ + (K/deg_out(i)) Σⱼ Wᵢⱼ sin(θⱼ−θᵢ−Aᵢⱼ)"""
    n = len(theta)
    inc = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j: 
                continue
            s += W[i,j] * np.sin(theta[j] - theta[i] - A[i,j])
        inc[i] = Delta[i] + (Kc/deg_out[i]) * s
    return mod2pi(theta + inc)

def iterate(theta0, Kc, W, deg_out, A, Delta, T=30000, burn=25000):
    th = np.array(theta0, dtype=float)
    post, rts = [], []
    for t in range(T):
        th = step_map(th, Kc, W, deg_out, A, Delta)
        if t >= burn:
            post.append(th.copy())
            rts.append(order_param(th))
    return np.array(post), np.array(rts)

# ---------- Jacobian of the map ----------
def jacobian_map(th, Kc, W, deg_out, A):
    """Jacobian M = ∂θ′/∂θ at θ for the map (synchronous update)."""
    n = len(th)
    M = np.eye(n)
    for i in range(n):
        csum = 0.0
        for j in range(n):
            if i == j: 
                continue
            c = np.cos(th[j] - th[i] - A[i,j])
            M[i,j] += (Kc/deg_out[i]) * W[i,j] * c
            csum += W[i,j] * c
        M[i,i] -= (Kc/deg_out[i]) * csum
    return M

def project_nongauge(M):
    """Project to 2D difference coordinates x=[θ1−θ0, θ2−θ0]."""
    A = np.array([[-1, 1, 0],
                  [-1, 0, 1]], float)      # 2x3
    B = np.array([[0, 0],
                  [1, 0],
                  [0, 1]], float)          # 3x2 (anchor δθ0=0)
    return A @ M @ B                       # 2x2

def detect_ns(Ks, radii, angles, angle_tol=0.15):
    """Find first index where |λ| crosses 1 upward with angle not ~0 or π."""
    Ks = np.asarray(Ks); radii = np.asarray(radii); ang = np.unwrap(np.asarray(angles))
    cand = np.where((radii[:-1] < 1.0) & (radii[1:] >= 1.0))[0]
    for i in cand:
        a = np.mod(abs(0.5*(ang[i]+ang[i+1])), np.pi)
        if a > angle_tol and abs(a - np.pi) > angle_tol:
            return i, 0.5*(Ks[i]+Ks[i+1])
    return None, None

# ---------- main ----------
def main():
    outdir = "figs_ns_enabled"
    os.makedirs(outdir, exist_ok=True)

    # You can adjust these two to make the NS crossing obvious:
    eps = 0.60                  # ↑ eps => stronger nonreciprocity => easier NS
    Delta = np.array([+0.04, 0.0, -0.04])  # tiny heterogeneity optional

    W, deg_out = make_directed_k3(eps=eps)
    A = make_alpha(phi=2*np.pi/9)   # fixes F_triangle = 2π/3 (mod 2π)
    rng = np.random.default_rng(1)
    theta0 = rng.uniform(-np.pi, np.pi, size=3)

    Ks = np.linspace(1.55, 2.15, 31)   # refine around the crossing if needed
    radii, angles, r_means, F_means = [], [], [], []

    print("\n=== K-sweep (directed K3 with flux) ===")
    for Kc in Ks:
        traj, rs = iterate(theta0, Kc, W, deg_out, A, Delta, T=30000, burn=25000)
        th = traj[-1]
        M = jacobian_map(th, Kc, W, deg_out, A)
        J = project_nongauge(M)
        ew = eig(J)[0]
        lam = ew[np.argmax(np.abs(ew))]
        radii.append(np.abs(lam))
        angles.append(np.angle(lam))
        r_means.append(np.mean(rs))
        F_vals = np.array([triangle_flux(x, A) for x in traj])
        F_means.append(np.mean(F_vals))
        print(f"K={Kc:4.2f} | |λ|max={np.abs(lam):.4f}, arg={np.angle(lam):+.3f} rad | "
              f"r_mean={np.mean(rs):.3f} | F_mean={np.mean(F_vals):+.6f} rad")

    # CSV
    csv_path = os.path.join(outdir, "ns_k_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["K","abs_lam_max","arg_lam_max","r_mean","F_mean"])
        for i in range(len(Ks)):
            w.writerow([f"{Ks[i]:.6f}", f"{radii[i]:.6f}", f"{angles[i]:.6f}",
                        f"{r_means[i]:.6f}", f"{F_means[i]:.6f}"])
    print(f"[CSV] wrote {csv_path}")

    # detect crossing
    idx, K_ns = detect_ns(Ks, radii, angles, angle_tol=0.15)
    if K_ns is None:
        print("\nNo clear NS crossing detected on this grid. "
              "Increase eps (e.g. 0.45–0.50) or refine Ks near where |λ|→1.")
        K_show = Ks[-1]
    else:
        print(f"\nEstimated NS at K ≈ {K_ns:.3f} (index {idx}→{idx+1}).")
        K_show = Ks[min(idx+1, len(Ks)-1)]

    # plots (saved, not shown)
    plt.figure(figsize=(5,4))
    plt.plot(Ks, radii, marker="o", label="max |λ| (nongauge)")
    plt.axhline(1.0, ls="--", c="k", alpha=0.6)
    if K_ns is not None: plt.axvline(K_ns, ls="--", c="r", alpha=0.7, label=f"NS ~ {K_ns:.2f}")
    plt.xlabel("K"); plt.ylabel("max |λ|"); plt.title("Spectral radius vs K"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "sweep_radius_vs_K.png"))

    plt.figure(figsize=(5,4))
    angu = np.unwrap(angles)
    plt.plot(Ks, angu, marker="o")
    plt.axhline(0.0, ls="--", c="k", alpha=0.4); plt.axhline(np.pi, ls="--", c="k", alpha=0.4)
    if K_ns is not None: plt.axvline(K_ns, ls="--", c="r", alpha=0.7)
    plt.xlabel("K"); plt.ylabel("arg λ_max (rad)"); plt.title("Eigenangle vs K")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "sweep_angle_vs_K.png"))

    plt.figure(figsize=(5,4))
    plt.plot(Ks, r_means, marker="o")
    plt.xlabel("K"); plt.ylabel("mean r"); plt.title("Coherence vs K")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "sweep_rmean_vs_K.png"))

    plt.figure(figsize=(5,4))
    plt.plot(Ks, F_means, marker="o")
    plt.xlabel("K"); plt.ylabel("F_triangle (rad)"); plt.title("Flux vs K (invariant)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "sweep_flux_vs_K.png"))

    # Post‑NS run (or last K if none detected)
    traj, rs = iterate(theta0, K_show, W, deg_out, A, Delta, T=35000, burn=30000)
    F_series = np.array([triangle_flux(x, A) for x in traj])
    su2_series = np.cos(0.5*F_series)
    print(f"\nPost‑NS demo at K={K_show:.3f}: "
          f"mean r={np.mean(rs):.3f},  F_mean={np.mean(F_series):+.6f} rad, "
          f"0.5 Tr(H) mean={np.mean(su2_series):.6f}, cos(F/2) mean={np.mean(su2_series):.6f}")

    plt.figure(figsize=(6,3.4))
    plt.plot(rs); plt.xlabel("steps after burn-in"); plt.ylabel("r(t)")
    plt.title(f"Order parameter r(t) at K={K_show:.2f}")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "rt_series_postNS.png"))

    plt.figure(figsize=(8,3.4))
    plt.subplot(1,2,1); plt.plot(F_series); plt.xlabel("steps"); plt.ylabel("F (rad)")
    plt.title("U(1) flux during run")
    plt.subplot(1,2,2); plt.plot(np.cos(0.5*F_series), label="cos(F/2)")
    plt.title("SU(2) trace–angle check"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "flux_su2_postNS.png"))

    print(f"\nAll outputs saved in: {outdir}\n")

if __name__ == "__main__":
    main()
