# rtg_kuramoto_k3_lag.py
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
def mod2pi(x):
    """Map angles to (-pi, pi], handling scalars and arrays."""
    y = (x + np.pi) % (2*np.pi) - np.pi
    if np.ndim(y) == 0:
        val = float(y)
        return np.pi if val == -np.pi else val
    y = y.astype(float, copy=False)
    y[y == -np.pi] = np.pi
    return y

def order_parameter_t(theta_t):
    z = np.exp(1j * theta_t)     # (T, N)
    return np.abs(z.mean(axis=1))

def su2_from_flux(F):
    H = np.array([[np.exp(1j*F/2), 0.0j],
                  [0.0j, np.exp(-1j*F/2)]], dtype=complex)
    tr_half = 0.5 * (H[0,0] + H[1,1]).real
    return H, tr_half

# -----------------------------
# Discrete Sakaguchi–Kuramoto on K3
# theta_i(t+1) = theta_i + Delta_i + (K/deg(i)) sum_j sin(theta_j - theta_i - alpha_ij)
# -----------------------------
def kuramoto_map_step_lag(theta, K, alpha, degrees, delta=None):
    th = np.asarray(theta, dtype=float)
    N = th.shape[0]
    if delta is None:
        delta = np.zeros(N, dtype=float)
    new = th.copy()
    for i in range(N):
        s = 0.0
        for j in range(N):
            if i == j: 
                continue
            if not np.isfinite(alpha[i, j]):  # allow masked edges if desired
                continue
            s += np.sin(th[j] - th[i] - alpha[i, j])
        new[i] = th[i] + delta[i] + (K / degrees[i]) * s
    return mod2pi(new)

def holonomy_flux_U1_with_alpha(theta, alpha):
    """
    U(1) flux using the connection A_ij = theta_j - theta_i - alpha_ij
    on the triangle (0,1,2), oriented 0->1->2->0.
    """
    th = np.asarray(theta, dtype=float)
    A01 = mod2pi(th[1] - th[0] - alpha[0,1])
    A12 = mod2pi(th[2] - th[1] - alpha[1,2])
    A20 = mod2pi(th[0] - th[2] - alpha[2,0])
    F = mod2pi(A01 + A12 + A20)
    H = np.exp(1j * F)
    return F, H

def run_sim(K=1.5, T=20000, burn=15000, seed=1, F_target=2*np.pi/3):
    rng = np.random.default_rng(seed)
    N = 3
    degrees = np.array([2,2,2], dtype=float)

    # --- Phase-lag (frustration) matrix alpha (antisymmetric) ---
    # Choose alpha_01=0, alpha_12=0, alpha_20=+F_target
    alpha = np.zeros((N,N), dtype=float)
    alpha[0,1] = 0.0;           alpha[1,0] = -alpha[0,1]
    alpha[1,2] = 0.0;           alpha[2,1] = -alpha[1,2]
    alpha[2,0] = +F_target;     alpha[0,2] = -alpha[2,0]

    # initial phases
    theta = rng.uniform(-np.pi, np.pi, size=N)

    # time series
    thetas = np.zeros((T, N), dtype=float)
    for t in range(T):
        thetas[t] = theta
        theta = kuramoto_map_step_lag(theta, K, alpha, degrees, delta=None)

    post = thetas[burn:]
    r = order_parameter_t(post)
    r_mean = float(r.mean())

    # Flux with alpha-corrected connection
    F, H_u1 = holonomy_flux_U1_with_alpha(post[-1], alpha)
    H_su2, tr_half = su2_from_flux(F)

    # For diagnostics: the theoretical loop-sum of alphas (should equal -F modulo 2π)
    sum_alpha = mod2pi(alpha[0,1] + alpha[1,2] + alpha[2,0])

    return {
        "alpha": alpha,
        "sum_alpha": float(sum_alpha),
        "thetas": thetas,
        "post": post,
        "r": r,
        "r_mean": r_mean,
        "F": float(F),
        "H_u1": H_u1,
        "H_su2": H_su2,
        "tr_half": float(tr_half),
        "K": float(K),
        "F_target": float(F_target)
    }

def main():
    out = run_sim(K=1.5, T=20000, burn=15000, seed=2, F_target=2*np.pi/3)

    print(f"K = {out['K']:.3f}")
    print(f"Mean order parameter r (post-burn) = {out['r_mean']:.3f}")
    print(f"Chosen sum of alphas (0->1->2->0): {out['sum_alpha']:.6f} rad")
    print(f"Expected flux (=-sum_alpha mod 2π): {(mod2pi(-out['sum_alpha'])):.6f} rad")
    print(f"Measured U(1) flux F_triangle: {out['F']:.6f} rad")
    print(f"exp(i F) = {out['H_u1']}")
    print(f"0.5 * Tr(H_su2) = {out['tr_half']:.6f}  (should equal cos(F/2)={np.cos(out['F']/2):.6f})")

    # Plot r(t) post-burn
    plt.figure(figsize=(6.4,3.2))
    plt.plot(out['r'], lw=1.2)
    plt.xlabel("time (steps after burn-in)")
    plt.ylabel("order parameter r(t)")
    plt.title("Kuramoto order parameter on K3 with phase-lag frustration")
    plt.tight_layout()
    plt.show()


    # Inspect final phases and edge differences (raw vs corrected)
    th = out['post'][-1]
    alpha = np.zeros((3,3)); alpha[2,0]=+2*np.pi/3; alpha[0,2]=-2*np.pi/3  # same as in the run

    def dphi(i,j):  # raw phase jump
        return mod2pi(th[j]-th[i])

    raw = np.array([dphi(0,1), dphi(1,2), dphi(2,0)])
    corr = np.array([mod2pi(raw[0]-alpha[0,1]),
                     mod2pi(raw[1]-alpha[1,2]),
                     mod2pi(raw[2]-alpha[2,0])])

    print("\nFinal phases (rad):", th)
    print("Raw edge diffs (0->1,1->2,2->0):", raw)
    print("Corrected diffs (minus alpha):  ", corr)
    print("Sum corrected diffs (mod 2π):   ", mod2pi(corr.sum()))


if __name__ == "__main__":
    main()
