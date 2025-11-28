# rtg_kuramoto_experiments.py
# ----------------------------------------------------------------------
# RTG simulations for frustrated Kuramoto motifs
# Implements:
#   - Sakaguchi–Kuramoto map with phase lags (alpha)
#   - Optional signed couplings => effective lag of pi on negative edges
#   - Triad (K3) experiment with time-series of r(t), F_triangle(t), energy E(t)
#   - Gauge-invariance check for flux
#   - K-sweep (with optional detuning sweep) + CSV export
#   - Naive K_c estimator based on coherence threshold
#   - Two-plaquette (two triangles sharing an edge) example
#
# Only depends on: numpy, matplotlib, csv (stdlib)
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass

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
    """Kuramoto order parameter r(t) over time (theta_t shape: (T, N))."""
    z = np.exp(1j * theta_t)           # (T, N)
    r = np.abs(z.mean(axis=1))         # (T,)
    return r

def su2_from_flux(F):
    """SU(2) holonomy from U(1) flux via the trace–angle relation."""
    H = np.array([[np.exp(1j*F/2), 0.0j],
                  [0.0j, np.exp(-1j*F/2)]], dtype=complex)
    tr_half = 0.5 * (H[0,0] + H[1,1]).real
    return H, tr_half

def energy_series(theta_t, edges_undirected, alpha_eff, weights=None):
    """
    E(t) = - sum_{(i,j) undirected} w_ij * cos(theta_j - theta_i - alpha_ij)
    For undirected edges we take alpha_ij as alpha[i,j] (i->j), which must be antisymmetric.
    """
    if weights is None:
        def w(i,j): return 1.0
    else:
        def w(i,j): return float(weights[i, j])
    T = theta_t.shape[0]
    E = np.zeros(T, dtype=float)
    for t in range(T):
        th = theta_t[t]
        s = 0.0
        for (i,j) in edges_undirected:
            s += w(i,j) * np.cos(th[j] - th[i] - alpha_eff[i, j])
        E[t] = -s
    return E

# -----------------------------
# Core map
# -----------------------------
def kuramoto_map_step_general(theta, K, neighbors, alpha_eff, weights, delta=None):
    """
    One synchronous update:
      theta_i(t+1) = theta_i(t) + Delta_i + (K/deg_i) sum_{j in N(i)} w_ij * sin(theta_j - theta_i - alpha_eff_ij)

    neighbors: list of lists, neighbors[i] -> iterable of js
    alpha_eff: NxN antisymmetric matrix of effective lags used in dynamics
    weights:   NxN nonnegative weights (magnitudes of couplings)
    """
    th = np.asarray(theta, dtype=float)
    N = th.shape[0]
    if delta is None:
        delta = np.zeros(N, dtype=float)
    new = th.copy()
    degrees = np.array([len(neighbors[i]) for i in range(N)], dtype=float)
    for i in range(N):
        s = 0.0
        for j in neighbors[i]:
            s += weights[i, j] * np.sin(th[j] - th[i] - alpha_eff[i, j])
        new[i] = th[i] + delta[i] + (K / degrees[i]) * s
    return mod2pi(new)

# -----------------------------
# Motif builders
# -----------------------------
@dataclass
class Triad:
    N: int
    neighbors: list         # adjacency as neighbor lists
    edges_undirected: list  # list of undirected edges (i<j)
    alpha: np.ndarray       # NxN antisymmetric (given by user)
    kappa: np.ndarray       # NxN signed couplings (magnitudes + signs)
    use_kappa_sign_as_pi: bool  # if True, treat negative sign as an added pi phase in alpha

def build_triad(F_target=2*np.pi/3, use_kappa_sign_as_pi=False, signed_pattern=None):
    """
    Build a triad (K3).
    By default, impose frustration via a single phase-lag edge:
      alpha[2,0] = +F_target; alpha[0,2] = -F_target; others 0.
    If use_kappa_sign_as_pi=True and signed_pattern provided (e.g., (+,+,-)),
    we also add an effective lag of pi on negative edges.
    """
    N = 3
    neighbors = {0:[1,2], 1:[0,2], 2:[0,1]}
    edges_undirected = [(0,1),(1,2),(0,2)]

    alpha = np.zeros((N,N), dtype=float)
    alpha[2,0] = +F_target
    alpha[0,2] = -F_target

    kappa = np.ones((N,N), dtype=float)
    for i in range(N):
        kappa[i,i] = 0.0

    if signed_pattern is not None:
        # signed_pattern like {(0,1):+1,(1,2):+1,(2,0):-1}; symmetrize
        for (i,j), s in signed_pattern.items():
            kappa[i,j] = s
            kappa[j,i] = s

    return Triad(
        N=N,
        neighbors=[neighbors[i] for i in range(N)],
        edges_undirected=edges_undirected,
        alpha=alpha,
        kappa=kappa,
        use_kappa_sign_as_pi=bool(use_kappa_sign_as_pi)
    )

@dataclass
class TwoPlaquettes:
    N: int
    neighbors: list
    edges_undirected: list
    cycles: list           # list of cycles, each as an ordered list of nodes
    alpha: np.ndarray
    kappa: np.ndarray
    use_kappa_sign_as_pi: bool

def build_two_plaquettes(F1=2*np.pi/3, F2=-2*np.pi/3, use_kappa_sign_as_pi=False):
    """
    4-node graph with two triangles sharing an edge:
      Triangle A: 0-1-2-0 (flux F1 via 2->0)
      Triangle B: 0-2-3-0 (flux F2 via 3->0)
    """
    N = 4
    neighbors = {
        0:[1,2,3],
        1:[0,2],
        2:[0,1,3],
        3:[0,2],
    }
    edges_undirected = [(0,1),(1,2),(0,2),(0,3),(2,3)]
    alpha = np.zeros((N,N), dtype=float)
    alpha[2,0] += F1; alpha[0,2] -= F1   # triangle A
    alpha[3,0] += F2; alpha[0,3] -= F2   # triangle B

    kappa = np.ones((N,N), dtype=float)
    for i in range(N):
        kappa[i,i]=0.0

    cycles = [
        [0,1,2,0],  # A
        [0,2,3,0],  # B
    ]
    return TwoPlaquettes(
        N=N,
        neighbors=[neighbors[i] for i in range(N)],
        edges_undirected=edges_undirected,
        cycles=cycles,
        alpha=alpha,
        kappa=kappa,
        use_kappa_sign_as_pi=bool(use_kappa_sign_as_pi)
    )

# -----------------------------
# Alpha/kappa handling
# -----------------------------
def effective_alpha(alpha, kappa, use_kappa_sign_as_pi=True):
    """
    Combine the user lag alpha with an extra pi on negative couplings if requested.
    Return an antisymmetric matrix alpha_eff to be used in the dynamics and diagnostics.
    """
    N = alpha.shape[0]
    alpha_eff = alpha.copy()
    if use_kappa_sign_as_pi:
        neg = (kappa < 0).astype(float) * np.pi
        alpha_eff = mod2pi(alpha_eff + neg)
        for i in range(N):
            for j in range(N):
                if i == j: 
                    continue
                alpha_eff[j,i] = -alpha_eff[i,j]
    return alpha_eff

def build_weights(kappa):
    """Use magnitude of kappa as interaction weights."""
    return np.abs(kappa)

# -----------------------------
# Flux on cycles
# -----------------------------
def flux_on_cycle(theta, cycle_nodes, alpha_eff):
    """
    F(C) = sum over oriented edges (i->j) along cycle of (theta_j - theta_i - alpha_eff_ij) mod 2π
    cycle_nodes: e.g. [0,1,2,0]
    """
    th = np.asarray(theta, dtype=float)
    F = 0.0
    for a in range(len(cycle_nodes)-1):
        i = cycle_nodes[a]
        j = cycle_nodes[a+1]
        F += mod2pi(th[j] - th[i] - alpha_eff[i, j])
    return mod2pi(F)

# -----------------------------
# Run helpers
# -----------------------------
def run_general(N, neighbors, edges_undirected, alpha, kappa, K=1.5, T=20000, burn=15000, seed=1,
                delta_vec=None, use_kappa_sign_as_pi=True, cycles=None):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=N)

    alpha_eff = effective_alpha(alpha, kappa, use_kappa_sign_as_pi=use_kappa_sign_as_pi)
    weights   = build_weights(kappa)
    if delta_vec is None:
        delta_vec = np.zeros(N, dtype=float)

    thetas = np.zeros((T, N), dtype=float)
    for t in range(T):
        thetas[t] = theta
        theta = kuramoto_map_step_general(theta, K, neighbors, alpha_eff, weights, delta=delta_vec)

    post = thetas[burn:]
    r = order_parameter_t(post)
    r_mean = float(r.mean())

    E = energy_series(post, edges_undirected, alpha_eff, weights=weights)
    flux_series = None
    if cycles:
        flux_series = []
        for t in range(post.shape[0]):
            th = post[t]
            Fs = [flux_on_cycle(th, cyc, alpha_eff) for cyc in cycles]
            flux_series.append(Fs)
        flux_series = np.array(flux_series)  # shape: (Tpost, n_cycles)

    return {
        "thetas": thetas,
        "post": post,
        "r": r,
        "r_mean": r_mean,
        "E": E,
        "alpha_eff": alpha_eff,
        "weights": weights,
        "flux_series": flux_series
    }

# -----------------------------
# Sweeps & CSV
# -----------------------------
def k_sweep(build_fn, K_values, detuning=0.0, use_kappa_sign_as_pi=True, seed=2, T=12000, burn=9000,
            csv_path=None):
    r_means = []
    F_last  = []
    Ks = []
    for K in K_values:
        motif = build_fn()
        N = motif.N
        cycles = getattr(motif, 'cycles', [[0,1,2,0]]) if hasattr(motif, 'cycles') else [[0,1,2,0]]
        delta_vec = np.zeros(N, dtype=float)
        if detuning != 0.0:
            if N >= 3:
                delta_vec = np.array([ detuning, 0.0, -detuning] + [0.0]*(N-3), dtype=float)
            else:
                delta_vec = detuning*np.ones(N, dtype=float)

        out = run_general(
            N=motif.N,
            neighbors=motif.neighbors,
            edges_undirected=motif.edges_undirected,
            alpha=motif.alpha,
            kappa=motif.kappa,
            K=K,
            T=T,
            burn=burn,
            seed=seed,
            delta_vec=delta_vec,
            use_kappa_sign_as_pi=use_kappa_sign_as_pi,
            cycles=cycles
        )
        r_means.append(out["r_mean"])
        if out["flux_series"] is not None:
            F_last.append([float(f) for f in out["flux_series"][-1]])
        else:
            F_last.append([np.nan])
        Ks.append(float(K))

    r_means = np.array(r_means, dtype=float)
    F_last = np.array(F_last, dtype=float)

    if csv_path is not None:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["K", "r_mean"] + [f"F_cycle{j}" for j in range(F_last.shape[1])]
            w.writerow(header)
            for i in range(len(Ks)):
                w.writerow([Ks[i], r_means[i]] + list(F_last[i]))

    return Ks, r_means, F_last

def estimate_Kc(Ks, r_means, r_threshold=0.70):
    """Naive onset: first K where r_mean >= r_threshold."""
    for K, r in zip(Ks, r_means):
        if r >= r_threshold:
            return float(K)
    return np.nan

# -----------------------------
# Main demos
# -----------------------------
def demo_triad_single(K=1.5, F_target=2*np.pi/3, detuning=0.0, T=20000, burn=15000, seed=1):
    motif = build_triad(F_target=F_target, use_kappa_sign_as_pi=False, signed_pattern=None)
    N = motif.N
    delta_vec = np.zeros(N, dtype=float)
    if detuning != 0.0:
        delta_vec = np.array([ detuning, 0.0, -detuning], dtype=float)

    out = run_general(
        N=motif.N,
        neighbors=motif.neighbors,
        edges_undirected=motif.edges_undirected,
        alpha=motif.alpha,
        kappa=motif.kappa,
        K=K,
        T=T,
        burn=burn,
        seed=seed,
        delta_vec=delta_vec,
        use_kappa_sign_as_pi=False,
        cycles=[[0,1,2,0]]
    )

    # Single-run summary
    F_last = float(out["flux_series"][-1,0])
    H_su2, tr_half = su2_from_flux(F_last)
    print(f"K = {K:.3f}")
    print(f"Mean order parameter r (post-burn) = {out['r_mean']:.3f}")
    print(f"Measured U(1) flux F_triangle (last) = {F_last:.6f} rad")
    print(f"0.5 * Tr(H_su2) = {tr_half:.6f}  (cos(F/2) = {np.cos(F_last/2):.6f})")

    # Time series plots: r(t), E(t), F(t)
    fig = plt.figure(figsize=(6.4,3.2))
    plt.plot(out["r"], lw=1.2)
    plt.xlabel("time (steps after burn-in)")
    plt.ylabel("order parameter r(t)")
    plt.title("Kuramoto order parameter on K3 with phase-lag frustration")
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(6.4,3.2))
    plt.plot(out["E"], lw=1.2)
    plt.xlabel("time (steps after burn-in)")
    plt.ylabel("energy E(t)")
    plt.title("Energy vs time")
    plt.tight_layout()
    plt.show()

    F_series = out["flux_series"][:,0]
    fig = plt.figure(figsize=(6.4,3.2))
    plt.plot(np.unwrap(F_series), lw=1.2)
    plt.xlabel("time (steps after burn-in)")
    plt.ylabel("unwrapped F_triangle(t) [rad]")
    plt.title("Flux on triangle vs time (unwrapped)")
    plt.tight_layout()
    plt.show()

    # Gauge-invariance sanity check
    th = out["post"][-1]
    alpha_eff = effective_alpha(motif.alpha, motif.kappa, use_kappa_sign_as_pi=False)
    F_base = flux_on_cycle(th, [0,1,2,0], alpha_eff)
    F_shift = flux_on_cycle(th + 1.234, [0,1,2,0], alpha_eff)
    print("Gauge-invariance check: F_shift - F_base =", mod2pi(F_shift - F_base))

def demo_triad_k_sweep(F_target=2*np.pi/3, detuning=0.0, K_min=0.2, K_max=2.5, nK=16, csv_path="k_sweep.csv"):
    def make():
        return build_triad(F_target=F_target, use_kappa_sign_as_pi=False, signed_pattern=None)

    Ks = np.linspace(K_min, K_max, nK)
    Ks, r_means, F_last = k_sweep(make, Ks, detuning=detuning, use_kappa_sign_as_pi=False, csv_path=csv_path)

    # Plots
    fig, ax = plt.subplots(1,2, figsize=(10,3.2))
    ax[0].plot(Ks, r_means, 'o-')
    ax[0].set_xlabel('K (coupling)')
    ax[0].set_ylabel('mean r (post-burn)')
    ax[0].set_title('Coherence vs K')

    ax[1].plot(Ks, ((F_last[:,0] + np.pi)%(2*np.pi)-np.pi), 'o-')
    ax[1].set_xlabel('K (coupling)')
    ax[1].set_ylabel('F_triangle (rad) (mod 2π)')
    ax[1].set_title('Flux on triangle vs K')
    plt.tight_layout()
    plt.show()

    Kc_est = estimate_Kc(Ks, r_means, r_threshold=0.70)
    print(f"Naive K_c estimate (r_threshold=0.70): {Kc_est}")

def demo_triad_detuning_sweep(F_target=2*np.pi/3, K=1.5, detunings=None, csv_path="detuning_sweep.csv"):
    if detunings is None:
        detunings = np.linspace(0.0, 0.05, 11)  # small detuning range

    vals = []
    for d in detunings:
        motif = build_triad(F_target=F_target, use_kappa_sign_as_pi=False, signed_pattern=None)
        N = motif.N
        delta_vec = np.array([d, 0.0, -d], dtype=float)

        out = run_general(
            N=motif.N,
            neighbors=motif.neighbors,
            edges_undirected=motif.edges_undirected,
            alpha=motif.alpha,
            kappa=motif.kappa,
            K=K,
            T=14000,
            burn=10000,
            seed=3,
            delta_vec=delta_vec,
            use_kappa_sign_as_pi=False,
            cycles=[[0,1,2,0]]
        )
        F_last = float(out["flux_series"][-1,0])
        vals.append((d, out["r_mean"], F_last))

    # CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detuning", "r_mean", "F_triangle"])
        for d, r_m, F in vals:
            w.writerow([d, r_m, F])

    # Plot
    det = np.array([v[0] for v in vals])
    r_m = np.array([v[1] for v in vals])
    F_l = np.array([v[2] for v in vals])
    fig = plt.figure(figsize=(6.4,3.2))
    plt.plot(det, r_m, 'o-')
    plt.xlabel("detuning magnitude (Δ)")
    plt.ylabel("mean r (post-burn)")
    plt.title("Coherence vs detuning (K fixed)")
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(6.4,3.2))
    plt.plot(det, ((F_l + np.pi)%(2*np.pi)-np.pi), 'o-')
    plt.xlabel("detuning magnitude (Δ)")
    plt.ylabel("F_triangle (rad) (mod 2π)")
    plt.title("Flux vs detuning (K fixed)")
    plt.tight_layout()
    plt.show()

def demo_two_plaquettes(K=1.5, F1=2*np.pi/3, F2=-2*np.pi/3, T=20000, burn=15000, seed=4):
    motif = build_two_plaquettes(F1=F1, F2=F2, use_kappa_sign_as_pi=False)
    out = run_general(
        N=motif.N,
        neighbors=motif.neighbors,
        edges_undirected=motif.edges_undirected,
        alpha=motif.alpha,
        kappa=motif.kappa,
        K=K,
        T=T,
        burn=burn,
        seed=seed,
        delta_vec=np.zeros(motif.N),
        use_kappa_sign_as_pi=False,
        cycles=motif.cycles
    )
    print(f"K={K:.3f}, mean r (post-burn) = {out['r_mean']:.3f}")

    # Flux time series for both cycles
    F_series = out["flux_series"]    # shape (Tpost, 2)
    fig = plt.figure(figsize=(6.4,3.2))
    plt.plot(np.unwrap(F_series[:,0]), lw=1.2, label="triangle A (0-1-2-0)")
    plt.plot(np.unwrap(F_series[:,1]), lw=1.2, label="triangle B (0-2-3-0)")
    plt.xlabel("time (steps after burn-in)")
    plt.ylabel("unwrapped flux F(t) [rad]")
    plt.title("Two plaquettes: flux vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Entry point
# -----------------------------
def main():
    # ==== Choose what to run ====
    RUN_TRIAD_SINGLE      = True
    RUN_TRIAD_K_SWEEP     = True
    RUN_TRIAD_DET_SWEEP   = True
    RUN_TWO_PLAQUETTES    = True

    # ==== Common knobs ====
    K_default   = 1.5
    F_target    = 2*np.pi/3
    detuning    = 0.0

    # ---- 1) Single triad run with full diagnostics ----
    if RUN_TRIAD_SINGLE:
        demo_triad_single(K=K_default, F_target=F_target, detuning=detuning,
                          T=20000, burn=15000, seed=2)

    # ---- 2) K sweep (CSV written to k_sweep.csv) ----
    if RUN_TRIAD_K_SWEEP:
        demo_triad_k_sweep(F_target=F_target, detuning=0.0, K_min=0.2, K_max=2.5, nK=16,
                           csv_path="k_sweep.csv")

    # ---- 3) Detuning sweep at fixed K (CSV written to detuning_sweep.csv) ----
    if RUN_TRIAD_DET_SWEEP:
        demo_triad_detuning_sweep(F_target=F_target, K=K_default,
                                  detunings=np.linspace(0.0, 0.05, 11),
                                  csv_path="detuning_sweep.csv")

    # ---- 4) Two-plaquette demonstration ----
    if RUN_TWO_PLAQUETTES:
        demo_two_plaquettes(K=K_default, F1=+2*np.pi/3, F2=-2*np.pi/3,
                            T=20000, burn=15000, seed=4)

if __name__ == "__main__":
    main()
