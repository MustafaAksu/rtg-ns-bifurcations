# rtg_kuramoto_extensions.py
# RTG – Discrete Sakaguchi–Kuramoto simulations with holonomy/flux diagnostics
# Includes: K-sweep, Flux-sweep, detuning sweep, K_c estimator, energy diagnostic,
#           signed-coupling demo and two-triangle graph.

import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

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
    z = np.exp(1j * theta_t)
    return np.abs(z.mean(axis=1))

def su2_from_flux(F):
    """Return SU(2) holonomy for flux F and 0.5*Tr(H). Axis chosen = z-hat."""
    H = np.array([[np.exp(1j*F/2), 0.0j],
                  [0.0j, np.exp(-1j*F/2)]], dtype=complex)
    tr_half = 0.5 * (H[0,0] + H[1,1]).real
    return H, tr_half

def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[CSV] wrote {path}")

# -----------------------------
# Graph helpers
# -----------------------------
@dataclass
class Graph:
    N: int
    undirected_edges: List[Tuple[int,int]]  # each edge once (i<j not required)
    neighbors: Dict[int, List[int]]
    degrees: np.ndarray  # shape (N,)

def make_graph(N: int, undirected_edges: List[Tuple[int,int]]) -> Graph:
    neighbors = {i: [] for i in range(N)}
    seen = set()
    for (i,j) in undirected_edges:
        if (i,j) in seen or (j,i) in seen:
            continue
        neighbors[i].append(j)
        neighbors[j].append(i)
        seen.add((i,j))
    degrees = np.array([len(neighbors[i]) for i in range(N)], dtype=float)
    return Graph(N, undirected_edges=list(seen), neighbors=neighbors, degrees=degrees)

def energy(theta, alpha, edges):
    """E = -sum_{(ij) in undirected edges} cos(theta_j - theta_i - alpha_ij)"""
    th = np.asarray(theta, dtype=float)
    E = 0.0
    for (i,j) in edges:
        E -= np.cos(th[j] - th[i] - alpha[i,j])
    return E

def corrected_diff(theta_i, theta_j, alpha_ij):
    return mod2pi((theta_j - theta_i) - alpha_ij)

def cycle_flux(theta, alpha, cycle_nodes: List[int]):
    """
    Oriented cycle given as [n0, n1, ..., nk, n0].
    Returns U(1) flux F and exp(iF).
    """
    th = np.asarray(theta, dtype=float)
    F = 0.0
    for a, b in zip(cycle_nodes[:-1], cycle_nodes[1:]):
        F += corrected_diff(th[a], th[b], alpha[a,b])
    F = mod2pi(F)
    return F, np.exp(1j*F)

# -----------------------------
# Dynamics (Sakaguchi–Kuramoto)
# -----------------------------
def step_sakaguchi(theta, K, alpha, graph: Graph, delta=None):
    th = np.asarray(theta, dtype=float)
    N = graph.N
    if delta is None:
        delta = np.zeros(N, dtype=float)
    new = th.copy()
    for i in range(N):
        s = 0.0
        for j in graph.neighbors[i]:
            s += np.sin(th[j] - th[i] - alpha[i,j])
        new[i] = th[i] + delta[i] + (K / graph.degrees[i]) * s
    return mod2pi(new)

@dataclass
class RunOut:
    thetas: np.ndarray  # (T, N)
    post: np.ndarray    # (T_burned, N)
    r: np.ndarray
    r_mean: float
    energy_t: np.ndarray
    F_meas: Dict[str, float]
    Hhalf_meas: Dict[str, float]
    params: Dict

def run_generic(graph: Graph,
                alpha: np.ndarray,
                cycles: Dict[str, List[int]],
                K=1.5, T=20000, burn=15000,
                delta=None, seed=1) -> RunOut:

    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=graph.N)

    thetas = np.zeros((T, graph.N), dtype=float)
    energy_t = np.zeros(T, dtype=float)

    for t in range(T):
        thetas[t] = theta
        energy_t[t] = energy(theta, alpha, graph.undirected_edges)
        theta = step_sakaguchi(theta, K, alpha, graph, delta=delta)

    post = thetas[burn:]
    r = order_parameter_t(post)
    r_mean = float(r.mean())

    # flux per cycle (last state)
    F_meas, Hhalf_meas = {}, {}
    last = post[-1]
    for name, cyc in cycles.items():
        F, _ = cycle_flux(last, alpha, cyc)
        F_meas[name] = float(F)
        _, trh = su2_from_flux(F)
        Hhalf_meas[name] = float(trh)

    return RunOut(
        thetas=thetas, post=post, r=r, r_mean=r_mean,
        energy_t=energy_t, F_meas=F_meas, Hhalf_meas=Hhalf_meas,
        params=dict(K=K, T=T, burn=burn, seed=seed)
    )

# -----------------------------
# K3 convenience builders
# -----------------------------
def alpha_from_flux_k3(F_target):
    """K3: nodes 0-1-2; set alpha_01=0, alpha_12=0, alpha_20=F_target."""
    N = 3
    alpha = np.zeros((N,N), dtype=float)
    alpha[0,1] = 0.0;            alpha[1,0] = -alpha[0,1]
    alpha[1,2] = 0.0;            alpha[2,1] = -alpha[1,2]
    alpha[2,0] = F_target;       alpha[0,2] = -alpha[2,0]
    return alpha

def graph_k3():
    return make_graph(3, [(0,1),(1,2),(2,0)])

def cycles_k3():
    return {"tri": [0,1,2,0]}

def alpha_from_signed_kappa_k3(kappa_signs):
    """
    Given signs on K3 edges (dictionary {(i,j): +/-1}), return equivalent
    phase-lag matrix with alpha_ij = 0 for +, pi for - (antisymmetric).
    """
    alpha = np.zeros((3,3), dtype=float)
    for (i,j), s in kappa_signs.items():
        a = 0.0 if s >= 0 else np.pi
        alpha[i,j] = a
        alpha[j,i] = -a
    return alpha

# -----------------------------
# Two-triangle (4-node) builder
# -----------------------------
def graph_two_triangles():
    # Nodes: 0,1,2,3 ; Tri1: 0-1-2-0 ; Tri2: 0-2-3-0 (share edge 0-2)
    return make_graph(4, [(0,1),(1,2),(2,0),(2,3),(3,0)])

def cycles_two_triangles():
    return {"tri1": [0,1,2,0],
            "tri2": [0,2,3,0]}

def alpha_for_two_triangles(F1, F2):
    """
    Build an antisymmetric alpha with tri1 flux = F1, tri2 flux = F2.
    One consistent assignment:
      alpha_01 = 0, alpha_12=0, alpha_20=F1,
      alpha_23=0, alpha_30 = F2 + F1.
    """
    N = 4
    alpha = np.zeros((N,N), dtype=float)
    alpha[0,1]=0.0; alpha[1,0]=-alpha[0,1]
    alpha[1,2]=0.0; alpha[2,1]=-alpha[1,2]
    alpha[2,0]=F1;  alpha[0,2]=-alpha[2,0]
    alpha[2,3]=0.0; alpha[3,2]=-alpha[2,3]
    alpha[3,0]=F2+F1; alpha[0,3]=-alpha[3,0]
    return alpha

# -----------------------------
# Sweeps & estimators
# -----------------------------
def sweep_K_k3(F_target, Ks, T=12000, burn=9000, delta=None, seed=2, csv_path=None):
    G = graph_k3()
    cyc = cycles_k3()
    alpha = alpha_from_flux_k3(F_target)
    rows = []
    r_means, F_vals = [], []
    for K in Ks:
        out = run_generic(G, alpha, cyc, K=K, T=T, burn=burn, delta=delta, seed=seed)
        r_means.append(out.r_mean)
        F_vals.append(out.F_meas["tri"])
        rows.append([K, out.r_mean, out.F_meas["tri"]])
    if csv_path:
        write_csv(csv_path, ["K", "r_mean", "F_triangle"], rows)
    return np.array(r_means), np.array(F_vals)

def sweep_flux_k3(K_fixed, F_targets, T=12000, burn=9000, delta=None, seed=3, csv_path=None):
    G = graph_k3()
    cyc = cycles_k3()
    rows = []
    r_means, F_meas, Hhalf, cosFhalf = [], [], [], []
    for F in F_targets:
        alpha = alpha_from_flux_k3(F)
        out = run_generic(G, alpha, cyc, K=K_fixed, T=T, burn=burn, delta=delta, seed=seed)
        r_means.append(out.r_mean)
        Fm = out.F_meas["tri"]
        F_meas.append(Fm)
        _, trh = su2_from_flux(Fm)
        Hhalf.append(trh)
        cosFhalf.append(np.cos(Fm/2.0))
        rows.append([F, out.r_mean, Fm, trh, np.cos(Fm/2.0)])
    if csv_path:
        write_csv(csv_path, ["F_target", "r_mean", "F_measured", "0.5Tr(H)", "cos(F/2)"], rows)
    return np.array(r_means), np.array(F_meas), np.array(Hhalf), np.array(cosFhalf)

def sweep_detuning_k3(F_target, K_fixed, amplitudes, T=30000, burn=22000, seed=4, csv_path=None):
    G = graph_k3()
    cyc = cycles_k3()
    alpha = alpha_from_flux_k3(F_target)
    rows = []
    r_mean_list, r_std_list, F_list = [], [], []
    for A in amplitudes:
        delta = np.array([-A, 0.0, +A], dtype=float)
        out = run_generic(G, alpha, cyc, K=K_fixed, T=T, burn=burn, delta=delta, seed=seed)
        r_mean = out.r_mean
        r_std  = float(np.std(out.r))
        r_mean_list.append(r_mean)
        r_std_list.append(r_std)
        F_list.append(out.F_meas["tri"])
        rows.append([A, r_mean, r_std, out.F_meas["tri"]])
    if csv_path:
        write_csv(csv_path, ["detuning_amp", "r_mean", "r_std", "F_triangle"], rows)
    return np.array(r_mean_list), np.array(r_std_list), np.array(F_list)

def estimate_Kc(Ks, r_means):
    """Two crude estimates: derivative peak and threshold crossing."""
    Ks = np.asarray(Ks, dtype=float)
    r = np.asarray(r_means, dtype=float)
    # derivative peak (central differences)
    dr = np.gradient(r, Ks)
    idx_peak = int(np.argmax(dr))
    Kc_peak = Ks[idx_peak]
    # threshold: first K where r exceeds baseline + eps
    eps = 0.05
    baseline = r[0]
    idx_thr = np.argmax(r > baseline + eps)
    Kc_thr = Ks[idx_thr] if (r > baseline + eps).any() else np.nan
    return Kc_peak, Kc_thr, dr

# -----------------------------
# Main demos
# -----------------------------
def demo_single_k3():
    print("\n=== Single-run K3 with flux F_target=2π/3 ===")
    F_target = 2*np.pi/3
    G = graph_k3()
    alpha = alpha_from_flux_k3(F_target)
    cyc = cycles_k3()
    out = run_generic(G, alpha, cyc, K=1.5, T=20000, burn=15000, delta=None, seed=1)
    print(f"K={out.params['K']}, r_mean={out.r_mean:.3f}")
    Fm = out.F_meas['tri']
    print(f"Measured F_triangle={Fm:.6f} rad, exp(iF)={np.exp(1j*Fm)}")
    _, trh = su2_from_flux(Fm)
    print(f"0.5*Tr(H)={trh:.6f}, cos(F/2)={np.cos(Fm/2):.6f}")
    # Diagnostics
    th = out.post[-1]
    A01 = corrected_diff(th[0], th[1], alpha[0,1])
    A12 = corrected_diff(th[1], th[2], alpha[1,2])
    A20 = corrected_diff(th[2], th[0], alpha[2,0])
    print(f"Corrected diffs (0->1,1->2,2->0): {A01:.6f}, {A12:.6f}, {A20:.6f}")
    print(f"Sum corrected (mod 2π): {mod2pi(A01+A12+A20):.6f}")

    # Plots
    fig, ax = plt.subplots(1,2, figsize=(10,3.2))
    ax[0].plot(out.r, lw=1.2); ax[0].set_xlabel("steps after burn-in"); ax[0].set_ylabel("r(t)")
    ax[0].set_title("Order parameter r(t)")
    ax[1].plot(out.energy_t, lw=1.0); ax[1].set_xlabel("time step"); ax[1].set_ylabel("E(t)")
    ax[1].set_title("Energy E(t) = -∑ cos(Δθ-α)")
    plt.tight_layout(); plt.show()

def demo_Ksweep_csv():
    print("\n=== K-sweep on K3 (flux fixed) ===")
    Ks = np.linspace(0.2, 2.5, 16)
    F_target = 2*np.pi/3
    r_means, F_vals = sweep_K_k3(F_target, Ks, T=12000, burn=9000, csv_path="k3_Ksweep.csv")
    Kc_peak, Kc_thr, dr = estimate_Kc(Ks, r_means)
    print(f"Estimated Kc (derivative peak) ~ {Kc_peak:.3f}")
    print(f"Estimated Kc (threshold)      ~ {Kc_thr:.3f}")

    fig, ax = plt.subplots(1,2, figsize=(10,3.2))
    ax[0].plot(Ks, r_means, 'o-'); ax[0].set_xlabel('K'); ax[0].set_ylabel('mean r'); ax[0].set_title('Coherence vs K')
    ax[0].axvline(Kc_peak, ls='--'); ax[0].axvline(Kc_thr, ls=':')
    ax[1].plot(Ks, (F_vals + np.pi)%(2*np.pi)-np.pi, 'o-'); ax[1].set_xlabel('K'); ax[1].set_ylabel('F_triangle (rad)')
    ax[1].set_title('Flux vs K (should be flat)')
    plt.tight_layout(); plt.show()

def demo_Fsweep_csv():
    print("\n=== Flux-sweep on K3 (K fixed) ===")
    K_fixed = 1.5
    F_targets = np.linspace(-np.pi, np.pi, 13)  # from -π to π
    r_means, F_meas, Hhalf, cosFhalf = sweep_flux_k3(K_fixed, F_targets, csv_path="k3_Fsweep.csv")
    fig, ax = plt.subplots(1,2, figsize=(10,3.2))
    ax[0].plot(F_targets, r_means, 'o-'); ax[0].set_xlabel('F_target (rad)'); ax[0].set_ylabel('mean r')
    ax[0].set_title('Coherence vs target flux')
    ax[1].plot(F_meas, Hhalf, 'o', label='0.5 Tr(H) measured')
    ax[1].plot(F_meas, np.cos(F_meas/2.0), '-', label='cos(F/2)')
    ax[1].set_xlabel('F_measured (rad)'); ax[1].set_ylabel('value')
    ax[1].set_title('SU(2) trace–angle check'); ax[1].legend()
    plt.tight_layout(); plt.show()

def demo_detuning_csv():
    print("\n=== Detuning sweep on K3 (NS visualization) ===")
    F_target = 2*np.pi/3
    K_fixed = 1.1
    amps = np.array([0.0, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3])
    r_mean, r_std, F_meas = sweep_detuning_k3(F_target, K_fixed, amps, csv_path="k3_detuning.csv")
    print("detuning amp   r_mean   r_std   F")
    for A, rm, rs, F in zip(amps, r_mean, r_std, F_meas):
        print(f"{A:11.5g}  {rm:7.3f}  {rs:7.4f}  {F:7.4f}")
    fig, ax = plt.subplots(1,2, figsize=(10,3.2))
    ax[0].plot(amps, r_mean, 'o-'); ax[0].set_xlabel('detuning amplitude'); ax[0].set_ylabel('mean r')
    ax[0].set_title('r_mean vs detuning')
    ax[1].plot(amps, r_std, 'o-'); ax[1].set_xlabel('detuning amplitude'); ax[1].set_ylabel('std r(t)')
    ax[1].set_title('Quasiperiodic modulation (std r)')
    plt.tight_layout(); plt.show()

def demo_signed_kappa():
    print("\n=== Signed-coupling demo (α=π on negative edges) ===")
    # One negative edge -> Z2 frustration (π flux)
    kappa_signs = {(0,1): +1, (1,2): +1, (2,0): -1}
    alpha = alpha_from_signed_kappa_k3(kappa_signs)
    G = graph_k3(); cyc = cycles_k3()
    out = run_generic(G, alpha, cyc, K=1.5, T=20000, burn=15000, seed=6)
    Fm = out.F_meas["tri"]; _, trh = su2_from_flux(Fm)
    print(f"Measured F_triangle ≈ {Fm:.6f} rad (should be ±π)")
    print(f"0.5*Tr(H) ≈ {trh:.6f} (should be ≈ 0)")
    print(f"r_mean ≈ {out.r_mean:.3f}")

def demo_two_triangles():
    print("\n=== Two-triangle graph demo (two independent fluxes) ===")
    G = graph_two_triangles(); cyc = cycles_two_triangles()
    F1 = 2*np.pi/3; F2 = np.pi/2
    alpha = alpha_for_two_triangles(F1, F2)
    out = run_generic(G, alpha, cyc, K=1.5, T=20000, burn=15000, seed=7)
    print(f"Measured tri1 flux: {out.F_meas['tri1']:.6f} rad (target {F1:+.6f})")
    print(f"Measured tri2 flux: {out.F_meas['tri2']:.6f} rad (target {F2:+.6f})")
    print(f"r_mean (4 nodes) ≈ {out.r_mean:.3f}")
    fig, ax = plt.subplots(1,2, figsize=(10,3.2))
    ax[0].plot(out.r, lw=1.2); ax[0].set_xlabel("steps after burn-in"); ax[0].set_ylabel("r(t)")
    ax[0].set_title("Global order parameter (4 nodes)")
    ax[1].plot(out.energy_t, lw=1.0); ax[1].set_xlabel("time step"); ax[1].set_ylabel("E(t)")
    ax[1].set_title("Energy E(t) (two triangles)")
    plt.tight_layout(); plt.show()

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    # 1) Single-run diagnostic (matches what you already verified)
    demo_single_k3()

    # 2) K-sweep + CSV + crude Kc estimate
    demo_Ksweep_csv()

    # 3) Flux-sweep + CSV + SU(2) check
    demo_Fsweep_csv()

    # 4) Detuning sweep (to visualize NS/tori) + CSV
    demo_detuning_csv()

    # 5) Signed-coupling -> π flux (Z2) demo
    demo_signed_kappa()

    # 6) Two-triangle graph with independent fluxes
    demo_two_triangles()
