# rtg_k4_basis.py
import numpy as np
from rtg_core import (
    load_config,          # loads YAML
    build_k4_spec,        # builds topology / weights
    jacobian_inertial,    # full 8x8 Jacobian
    gauge_free_basis_theta,
    block_diag,
)

cfg = load_config("configs/k4_quat_baseline.yaml")
K1 = cfg["K1_star"]
K2 = cfg["K2_star"]
K_mid = 0.5 * (K1 + K2)

spec = build_k4_spec(cfg)
J_full = jacobian_inertial(K_mid, cfg, spec)   # 8x8

# Build gauge-free projection Q: 8x6
n = 4
Q_theta = gauge_free_basis_theta(n)           # 4x3 (removes uniform phase)
Q = block_diag(Q_theta, Q_theta)              # 8x6 (angles + velocities)

J_red = Q.T @ J_full @ Q                      # 6x6

# Eigen-analysis
evals, evecs = np.linalg.eig(J_red)
idx = np.argsort(np.abs(evals))[::-1]         # sort by |λ| descending
idx_center = idx[:4]                          # top 4 near unit circle
evals_center = evals[idx_center]
V_center = evecs[:, idx_center].real          # 6x4 real basis

np.save("data/k4_V_center.npy", V_center)
np.save("data/k4_evals_center.npy", evals_center)
print("Center eigenvalues:", evals_center)
print("Saved basis to data/k4_V_center.npy")
