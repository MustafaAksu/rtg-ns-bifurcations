# rtg_su2_lift.py
# SU(2) lifts for K4 with a common z-axis and prescribed face fluxes.
#
# This module is *pure geometry*: it knows nothing about the dynamics.
# It takes a 4-vector of face fluxes F = (F0, F1, F2, F3) and returns:
#   - φ_e for the 6 edges of K4, solving B φ = F in the least-norm sense
#   - U_e ∈ SU(2) for each edge, with axis = z
#   - face holonomies H_f built from U_e and the same incidence B
#
# The incidence convention here is:
#   vertices: 0,1,2,3
#   edges (undirected pairs, with a fixed orientation):
#     e0: (0,1)
#     e1: (0,2)
#     e2: (0,3)
#     e3: (1,2)
#     e4: (1,3)
#     e5: (2,3)
#
#   faces (oriented triangles):
#     f0: (0,1,2): 0 -> 1 -> 2 -> 0
#     f1: (0,1,3): 0 -> 1 -> 3 -> 0
#     f2: (0,2,3): 0 -> 2 -> 3 -> 0
#     f3: (1,2,3): 1 -> 2 -> 3 -> 1
#
#   The oriented boundary of each face in terms of edges is:
#     f0: +e0  -e1  +e3
#     f1: +e0  -e2  +e4
#     f2: +e1  -e2  +e5
#     f3: +e3  -e4  +e5
#
# So the 4x6 incidence matrix B has rows:
#   f0: [ 1, -1,  0,  1,  0,  0]
#   f1: [ 1,  0, -1,  0,  1,  0]
#   f2: [ 0,  1, -1,  0,  0,  1]
#   f3: [ 0,  0,  0,  1, -1,  1]
#
# With this convention, if B φ = F, then the SU(2) face holonomy is
#   H_f = exp( (F_f/2) i σ_z ),
# so Tr(H_f)/2 = cos(F_f/2) exactly in the commuting (aligned-axis) case.

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Edge / face structure and incidence
# ---------------------------------------------------------------------

# Edge list: e0..e5 as oriented pairs (i -> j).
K4_EDGES: List[Tuple[int, int]] = [
    (0, 1),  # e0
    (0, 2),  # e1
    (0, 3),  # e2
    (1, 2),  # e3
    (1, 3),  # e4
    (2, 3),  # e5
]

# Per-face edge orientations: (edge_index, sign)
# Matching the incidence description in the header.
K4_FACE_EDGES: List[List[Tuple[int, int]]] = [
    # f0: 0->1->2->0  => +e0, -e1, +e3
    [(0, +1), (1, -1), (3, +1)],
    # f1: 0->1->3->0  => +e0, -e2, +e4
    [(0, +1), (2, -1), (4, +1)],
    # f2: 0->2->3->0  => +e1, -e2, +e5
    [(1, +1), (2, -1), (5, +1)],
    # f3: 1->2->3->1  => +e3, -e4, +e5
    [(3, +1), (4, -1), (5, +1)],
]


def k4_incidence_matrix() -> np.ndarray:
    """
    Build the 4x6 incidence matrix B whose (f,e)-entry is the orientation
    sign of edge e in the boundary of face f.
    """
    B = np.zeros((4, 6), dtype=float)
    for f_idx, edges in enumerate(K4_FACE_EDGES):
        for e_idx, sgn in edges:
            B[f_idx, e_idx] = float(sgn)
    return B


# ---------------------------------------------------------------------
# SU(2) utilities (z-axis only)
# ---------------------------------------------------------------------

def su2_from_angle_z(phi: float) -> np.ndarray:
    """
    SU(2) matrix for a rotation by angle 'phi' about the z-axis.

        U(phi) = exp( (phi/2) i σ_z )
                = [[cos(phi/2) + i sin(phi/2), 0],
                   [0,          cos(phi/2) - i sin(phi/2)]]
    """
    half = 0.5 * phi
    c = float(np.cos(half))
    s = float(np.sin(half))
    return np.array([[c + 1j * s, 0.0 + 0.0j],
                     [0.0 + 0.0j, c - 1j * s]], dtype=complex)


def k4_scalar_lift(F_faces: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Minimal-norm scalar lift on K4 for prescribed face fluxes F_faces.

    Parameters
    ----------
    F_faces : array-like, shape (4,)
        Fluxes (angles) on faces f=0,1,2,3.

    Returns
    -------
    phi : np.ndarray, shape (6,)
        Edge angles φ_e such that B φ = F (in least-norm sense).
    U_edges : list of 6 np.ndarray
        SU(2) edge transports, one 2x2 matrix per edge e.
    residual : np.ndarray, shape (4,)
        The linear residual B φ - F_faces (should be ~0 if F is compatible).
    """
    F = np.asarray(F_faces, dtype=float).reshape(4)
    B = k4_incidence_matrix()

    # Least-squares with SVD => minimal-norm solution when F ∈ row(B).
    phi, *_ = np.linalg.lstsq(B, F, rcond=None)
    residual = B @ phi - F

    U_edges = [su2_from_angle_z(phi_e) for phi_e in phi]
    return phi, U_edges, residual


def k4_gauge_energy(phi: np.ndarray) -> float:
    """
    Gauge 'energy' C* = sum_e φ_e^2 for a scalar lift.
    """
    phi = np.asarray(phi, dtype=float)
    return float(np.sum(phi ** 2))


def face_holonomy(U_edges: List[np.ndarray], face_index: int) -> np.ndarray:
    """
    Compute the SU(2) holonomy H_f around face 'face_index' by multiplying
    edge transports in the oriented order.

    Parameters
    ----------
    U_edges : list of 6 SU(2) matrices
    face_index : int in {0,1,2,3}

    Returns
    -------
    H_f : 2x2 complex ndarray
    """
    H = np.eye(2, dtype=complex)
    for e_idx, sgn in K4_FACE_EDGES[face_index]:
        U = U_edges[e_idx]
        if sgn >= 0:
            H = U @ H
        else:
            # negative orientation: use inverse = conjugate transpose
            H = U.conj().T @ H
    return H
