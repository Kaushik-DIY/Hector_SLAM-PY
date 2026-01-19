# slam/pose_graph.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def v2t(x: np.ndarray) -> np.ndarray:
    """SE2 vector -> 3x3 homogeneous transform."""
    c = np.cos(x[2]); s = np.sin(x[2])
    T = np.array([[c, -s, x[0]],
                  [s,  c, x[1]],
                  [0,  0, 1.0]], dtype=float)
    return T

def t2v(T: np.ndarray) -> np.ndarray:
    """3x3 homogeneous transform -> SE2 vector."""
    th = np.arctan2(T[1,0], T[0,0])
    return np.array([T[0,2], T[1,2], th], dtype=float)

def se2_inv(x: np.ndarray) -> np.ndarray:
    c = np.cos(x[2]); s = np.sin(x[2])
    tx, ty = x[0], x[1]
    xinv = np.zeros(3, dtype=float)
    xinv[2] = wrap_angle(-x[2])
    xinv[0] = -( c*tx + s*ty)
    xinv[1] = -(-s*tx + c*ty)
    return xinv

def se2_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a ⊕ b"""
    ca = np.cos(a[2]); sa = np.sin(a[2])
    out = np.zeros(3, dtype=float)
    out[0] = a[0] + ca*b[0] - sa*b[1]
    out[1] = a[1] + sa*b[0] + ca*b[1]
    out[2] = wrap_angle(a[2] + b[2])
    return out

def se2_between(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
    """inv(xi) ⊕ xj"""
    return se2_compose(se2_inv(xi), xj)

@dataclass
class Edge:
    i: int
    j: int
    z: np.ndarray          # measurement in i frame: [dx, dy, dtheta]
    Omega: np.ndarray      # information matrix 3x3

class PoseGraph2D:
    def __init__(self):
        self.nodes: list[np.ndarray] = []
        self.edges: list[Edge] = []

    def add_node(self, x: np.ndarray) -> int:
        self.nodes.append(np.array(x, dtype=float))
        return len(self.nodes) - 1

    def add_edge(self, i: int, j: int, z: np.ndarray, Omega: np.ndarray):
        self.edges.append(Edge(i=i, j=j, z=np.array(z, dtype=float), Omega=np.array(Omega, dtype=float)))

    def linearize_edge(self, xi: np.ndarray, xj: np.ndarray, z: np.ndarray):
        """
        pred = inv(xi) ⊕ xj
        error e = pred - z  (theta wrapped)
        Jacobians A=de/dxi, B=de/dxj
        """
        th = xi[2]
        c = np.cos(th); s = np.sin(th)
        dxw = xj[0] - xi[0]
        dyw = xj[1] - xi[1]

        pred_dx =  c*dxw + s*dyw
        pred_dy = -s*dxw + c*dyw
        pred_dth = wrap_angle(xj[2] - xi[2])

        e = np.array([pred_dx - z[0], pred_dy - z[1], wrap_angle(pred_dth - z[2])], dtype=float)

        # A = de/dxi
        A = np.zeros((3,3), dtype=float)
        A[0,0] = -c
        A[0,1] = -s
        A[0,2] = -s*dxw + c*dyw

        A[1,0] =  s
        A[1,1] = -c
        A[1,2] = -c*dxw - s*dyw

        A[2,2] = -1.0

        # B = de/dxj
        B = np.zeros((3,3), dtype=float)
        B[0,0] =  c
        B[0,1] =  s
        B[1,0] = -s
        B[1,1] =  c
        B[2,2] =  1.0

        return e, A, B

    def optimize(self, iters: int = 10, damping: float = 1e-6, fix_first: bool = True):
        """
        Gauss-Newton pose graph optimization.
        Anchors node 0 if fix_first=True.
        """
        n = len(self.nodes)
        if n < 2 or len(self.edges) == 0:
            return

        for _ in range(iters):
            H = sp.lil_matrix((3*n, 3*n), dtype=float)
            b = np.zeros((3*n,), dtype=float)

            for ed in self.edges:
                xi = self.nodes[ed.i]
                xj = self.nodes[ed.j]
                e, A, B = self.linearize_edge(xi, xj, ed.z)
                Om = ed.Omega

                # --- Robust downweighting (Huber-like) ---
                delta = 3.0  # try 2.0, 3.0, 5.0
                r = float(np.sqrt(e.T @ Om @ e))  # whitened residual magnitude
                w = 1.0 if r <= delta else (delta / r)
                Om_eff = w * Om

                ii = slice(3*ed.i, 3*ed.i+3)
                jj = slice(3*ed.j, 3*ed.j+3)

                H[ii, ii] += A.T @ Om_eff @ A
                H[ii, jj] += A.T @ Om_eff @ B
                H[jj, ii] += B.T @ Om_eff @ A
                H[jj, jj] += B.T @ Om_eff @ B

                b[ii] += A.T @ Om_eff @ e
                b[jj] += B.T @ Om_eff @ e
            

            if fix_first:
                # Fix node 0 pose to prevent gauge freedom
                # Force dx0 = 0 by setting strong diagonal constraints
                w = 1e6
                for k in range(3):
                    H[k, k] += w
                    b[k] += 0.0

            H = H.tocsr() # Convert to fast solve format
            H = H + damping * sp.eye(3*n, format="csr")

            # if fix_first:

            try:
                dx = -spla.spsolve(H, b)
            except Exception as e:
                print("Sparse solve failed:", e)
                break

            # apply increments
            max_step = 0.0
            for k in range(n):
                d = dx[3*k:3*k+3]
                self.nodes[k][0] += d[0]
                self.nodes[k][1] += d[1]
                self.nodes[k][2] = wrap_angle(self.nodes[k][2] + d[2])
                max_step = max(max_step, float(np.linalg.norm(d)))

            if max_step < 1e-6:
                break


