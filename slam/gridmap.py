from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class GridMap:
    """
    2D occupancy grid in log-odds form.
    Coordinate convention:
      - World frame is meters (x, y).
      - Grid coords are (gx, gy) where gx increases with world x,
        gy increases with world y.
      - Storage is array [gy, gx].
    """
    res: float
    size_m: float
    l0: float = 0.0
    l_min: float = -5.0
    l_max: float = 5.0

    def __post_init__(self):
        self.size = int(np.ceil(self.size_m / self.res))
        if self.size % 2 == 1:
            self.size += 1

        # Origin (world [0, 0]) placed at centre of the grid
        self.origin = np.array([self.size / 2.0, self.size / 2.0], dtype = float)

        # Log-odds grid
        self.logodds = np.full((self.size, self.size), self.l0, dtype = np.float32)


    # ---------------- Coordinate transforms ----------------
    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        """
        xy: (N,2) world meters -> (N,2) grid coords (float)
        """
        xy = np.asarray(xy, dtype=float)
        gx = xy[:, 0] / self.res + self.origin[0]
        gy = xy[:, 1] / self.res + self.origin[1]
        return np.stack([gx, gy], axis=1)
    
    def grid_to_world(self, gxy: np.ndarray) -> np.ndarray:
       """
        gxy: (N,2) grid coords (float) -> (N,2) world meters
       """
       gxy = np.asarray(gxy, dtype=float)
       x = (gxy[:, 0] - self.origin[0]) * self.res
       y = (gxy[:, 1] - self.origin[1]) * self.res
       return np.stack([x, y], axis=1)

    def in_bounds(self,gxy: np.ndarray) -> np.ndarray:
          """
        For bilinear interpolation we need a 1-cell margin.
          """
          x = gxy[:, 0]
          y = gxy[:, 1]
          return (x >= 1.0) & (y >= 1.0) & (x < self.size - 2.0) & (y < self.size - 2.0)
    

    # ------------- Ocuupancy representation ------------------
    def prob(self) -> np.ndarray:
        """
        Convert log-odds to occupancy probability.
        p = 1/(1+exp(-l))
        """
        l = self.logodds.astype(np.float32)
        return 1.0 / (1.0 + np.exp(-l))

    # ------------- interpolation (needed for Hector GN) --------
    @staticmethod
    def _bilinear_sample(grid: np.ndarray, gxy: np.ndarray) -> np.ndarray:
        """
        grid: (H,W) float
        gxy: (N,2) float grid coordinates (gx,gy)
        return: (N,) sampled values
        """
        x = gxy[:, 0]
        y = gxy[:, 1]
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        dx = x - x0
        dy = y - y0

        v00 = grid[y0,    x0]
        v10 = grid[y0,    x0 + 1]
        v01 = grid[y0 + 1, x0]
        v11 = grid [y0 + 1, x0 + 1]

        v0 = v00 * (1.0 - dx) + v10 * dx
        v1 = v01 * (1.0 - dx) + v11 * dx
        return v0 * (1.0 - dy) + v1 * dy

    def sample_prob(self, gxy: np.ndarray, prob_grid: np.ndarray | None = None):
        """
        Bilinear sample occupancy probability at float grid coords.
        """
        if prob_grid is None: 
            prob_grid = self.prob()
        return self._bilinear_sample(prob_grid, gxy)

    def gradients_prob(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gradients of probability map in GRID coordinates (cells).
        We'll convert to world derivatives later by dividing by res.

        Uses central differences:
          d/dx ≈ (p(x+1)-p(x-1))/2
          d/dy ≈ (p(y+1)-p(y-1))/2
        """
        p = self.prob().astype(np.float32)

        # d/dx on columns
        gx = 0.5 * (p[:, 2:] - p[:, :-2])
        gx = np.pad(gx, ((0, 0), (1, 1)), mode="edge")

        # d/dy on rows
        gy = 0.5 * (p[2:, :] - p[:-2, :])
        gy = np.pad(gy, ((1, 1), (0, 0)), mode="edge")
        return gx, gy 
    
    # ---------------- mapping update --------------------

    def add_logodds_at(self, gxy_init: np.ndarray, delta: float):
        """
        Add log-odds delta to integer grid cells, clipped.
        gxy_init: (N,2) int indices (gx,gy)
        """
        x = gxy_init[:, 0].astype(np.int32)
        y = gxy_init[:, 1].astype(np.int32)

        # additional check
        m = (x >= 0) & (y >= 0) & (x < self.size) & (y < self.size)
        x, y = x[m], y[m]
        self.logodds[y, x] = np.clip(self.logodds[y, x] + delta, self.l_min, self.l_max)
      
    
    def integrate_scan_simple(
        self,
        pose: np.ndarray,
        pts_world: np.ndarray,
        l_free: float,
        l_occ: float,
        ray_steps: int = 40
    ):
        """
        Simple mapping update (baseline):
          - For each endpoint, sample points along the ray and mark as free
          - Mark the endpoint cell as occupied
        """
        # robot origin in grid coordinates
        origin_w = np.array([[pose[0], pose[1]]], dtype=float)
        g0 = self.world_to_grid(origin_w)[0]
        x0, y0 = float(g0[0]), float(g0[1])

        # endpoints in grid coordinates
        g_end = self.world_to_grid(pts_world)
        mask = self.in_bounds(g_end)
        g_end = g_end[mask]
        if g_end.shape[0] == 0:
            return
        
        # free cells along ray (coarse sampling)
        for i in range(g_end.shape[0]):
            x1, y1 = float(g_end[i, 0]), float(g_end[i, 1])

            xs = np.linspace(x0, x1, ray_steps, dtype=float)
            ys = np.linspace(y0, y1, ray_steps, dtype=float)

            xi = np.round(xs[:-1]).astype(int)
            yi = np.round(ys[:-1]).astype(int)
            pts = np.stack([xi, yi], axis=1)

            # mark free along ray
            self.add_logodds_at(pts, l_free)

          # ocuupied endpoints
        xe = np.round(g_end[:, 0]).astype(int)
        ye = np.round(g_end[:, 1]).astype(int)
        end_pts = np.stack([xe, ye], axis=1)
        self.add_logodds_at(end_pts, l_occ) 
           


         


