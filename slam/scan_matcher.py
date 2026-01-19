from __future__ import annotations
import numpy as np
from .se2 import transform_points, d_world_d_theta, wrap_angle
from .gridmap import GridMap


def align_pose_gauss_newton(
    grid: GridMap,
    init_pose: np.ndarray,
    pts_lidar: np.ndarray,
    iters: int = 10,
    damping: float = 1e-3,
    min_points: int = 20
) -> np.ndarray:
    """
    Align pose by maximizing occupancy probability at scan endpoints.

    We minimize:
      E = sum_i (1 - M(p_i_world))^2

    M(.) is occupancy probability from the grid map.
    """
    pose = np.array(init_pose, dtype=float)
    pose[2] = wrap_angle(pose[2])

    # Precompute probability grid + gradients once per call
    prob_grid = grid.prob()
    grad_x, grad_y = grid.gradients_prob()

    for _ in range(iters):
        # 1) Transform scan points to world
        pts_w = transform_points(pose, pts_lidar)  # (N,2)

        # 2) Convert to grid coordinates
        gxy = grid.world_to_grid(pts_w)   # (N,2) float

        # 3) Keep only in-bound points (needed for bilinear samplings)
        mask = grid.in_bounds(gxy)
        if int(mask.sum()) < min_points:
            break

        gxy_use = gxy[mask]
        pts_use = pts_lidar[mask]  # points in lidar frame for theta values

        # 4) Sample map probabilities at endpoints
        m = grid.sample_prob(gxy_use, prob_grid)
        r = (1.0 - m).reshape(-1, 1)

        # 5) Sample probability gradients (in grid units)
        gx = GridMap._bilinear_sample(grad_x, gxy_use)
        gy = GridMap._bilinear_sample(grad_y, gxy_use)

        # Convert gradient from "per grid cell" to "per meter"
        dM_dworld = np.stack([gx, gy], axis=1) / grid.res

        # 6) Derivative of world points wrt theta (M,2)
        dPw_dth = d_world_d_theta(pose[2], pts_use)

        # 7) Jacobian of residual r = (1 - M) wrt pose
        # dr/dpose = -dM/dpose
        # dM/dx = dM/dworld_x * 1
        # dM/dy = dM/dworld_y * 1
        # dM/dtheta = dM/dworld · dPw/dtheta
        J = np.zeros((gxy_use.shape[0], 3), dtype=float)
        J[:, 0] = -dM_dworld[:, 0]
        J[:, 1] = -dM_dworld[:, 1]
        J[:, 2] = -(dM_dworld * dPw_dth).sum(axis=1)

        # 8) Gauss-Newton solve: (J^T J + λI) Δ = J^T r)
        H = (J.T @ J) + damping + np.eye(3)
        g = (J.T @ r).reshape(3)

        try:
            delta = -np.linalg.solve(H, g)
            # limit step size (helps avoid overshoot when gradients are sharp)
            delta[0] = np.clip(delta[0], -0.03, 0.03)  # meters
            delta[1] = np.clip(delta[1], -0.03, 0.03)  # meters
            delta[2] = np.clip(delta[2], -np.deg2rad(1.0), np.deg2rad(1.0))  # radians

            
        except np.linalg.LinAlgError:
            break

        # 9) Update pose
        pose[0] += float(delta[0])
        pose[1] += float(delta[1])
        pose[2] = wrap_angle(pose[2] + float(delta[2]))

        # Optional early stop if tiny update
        if float(np.linalg.norm(delta)) < 1e-6:
            break
    pose[2] = wrap_angle(pose[2])
    return pose

