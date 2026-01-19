from __future__ import annotations
import numpy as np
import config as cfg
from .pyramid import MapPyramid
from .scan_matcher import align_pose_gauss_newton
from .se2 import transform_points, wrap_angle


class HectorSLAM2D:
    def __init__(self):
        self.pyr = MapPyramid.create(
            base_res=cfg.MAP_RESOLUTION,
            size_m=cfg.MAP_SIZE_METERS,
            num_levels=cfg.PYRAMID_LEVELS,
            l0=cfg.L0,
            l_min=cfg.L_MIN,
            l_max=cfg.L_MAX
        )
        self.pose = np.array([0.0, 0.0, 0.0], dtype=float)
        self.initialized = False
        self.trajectory = []

    def step(
        self,
        pts_lidar: np.ndarray,
        pose_prior: np.ndarray | None = None,
        do_mapping: bool = True
    ) -> np.ndarray:

        # 0) Choose initial guess (prior)
        if pose_prior is None:
            pose_est = self.pose.copy()
        else:
            pose_est = np.array(pose_prior, dtype=float)
        pose_est[2] = wrap_angle(pose_est[2])

        # 1) If not initialized: integrate first scan only
        if not self.initialized:
            self.pose = pose_est
            self.pose[2] = wrap_angle(self.pose[2])
            self.trajectory.append(self.pose.copy())

            if do_mapping:
                pts_world = transform_points(self.pose, pts_lidar)
                for grid in self.pyr.levels:
                    grid.integrate_scan_simple(
                        pose=self.pose,
                        pts_world=pts_world,
                        l_free=cfg.L_FREE,
                        l_occ=cfg.L_OCC,
                        ray_steps=cfg.RAY_STEPS
                    )

            self.initialized = True
            return self.pose

        # 2) Localization: COARSE -> FINE scan matching
        # If your pyramid stores finest first, reverse it.
        levels = list(self.pyr.levels)
        # Heuristic: larger res means coarser map
        if hasattr(levels[0], "res") and hasattr(levels[-1], "res"):
            if levels[0].res < levels[-1].res:
                # finest -> coarsest stored; reverse to coarse->fine
                levels = levels[::-1]

        for lvl, grid in enumerate(levels):
            iters = cfg.GN_ITERS_PER_LEVEL[min(lvl, len(cfg.GN_ITERS_PER_LEVEL) - 1)]
            pose_est = align_pose_gauss_newton(
                grid=grid,
                init_pose=pose_est,
                pts_lidar=pts_lidar,
                iters=iters,
                damping=cfg.GN_DAMPING
            )
            pose_est[2] = wrap_angle(pose_est[2])

        # 3) Commit pose + store trajectory
        self.pose = pose_est
        self.pose[2] = wrap_angle(self.pose[2])
        self.trajectory.append(self.pose.copy())

        # 4) Mapping
        if do_mapping:
            pts_world = transform_points(self.pose, pts_lidar)
            for grid in self.pyr.levels:
                grid.integrate_scan_simple(
                    pose=self.pose,
                    pts_world=pts_world,
                    l_free=cfg.L_FREE,
                    l_occ=cfg.L_OCC,
                    ray_steps=cfg.RAY_STEPS
                )

        return self.pose
