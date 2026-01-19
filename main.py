import os
import numpy as np
from tqdm import tqdm

import config as cfg
from slam.lidar import scan_to_points
from slam.hector import HectorSLAM2D
from viz.live_view import LiveView
from dataio.carmen import read_carmen_log
from viz.plot_final import plot_map_and_traj


def run_freiburg_carmen(
    
    log_path: str,
    angle_min: float,
    angle_inc: float,
    out_dir: str = "outputs/run_no_lc",
    max_scans: int | None = None,
    live_every: int = 50,
    map_every: int = 5,
):
    """
    Runs Hector SLAM on Freiburg CARMEN log and saves baseline outputs.

    Outputs (inside out_dir):
      - trajectory.txt
      - timestamps.txt
      - odom.txt
      - final_map.npy
      - final_map_traj.png
    """
    print(">>> run_freiburg_carmen() STARTED <<<")
    os.makedirs(out_dir, exist_ok=True)

    data = read_carmen_log(log_path)
    if not data:
        print("No flaser entries found. Check the log file")
        return

    # Limit scans if requested (useful for quick tests)
    if max_scans is not None:
        data = data[:max_scans]

    slam = HectorSLAM2D()
    view = LiveView()

    # --- Initialize with first scan ---
    first = data[0]
    pts0 = scan_to_points(
        first["ranges"],
        angle_min=angle_min,
        angle_inc=angle_inc,
        rmin=cfg.LIDAR_MIN_RANGE,
        rmax=cfg.LIDAR_MAX_RANGE,
        stride=cfg.BEAM_STRIDE,
    )
    slam.step(
        pts_lidar=pts0,
        pose_prior=np.array(first["odom"], dtype=float),
        do_mapping=True
    )

    # --- Main loop ---
    for k, e in enumerate(tqdm(data[1:], desc="Freiburg SLAM")):
        pts = scan_to_points(
            e["ranges"],
            angle_min=angle_min,
            angle_inc=angle_inc,
            rmin=cfg.LIDAR_MIN_RANGE,
            rmax=cfg.LIDAR_MAX_RANGE,
            stride=cfg.BEAM_STRIDE,
        )

        prior = np.array(e["odom"], dtype=float)

        # mapping throttling for speed (still estimates pose every scan)
        do_mapping = (k % map_every == 0)

        slam.step(
            pts_lidar=pts,
            pose_prior=prior,
            do_mapping=do_mapping
        )

        if live_every is not None and live_every > 0 and (k % live_every == 0):
            view.update(slam.pyr.finest(), slam.trajectory)

    # --- Save baseline outputs ---
    traj = np.array(slam.trajectory[1:], dtype=float)  # drop init pose

    np.savetxt(f"{out_dir}/trajectory.txt", traj, fmt="%.6f", header="x y theta")

    timestamps = np.array([e["t"] for e in data[:len(traj)]], dtype=float)
    assert len(traj) == len(timestamps), "Trajectory / timestamp length mismatch"
    np.savetxt(f"{out_dir}/timestamps.txt", timestamps, fmt="%.6f", header="t")

    odom = np.array([e["odom"] for e in data[:len(traj)]], dtype=float)
    np.savetxt(f"{out_dir}/odom.txt", odom, fmt="%.6f", header="x y theta")

    # Save map
    prob = slam.pyr.finest().prob().astype(np.float32)
    np.save(f"{out_dir}/final_map.npy", prob)

    # Plot baseline map + traj
    out_img = f"{out_dir}/final_map_traj.png"
    plot_map_and_traj(slam.pyr.finest(), traj, out_path=out_img, title="Hector SLAM without Loop Closure")
    print(f"saved {out_img}")

    # Quick sanity metric (NOT ground truth)
    diff = traj - odom
    print("Mean |pose-odom|:", np.mean(np.linalg.norm(diff[:, :2], axis=1)))
    print("Max  |pose-odom|:", np.max(np.linalg.norm(diff[:, :2], axis=1)))

    print("Saved baseline outputs to:", out_dir)
    print("Trajectory length:", len(traj))
    print("Number of beams:", len(first["ranges"]))
    print("angle span (deg):", (len(first["ranges"]) - 1) * np.rad2deg(angle_inc))
    print("Trajectory span X:", traj[:, 0].min(), traj[:, 0].max())
    print("Trajectory span Y:", traj[:, 1].min(), traj[:, 1].max())

    return traj, timestamps, data


if __name__ == "__main__":
    log_path = "datasets/freiburg/fr079.clf"

    # Baseline (no loop closure)
    run_freiburg_carmen(
        log_path,
        angle_min=-np.pi / 2,
        angle_inc=np.deg2rad(0.5),
        out_dir="outputs/fr079_no_lc",
        max_scans=None,      # set e.g. 1000 for quick testing
        live_every=50,
        map_every=5,
    )
    

