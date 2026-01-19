import numpy as np

def scan_to_points(
    ranges: np.ndarray,
    angle_min: float,
    angle_inc: float,
    rmin: float,
    rmax: float,
    stride: int = 1 
) -> np.ndarray:
    """
    Convert a polar 2D LiDAR scan to Nx2 points in LiDAR frame.

    ranges: (N,) array (meters)
    angle_min: angle of first beam (rad)
    angle_inc: angular step (rad)
    rmin/rmax: range limits (meters)
    stride: use every k-th beam for speed

    returns: (M,2) points [x,y] in meters
    """
    ranges = np.asarray(ranges, dtype=float)

    idx = np.arange(0, ranges.size, stride)
    r = ranges[idx]
    a = angle_min + idx * angle_inc

    valid = np.isfinite(r) & (r > rmin) & (r < rmax)
    r = r[valid]
    a = a[valid]
    x = r * np.cos(a)
    y = r * np.sin(a)
    return np.stack([x, y], axis=1)
