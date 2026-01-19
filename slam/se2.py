import numpy as np

def wrap_angle(theta: float) -> float:
    """Wrap angles to [-pi, pi)."""
    return (theta +np.pi) % (2.0 * np.pi) - np.pi


def rot2(theta: float) -> np.ndarray:
    """2x2 rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def transform_points(pose: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """
    pose: (3,) -> [x, y, theta]
    pts_xy: (N,2) points in sensor frame
    return: (N,2) points in world frame
    """
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    R = rot2(th)
    return pts_xy @ R.T + np.array([x, y], dtype = float)

def d_world_d_theta(theta: float, pts_xy: np.ndarray) -> np.ndarray:
    """
    Derivative of world point wrt theta:
      world = R(theta) p + t
      d/dtheta (R p) = (dR/dtheta) p
    where dR/dtheta = [[-sin, -cos],[cos, -sin]]
    returns (N,2)
    """
    s, c = np.sin(theta), np.cos(theta)
    dR = np.array([[-s, -c], [c, -s]], dtype=float)
    return pts_xy @ dR.T
