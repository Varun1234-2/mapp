"""
Geometry utilities, camera model, and Pose class.

Coordinate conventions
======================
Camera frame : x right, y down, z forward  (OpenCV / pinhole convention)
World  frame : identical axes; origin = first camera position

Pose T_wc (4×4 homogeneous) transforms points FROM camera TO world:
    p_world = T_wc @ [p_camera; 1]

Composition: if T_wc1 is frame-1-in-world and T_12 maps frame-2 into
frame-1, then T_wc2 = T_wc1 @ T_12.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
#  Camera intrinsics helpers                                                   #
# --------------------------------------------------------------------------- #

def make_camera_matrix(hfov_deg: float, width: int, height: int) -> np.ndarray:
    """Build a 3×3 camera matrix K from horizontal FOV and image size.

    Assumes square pixels, principal point at the image centre.
    """
    fx = width / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load K and distortion coefficients from a .npz saved by calibrate.py."""
    data = np.load(path)
    return data["K"].astype(np.float64), data["dist"].astype(np.float64)


# --------------------------------------------------------------------------- #
#  Pose class                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class Pose:
    """Rigid-body transformation as a 4×4 homogeneous matrix.

    Convention: T_wc — transforms points FROM camera frame TO world frame.
    """
    T: np.ndarray  # (4, 4) float64

    # ---- constructors ----
    @staticmethod
    def identity() -> Pose:
        return Pose(np.eye(4, dtype=np.float64))

    @staticmethod
    def from_Rt(R: np.ndarray, t: np.ndarray) -> Pose:
        """Build from 3×3 rotation and 3×1 (or 3,) translation."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
        return Pose(T)

    # ---- properties ----
    @property
    def R(self) -> np.ndarray:
        return self.T[:3, :3]

    @property
    def t(self) -> np.ndarray:
        """Translation — camera origin in world coordinates."""
        return self.T[:3, 3]

    # ---- operations ----
    def compose(self, other: Pose) -> Pose:
        """self ∘ other  →  T_self @ T_other."""
        return Pose(self.T @ other.T)

    def inverse(self) -> Pose:
        """T^{-1} using the rotation/translation structure (exact)."""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Pose.from_Rt(R_inv, t_inv)

    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """Transform N×3 points by this pose.  Returns N×3."""
        return (self.R @ pts.T).T + self.t

    def to_kitti_row(self) -> str:
        """First three rows of T flattened, space-separated (KITTI format)."""
        return " ".join(f"{v:.6e}" for v in self.T[:3, :].ravel())


# --------------------------------------------------------------------------- #
#  Back-projection and depth-edge filtering                                    #
# --------------------------------------------------------------------------- #

def back_project(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray | None = None,
    subsample: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth image into 3-D points in the camera frame.

    Args:
        depth:      H×W float32 depth in metres.
        K:          3×3 intrinsic matrix.
        mask:       optional H×W bool — True keeps the pixel.
        subsample:  take every *n*-th pixel in each direction (speed).

    Returns:
        points  (N, 3) float32  — 3-D points in camera frame.
        pixels  (N, 2) int32    — corresponding (u, v) pixel coords.
    """
    h, w = depth.shape[:2]
    # Create pixel grid
    u = np.arange(0, w, subsample, dtype=np.float32)
    v = np.arange(0, h, subsample, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # each (h', w')

    d = depth[::subsample, ::subsample].astype(np.float32)

    # Apply mask
    valid = d > 0.01
    if mask is not None:
        valid &= mask[::subsample, ::subsample]

    uu = uu[valid]
    vv = vv[valid]
    d = d[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (uu - cx) * d / fx
    y = (vv - cy) * d / fy
    z = d

    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    pixels = np.stack([uu, vv], axis=-1).astype(np.int32)
    return points, pixels


def flying_pixel_mask(depth: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Return a boolean mask that is *True* for pixels to KEEP.

    Rejects "flying pixels" at depth discontinuities where the depth
    gradient magnitude exceeds *threshold* × local depth value.
    """
    import cv2
    d = depth.astype(np.float32)
    # Sobel gradient magnitude (normalised by depth)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    # Relative gradient: pixels where grad/depth > threshold are edges
    safe_depth = np.maximum(d, 1e-3)
    rel_grad = grad / safe_depth

    return (rel_grad < threshold) & (d > 0.01)

