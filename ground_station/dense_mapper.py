"""
Dense mapping — coloured point-cloud fusion and occupancy-grid export.

Each keyframe's depth map is back-projected into the camera frame, the
"flying pixel" filter removes unreliable points at depth edges, the
remaining points are coloured from the RGB frame, transformed into the
world frame, and merged into a global Open3D point cloud.  After each
merge the cloud is voxel-down-sampled to limit memory.

Occupancy grid
--------------
``save_occupancy`` slices the cloud at a configurable height band
(default −0.5 m … +0.5 m in world-y, i.e. roughly table-height for an
indoor flight) and rasterises a top-down bitmap: free / occupied /
unknown.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import open3d as o3d

from .utils import Pose, back_project, flying_pixel_mask

logger = logging.getLogger(__name__)


class DenseMapper:
    """Incrementally fuses coloured depth keyframes into a world-frame cloud.

    Parameters
    ----------
    K : 3×3 camera intrinsics.
    voxel_size : voxel side length (metres) for down-sampling.
    subsample : take every *n*-th pixel when back-projecting (speed vs density).
    flying_px_thresh : relative-gradient threshold for the flying-pixel filter.
    """

    def __init__(
        self,
        K: np.ndarray,
        voxel_size: float = 0.02,
        subsample: int = 2,
        flying_px_thresh: float = 0.05,
    ):
        self.K = K.astype(np.float64)
        self.voxel_size = voxel_size
        self.subsample = subsample
        self.flying_px_thresh = flying_px_thresh

        self._cloud = o3d.geometry.PointCloud()
        self._n_keyframes = 0
        self._total_points = 0

    # ------------------------------------------------------------------ #
    #  Add a keyframe                                                      #
    # ------------------------------------------------------------------ #

    def add_keyframe(
        self,
        color_bgr: np.ndarray,
        depth: np.ndarray,
        pose_wc: Pose,
    ) -> None:
        """Back-project, filter, colour, transform and fuse one keyframe.

        Args:
            color_bgr : H×W×3 uint8 BGR image.
            depth     : H×W float32 depth in metres.
            pose_wc   : camera-to-world pose for this keyframe.
        """
        # 1. Flying-pixel mask (True = keep)
        keep = flying_pixel_mask(depth, self.flying_px_thresh)

        # 2. Back-project to camera-frame 3-D points
        pts_cam, pix = back_project(
            depth, self.K, mask=keep, subsample=self.subsample,
        )
        if len(pts_cam) == 0:
            logger.warning("Keyframe %d: no valid points", self._n_keyframes)
            return

        # 3. Colour from the BGR frame  →  RGB float [0,1]
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rows = np.clip(pix[:, 1], 0, color_rgb.shape[0] - 1)
        cols = np.clip(pix[:, 0], 0, color_rgb.shape[1] - 1)
        colours = color_rgb[rows, cols].astype(np.float64) / 255.0

        # 4. Transform to world frame
        pts_world = pose_wc.transform_points(pts_cam)

        # 5. Build partial cloud and merge (ensure C-contiguous float64 for Open3D)
        partial = o3d.geometry.PointCloud()
        partial.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(pts_world, dtype=np.float64)
        )
        partial.colors = o3d.utility.Vector3dVector(
            np.ascontiguousarray(colours, dtype=np.float64)
        )

        self._cloud += partial

        # 6. Voxel down-sample to keep memory bounded
        self._cloud = self._cloud.voxel_down_sample(self.voxel_size)

        self._n_keyframes += 1
        n = len(self._cloud.points)
        self._total_points = n
        logger.info(
            "Fused keyframe %d: +%d pts → %d total (voxel %.3f m)",
            self._n_keyframes, len(pts_world), n, self.voxel_size,
        )

    # ------------------------------------------------------------------ #
    #  Accessors                                                           #
    # ------------------------------------------------------------------ #

    def get_cloud(self) -> o3d.geometry.PointCloud:
        return self._cloud

    @property
    def num_keyframes(self) -> int:
        return self._n_keyframes

    @property
    def num_points(self) -> int:
        return self._total_points

    # ------------------------------------------------------------------ #
    #  Outputs                                                             #
    # ------------------------------------------------------------------ #

    def save_ply(self, path: str) -> None:
        o3d.io.write_point_cloud(path, self._cloud)
        logger.info("Saved %d points → %s", len(self._cloud.points), path)

    def save_occupancy(
        self,
        path: str,
        height_band: tuple[float, float] = (-0.5, 0.5),
        resolution: float = 0.05,
    ) -> None:
        """Render a top-down occupancy image from a height slice.

        The height band is along the **world-y** axis (y-down convention,
        so *negative* y = above the start camera).

        Pixels:  0 = free (no points),  128 = unknown,  255 = occupied.
        """
        pts = np.asarray(self._cloud.points)
        if len(pts) == 0:
            logger.warning("No points — skipping occupancy export")
            return

        # Slice by height (y axis, pointing down)
        y_lo, y_hi = height_band
        mask = (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)
        sliced = pts[mask]
        if len(sliced) == 0:
            logger.warning("No points in height band [%.1f, %.1f]", y_lo, y_hi)
            return

        # Project to x-z plane
        xs = sliced[:, 0]
        zs = sliced[:, 2]
        x_min, x_max = xs.min(), xs.max()
        z_min, z_max = zs.min(), zs.max()

        cols = int((x_max - x_min) / resolution) + 1
        rows = int((z_max - z_min) / resolution) + 1
        cols = max(cols, 1)
        rows = max(rows, 1)

        grid = np.full((rows, cols), 128, dtype=np.uint8)  # unknown

        ci = ((xs - x_min) / resolution).astype(int)
        ri = ((zs - z_min) / resolution).astype(int)
        ci = np.clip(ci, 0, cols - 1)
        ri = np.clip(ri, 0, rows - 1)

        grid[ri, ci] = 255  # occupied

        cv2.imwrite(path, grid)
        logger.info(
            "Saved occupancy %dx%d → %s  (height band [%.1f, %.1f])",
            cols, rows, path, y_lo, y_hi,
        )

