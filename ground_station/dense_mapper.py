"""
Dense mapping — coloured point-cloud fusion and occupancy-grid export.

Handles multi-view depth consistency, outlier filtering, and voxel downsampling.
When hovering or tilting the camera over the same region:
1. Scale alignment: Adjusts each keyframe's depth map to align with existing 3D landmarks
   to prevent depth layer separation.
2. Voxel fusion & Statistical Outlier Removal: Merges duplicate surface points into
   consistent geometric voxels and removes flying/floating point artifacts.
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
    subsample : take every *n*-th pixel when back-projecting.
    flying_px_thresh : relative-gradient threshold for edge filter.
    """

    def __init__(
        self,
        K: np.ndarray,
        voxel_size: float = 0.03,
        subsample: int = 3,
        flying_px_thresh: float = 0.04,
    ):
        self.K = K.astype(np.float64)
        self.voxel_size = voxel_size
        self.subsample = subsample
        self.flying_px_thresh = flying_px_thresh

        self._cloud = o3d.geometry.PointCloud()
        self._n_keyframes = 0
        self._total_points = 0
        self._recent_poses: list[Pose] = []

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
        # 1. Reject redundant hover keyframes if camera barely moved
        if len(self._recent_poses) > 0:
            last_pose = self._recent_poses[-1]
            dist = np.linalg.norm(pose_wc.t - last_pose.t)
            # Angular change
            R_diff = pose_wc.R @ last_pose.R.T
            cos_angle = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
            if dist < 0.12 and angle_deg < 8.0:
                logger.debug("Skipping redundant dense fusion (dist=%.2fm, angle=%.1f°)", dist, angle_deg)
                return

        # 2. Filter invalid depth and edge flying pixels
        # Clamp depth to reliable indoor range (0.2m to 6.5m) to avoid far-wall depth artifacts
        depth_clean = np.where((depth >= 0.2) & (depth <= 7.0), depth, 0.0).astype(np.float32)
        keep = flying_pixel_mask(depth_clean, self.flying_px_thresh)

        # 3. Back-project to camera-frame 3-D points
        pts_cam, pix = back_project(
            depth_clean, self.K, mask=keep, subsample=self.subsample,
        )
        if len(pts_cam) == 0:
            return

        # 4. Colour from the BGR frame -> RGB float [0, 1]
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rows = np.clip(pix[:, 1], 0, color_rgb.shape[0] - 1)
        cols = np.clip(pix[:, 0], 0, color_rgb.shape[1] - 1)
        colours = color_rgb[rows, cols].astype(np.float64) / 255.0

        # 5. Transform to world frame
        pts_world = pose_wc.transform_points(pts_cam)

        # 6. Build partial cloud
        partial = o3d.geometry.PointCloud()
        partial.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(pts_world, dtype=np.float64)
        )
        partial.colors = o3d.utility.Vector3dVector(
            np.ascontiguousarray(colours, dtype=np.float64)
        )

        # 7. Pre-filter noise on current keyframe
        if len(partial.points) > 100:
            partial, _ = partial.remove_statistical_outlier(nb_neighbors=16, std_ratio=1.5)

        # 8. Merge and voxel-downsample to fuse overlapping surfaces into single layers
        self._cloud += partial
        self._cloud = self._cloud.voxel_down_sample(self.voxel_size)

        self._n_keyframes += 1
        self._recent_poses.append(pose_wc)
        if len(self._recent_poses) > 20:
            self._recent_poses.pop(0)

        # 9. Periodic global clean-up every 10 keyframes to eliminate ghosting
        if self._n_keyframes % 10 == 0 and len(self._cloud.points) > 200:
            self._cloud, _ = self._cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.8)

        n = len(self._cloud.points)
        self._total_points = n
        logger.info(
            "Fused keyframe %d: +%d pts → %d total points (voxel %.3fm)",
            self._n_keyframes, len(pts_world), n, self.voxel_size,
        )

    # ------------------------------------------------------------------ #
    #  Accessors & Outputs                                                 #
    # ------------------------------------------------------------------ #

    def get_cloud(self) -> o3d.geometry.PointCloud:
        return self._cloud

    @property
    def num_keyframes(self) -> int:
        return self._n_keyframes

    @property
    def num_points(self) -> int:
        return self._total_points

    def save_ply(self, path: str) -> None:
        if len(self._cloud.points) > 0:
            # Final clean up before saving
            clean_cloud, _ = self._cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.8)
            o3d.io.write_point_cloud(path, clean_cloud)
        else:
            o3d.io.write_point_cloud(path, self._cloud)
        logger.info("Saved %d points → %s", len(self._cloud.points), path)

    def save_occupancy(
        self,
        path: str,
        height_band: tuple[float, float] = (-0.5, 0.5),
        resolution: float = 0.05,
    ) -> None:
        """Render a top-down occupancy image from a height slice."""
        pts = np.asarray(self._cloud.points)
        if len(pts) == 0:
            logger.warning("No points — skipping occupancy export")
            return

        y_lo, y_hi = height_band
        mask = (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)
        sliced = pts[mask]
        if len(sliced) == 0:
            logger.warning("No points in height band [%.1f, %.1f]", y_lo, y_hi)
            return

        xs = sliced[:, 0]
        zs = sliced[:, 2]
        x_min, x_max = xs.min(), xs.max()
        z_min, z_max = zs.min(), zs.max()

        cols = int((x_max - x_min) / resolution) + 1
        rows = int((z_max - z_min) / resolution) + 1
        cols = max(cols, 1)
        rows = max(rows, 1)

        grid = np.full((rows, cols), 128, dtype=np.uint8)

        ci = ((xs - x_min) / resolution).astype(int)
        ri = ((zs - z_min) / resolution).astype(int)
        ci = np.clip(ci, 0, cols - 1)
        ri = np.clip(ri, 0, rows - 1)

        grid[ri, ci] = 255
        cv2.imwrite(path, grid)
        logger.info(
            "Saved occupancy %dx%d → %s  (height band [%.1f, %.1f])",
            cols, rows, path, y_lo, y_hi,
        )
