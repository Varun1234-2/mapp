"""
Live visualisation — Open3D 3-D map window + OpenCV 2-D HUD.

Open3D runs non-blocking: ``poll_events()`` and ``update_renderer()`` are
called once per iteration from the main loop.  The cloud and trajectory
line-set are updated in-place so the window never freezes.

The OpenCV HUD shows the current camera frame with an overlay:
FPS, keyframe count, inlier count, tracking status, and scale.
"""

from __future__ import annotations

import time
import logging
from typing import Sequence

import cv2
import numpy as np
import open3d as o3d

from .utils import Pose

logger = logging.getLogger(__name__)


class MapVisualizer:
    """Combined Open3D + OpenCV live visualisation.

    Call ``setup()`` once, then ``update()`` every frame, and finally
    ``destroy()`` on exit.
    """

    def __init__(self, window_name: str = "SLAM Map"):
        self._win_name = window_name
        self._vis: o3d.visualization.Visualizer | None = None

        # Persistent geometries (mutated in-place)
        self._cloud_geom = o3d.geometry.PointCloud()
        self._traj_geom = o3d.geometry.LineSet()
        self._cloud_added = False
        self._traj_added = False

        self._fps_filter = 0.0
        self._last_time = time.perf_counter()

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """Create the Open3D window (must be called from the main thread)."""
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(
            window_name=self._win_name, width=960, height=720,
        )
        opt = self._vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        logger.info("Open3D visualiser created")

    def update(
        self,
        frame_bgr: np.ndarray,
        num_inliers: int,
        num_features: int,
        is_lost: bool,
        keyframe_count: int,
        scale: float,
        cloud: o3d.geometry.PointCloud,
        trajectory: Sequence[Pose],
    ) -> bool:
        """Push new data to both windows.  Returns False if user closed O3D."""
        # ---- Open3D ----
        if not self._update_3d(cloud, trajectory):
            return False

        # ---- OpenCV HUD ----
        self._update_hud(
            frame_bgr, num_inliers, num_features,
            is_lost, keyframe_count, scale,
        )
        return True

    def destroy(self) -> None:
        if self._vis is not None:
            self._vis.destroy_window()
        cv2.destroyAllWindows()

    def should_quit_cv(self) -> bool:
        """Check if 'q' was pressed in the OpenCV window."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return True
        if key == ord("s"):
            return False  # 's' handled by caller for saving
        return False

    def last_key(self) -> int:
        """Return the last key pressed in OpenCV, or -1."""
        return cv2.waitKey(1) & 0xFF

    # ------------------------------------------------------------------ #
    #  Open3D                                                              #
    # ------------------------------------------------------------------ #

    def _update_3d(
        self,
        cloud: o3d.geometry.PointCloud,
        trajectory: Sequence[Pose],
    ) -> bool:
        vis = self._vis
        if vis is None:
            return True

        # Update cloud
        if len(cloud.points) > 0:
            self._cloud_geom.points = cloud.points
            self._cloud_geom.colors = cloud.colors
            if not self._cloud_added:
                vis.add_geometry(self._cloud_geom)
                self._cloud_added = True
            else:
                vis.update_geometry(self._cloud_geom)

        # Update trajectory as a line-set
        if len(trajectory) >= 2:
            centres = np.array([p.t for p in trajectory], dtype=np.float64)
            n = len(centres)
            lines = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
            colors = np.array([[0.0, 1.0, 0.0]] * len(lines), dtype=np.float64)  # green

            self._traj_geom.points = o3d.utility.Vector3dVector(
                np.ascontiguousarray(centres, dtype=np.float64)
            )
            self._traj_geom.lines = o3d.utility.Vector2iVector(
                np.ascontiguousarray(lines, dtype=np.int32)
            )
            self._traj_geom.colors = o3d.utility.Vector3dVector(
                np.ascontiguousarray(colors, dtype=np.float64)
            )
            if not self._traj_added:
                vis.add_geometry(self._traj_geom)
                self._traj_added = True
            else:
                vis.update_geometry(self._traj_geom)

        vis.poll_events()
        vis.update_renderer()

        # Open3D returns False from poll_events when window is closed on
        # some platforms, but not all.  We cannot rely on it.
        return True

    # ------------------------------------------------------------------ #
    #  OpenCV HUD                                                          #
    # ------------------------------------------------------------------ #

    def _update_hud(
        self,
        frame: np.ndarray,
        n_inliers: int,
        n_features: int,
        is_lost: bool,
        kf_count: int,
        scale: float,
    ) -> None:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        fps = 1.0 / max(dt, 1e-6)
        self._fps_filter = 0.9 * self._fps_filter + 0.1 * fps

        hud = frame.copy()
        h, w = hud.shape[:2]

        # Status colour
        colour = (0, 0, 255) if is_lost else (0, 255, 0)
        status = "LOST" if is_lost else "TRACKING"

        lines = [
            f"FPS: {self._fps_filter:.1f}",
            f"Status: {status}",
            f"Features: {n_features}",
            f"Inliers: {n_inliers}",
            f"Keyframes: {kf_count}",
            f"Scale: {scale:.3f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                hud, line, (10, 25 + 25 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
            )

        cv2.imshow("SLAM Feed", hud)

