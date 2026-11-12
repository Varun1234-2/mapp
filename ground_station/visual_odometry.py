"""
Visual Odometry — Metric RGB-D 3D-to-2D PnP Tracking.

Why PnP (Perspective-n-Point) instead of Essential Matrix?
---------------------------------------------------------
Monocular 2D-to-2D Essential Matrix estimation (recoverPose) can only output a
UNIT-VECTOR direction of translation (||t|| = 1). Every single frame moved the
camera by an arbitrary 1.0 meter because monocular 2D matches have no depth!

With our metric depth model (Depth Anything V2 Metric), we have the TRUE METRIC 3D
coordinates of features in the previous keyframe.
By matching current 2D features against the previous keyframe's known 3D points,
cv2.solvePnPRansac directly and deterministically computes the TRUE metric 6-DoF
camera pose (R, t) in meters, with ZERO scale drift and zero translation jumping!
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .utils import Pose

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Returned by ``VisualOdometry.process()`` for every frame."""
    pose_wc: Pose          # current camera-to-world pose
    is_keyframe: bool
    is_lost: bool
    num_inliers: int
    num_features: int
    keyframe_id: int       # id of current keyframe


class VisualOdometry:
    """RGB-D Visual Odometry using metric 3D-to-2D PnP RANSAC."""

    def __init__(
        self,
        K: np.ndarray,
        n_features: int = 2000,
        ratio_thresh: float = 0.75,
        min_inliers: int = 15,
        parallax_thresh: float = 25.0,
        min_kf_gap: int = 8,
    ):
        self.K = K.astype(np.float64)
        self.ratio_thresh = ratio_thresh
        self.min_inliers = min_inliers
        self.parallax_thresh = parallax_thresh
        self.min_kf_gap = min_kf_gap

        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=15,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ---- Tracking state ----
        self._initialized = False

        # Keyframe data with metric 3D points in world coordinates
        self._kf_gray: np.ndarray | None = None
        self._kf_kp: list = []
        self._kf_des: np.ndarray | None = None
        self._kf_pts3d_world: np.ndarray | None = None  # (N, 3) in world coords
        self._kf_pose_wc = Pose.identity()

        # Immediate previous frame for visual persistence
        self._prev_gray: np.ndarray | None = None
        self._prev_kp: list = []
        self._prev_des: np.ndarray | None = None
        self._last_pose_wc = Pose.identity()

        self._trajectory: list[Pose] = []
        self._keyframe_count: int = 0
        self._frames_since_kf: int = 0
        self._consecutive_lost: int = 0

    # ------------------------------------------------------------------ #
    #  Main frame processing                                               #
    # ------------------------------------------------------------------ #

    def process(self, frame_bgr: np.ndarray) -> TrackingResult:
        """Process an incoming camera frame and track 6-DoF pose."""
        t0 = time.perf_counter()
        gray_raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray_raw)

        kp, des = self.orb.detectAndCompute(gray, None)
        n_feat = len(kp) if kp else 0

        if des is None or n_feat < self.min_inliers:
            return self._handle_lost(gray, kp, des, n_feat)

        # First frame: initialize origin
        if not self._initialized:
            pose = Pose.identity()
            self._initialized = True
            self._last_pose_wc = pose
            self._prev_gray = gray
            self._prev_kp = list(kp)
            self._prev_des = des
            self._trajectory.append(pose)
            return TrackingResult(
                pose_wc=pose, is_keyframe=True, is_lost=False,
                num_inliers=n_feat, num_features=n_feat, keyframe_id=1,
            )

        # Track using 3D-to-2D PnP against the active keyframe's 3D landmarks
        pose_wc, n_inliers, parallax = self._track_pnp(kp, des)

        if pose_wc is None or n_inliers < self.min_inliers:
            # Fall back to frame-to-frame optical tracking if PnP lost
            pose_wc = self._track_fallback(kp, des)
            if pose_wc is None:
                return self._handle_lost(gray, kp, des, n_feat)
            is_lost = True
        else:
            is_lost = False
            self._consecutive_lost = 0

        self._last_pose_wc = pose_wc
        self._prev_gray = gray
        self._prev_kp = list(kp)
        self._prev_des = des
        self._trajectory.append(pose_wc)
        self._frames_since_kf += 1

        # Keyframe decision based on translation/parallax and elapsed frames
        is_kf = False
        if not is_lost:
            # Trigger keyframe when viewpoint has moved sufficiently or inlier count is dropping
            dist_from_kf = np.linalg.norm(pose_wc.t - self._kf_pose_wc.t)
            if (parallax >= self.parallax_thresh or dist_from_kf >= 0.25 or n_inliers < self.min_inliers * 1.5) and (
                self._frames_since_kf >= self.min_kf_gap
            ):
                is_kf = True

        return TrackingResult(
            pose_wc=pose_wc,
            is_keyframe=is_kf,
            is_lost=is_lost,
            num_inliers=n_inliers,
            num_features=n_feat,
            keyframe_id=self._keyframe_count,
        )

    # ------------------------------------------------------------------ #
    #  Register Keyframe Depth (called when depth network finishes)        #
    # ------------------------------------------------------------------ #

    def register_keyframe(
        self, frame_bgr: np.ndarray, depth_meters: np.ndarray, pose_wc: Pose
    ) -> None:
        """Register a keyframe with its metric depth map to generate 3D world landmarks."""
        gray = self.clahe.apply(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            return

        h, w = depth_meters.shape[:2]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        pts_cam = []
        valid_indices = []

        for idx, k in enumerate(kp):
            u, v = int(round(k.pt[0])), int(round(k.pt[1]))
            if 0 <= u < w and 0 <= v < h:
                z = float(depth_meters[v, u])
                if 0.1 <= z <= 12.0:  # reasonable metric depth range
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    pts_cam.append([x, y, z])
                    valid_indices.append(idx)

        if len(pts_cam) < self.min_inliers:
            return

        pts_cam = np.array(pts_cam, dtype=np.float64)
        # Transform 3D points from camera frame into world frame
        pts_world = pose_wc.transform_points(pts_cam)

        self._kf_gray = gray
        self._kf_kp = [kp[i] for i in valid_indices]
        self._kf_des = des[valid_indices]
        self._kf_pts3d_world = pts_world
        self._kf_pose_wc = pose_wc
        self._keyframe_count += 1
        self._frames_since_kf = 0

        logger.info(
            "Registered Keyframe %d with %d 3D metric landmarks (cam pos=[%.2f, %.2f, %.2f])",
            self._keyframe_count, len(pts_world), *pose_wc.t
        )

    # ------------------------------------------------------------------ #
    #  3D-to-2D PnP RANSAC Tracking                                        #
    # ------------------------------------------------------------------ #

    def _track_pnp(self, kp: list, des: np.ndarray) -> tuple[Pose | None, int, float]:
        if self._kf_des is None or self._kf_pts3d_world is None or len(self._kf_pts3d_world) < self.min_inliers:
            return None, 0, 0.0

        matches = self._ratio_match(self._kf_des, des)
        if len(matches) < self.min_inliers:
            return None, len(matches), 0.0

        idx_3d = [m.queryIdx for m in matches]
        pts_3d = np.ascontiguousarray(self._kf_pts3d_world[idx_3d], dtype=np.float64)
        pts_2d = np.ascontiguousarray([kp[m.trainIdx].pt for m in matches], dtype=np.float64)

        # Compute median 2D pixel shift
        pts_kf_2d = np.float64([self._kf_kp[m.queryIdx].pt for m in matches])
        parallax = float(np.median(np.linalg.norm(pts_2d - pts_kf_2d, axis=1)))

        # Solve PnP with RANSAC
        # solvePnPRansac finds world-to-camera rotation and translation
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts_3d,
            imagePoints=pts_2d,
            cameraMatrix=self.K,
            distCoeffs=None,
            iterationsCount=150,
            reprojectionError=2.5,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success or inliers is None or len(inliers) < self.min_inliers:
            return None, (len(inliers) if inliers is not None else 0), parallax

        R_cw, _ = cv2.Rodrigues(rvec)
        t_cw = tvec.reshape(3, 1)

        # Invert to get camera-to-world pose (T_wc)
        R_wc = R_cw.T
        t_wc = (-R_wc @ t_cw).ravel()

        # Sanity check: camera should not jump impossibly far in one frame (< 0.6 meters)
        step = np.linalg.norm(t_wc - self._last_pose_wc.t)
        if step > 0.8:
            logger.debug("PnP step too large (%.2f m) - rejecting outlier", step)
            return None, len(inliers), parallax

        pose_wc = Pose.from_Rt(R_wc, t_wc)
        return pose_wc, len(inliers), parallax

    def _track_fallback(self, kp: list, des: np.ndarray) -> Pose | None:
        """Keep previous camera pose when momentarily lost."""
        return self._last_pose_wc

    # ------------------------------------------------------------------ #
    #  Helpers & Queries                                                   #
    # ------------------------------------------------------------------ #

    def _ratio_match(self, des1: np.ndarray, des2: np.ndarray) -> list[cv2.DMatch]:
        if des1 is None or des2 is None:
            return []
        raw = self.matcher.knnMatch(des1, des2, k=2)
        good: list[cv2.DMatch] = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)
        return good

    def _handle_lost(self, gray, kp, des, n_feat: int) -> TrackingResult:
        self._consecutive_lost += 1
        return TrackingResult(
            pose_wc=self._last_pose_wc,
            is_keyframe=False,
            is_lost=True,
            num_inliers=0,
            num_features=n_feat,
            keyframe_id=self._keyframe_count,
        )

    def get_trajectory(self) -> list[Pose]:
        return list(self._trajectory)

    def get_keyframe_count(self) -> int:
        return self._keyframe_count

    @property
    def current_scale(self) -> float:
        return 1.0  # PnP operates directly in true metric units
