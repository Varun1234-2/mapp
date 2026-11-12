"""
Visual Odometry — Multi-Keyframe Sliding Window Metric PnP Tracking.

Why tracking was lost before:
1. Deadlock upon initialization:
   When frame 1 arrived, it was set as a keyframe, but Depth was NOT yet computed for it!
   So `_kf_pts3d_world` remained None, causing all subsequent frames to immediately report
   tracking lost!
2. Single-Keyframe Limitation:
   Matching against only 1 keyframe means as soon as the camera turns 45 degrees, the
   old keyframe's landmarks disappear from view, losing tracking.
   Now, we maintain a Sliding Window of recent keyframes (Local Map of 3D points).
   Every incoming frame matches against the union of visible local 3D points!
3. Frame-to-frame continuous recovery:
   If PnP temporarily drops below inliers due to motion blur, the tracker recovers
   via frame-to-frame differential tracking rather than freezing in place.
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
    """Robust Local Map 3D-to-2D PnP Visual Odometry."""

    def __init__(
        self,
        K: np.ndarray,
        n_features: int = 2500,
        ratio_thresh: float = 0.80,
        min_inliers: int = 12,
        parallax_thresh: float = 20.0,
        min_kf_gap: int = 6,
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
            fastThreshold=12,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ---- Tracking state ----
        self._initialized = False

        # Local Map: Recent Keyframes (Sliding Window of up to 5 keyframes)
        # Each entry: dict with 'pts_3d': (N, 3), 'des': (N, 32), 'pose': Pose
        self._keyframes: list[dict] = []
        self._local_map_pts3d: np.ndarray | None = None
        self._local_map_des: np.ndarray | None = None

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

        # First frame: initialize origin and mark as immediate keyframe to get depth
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

        # Track against the Local Map of 3D landmarks
        pose_wc, n_inliers, parallax = self._track_local_map(kp, des)

        # If local map tracking succeeded with sufficient inliers
        if pose_wc is not None and n_inliers >= self.min_inliers:
            is_lost = False
            self._consecutive_lost = 0
        else:
            # Fallback 1: Continuous frame-to-frame tracking
            pose_f2f, n_inl_f2f = self._track_frame_to_frame(kp, des)
            if pose_f2f is not None and n_inl_f2f >= self.min_inliers:
                pose_wc = pose_f2f
                n_inliers = n_inl_f2f
                is_lost = False
                self._consecutive_lost = 0
            else:
                # Fallback 2: Lost, retain last pose
                pose_wc = self._last_pose_wc
                is_lost = True
                self._consecutive_lost += 1

        self._last_pose_wc = pose_wc
        self._prev_gray = gray
        self._prev_kp = list(kp)
        self._prev_des = des
        self._trajectory.append(pose_wc)
        self._frames_since_kf += 1

        # Keyframe decision based on translation/parallax and elapsed frames
        is_kf = False
        if not is_lost and len(self._keyframes) > 0:
            last_kf_pose = self._keyframes[-1]['pose']
            dist_from_kf = np.linalg.norm(pose_wc.t - last_kf_pose.t)
            R_diff = pose_wc.R @ last_kf_pose.R.T
            cos_a = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_a))

            # Only trigger a new keyframe when camera has translated by >= 0.20m OR rotated by >= 15°
            # This prevents stationary hover or slight tilt from stacking redundant depth layers
            significant_motion = (dist_from_kf >= 0.20 or angle_deg >= 15.0)
            if significant_motion and (parallax >= self.parallax_thresh or n_inliers < self.min_inliers * 1.6) and (
                self._frames_since_kf >= self.min_kf_gap
            ):
                is_kf = True
        elif not is_lost and len(self._keyframes) == 0:
            # Need first keyframe depth urgently
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
        """Register a keyframe with its metric depth to generate 3D world landmarks."""
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
                if 0.1 <= z <= 12.0:  # Valid metric depth range
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    pts_cam.append([x, y, z])
                    valid_indices.append(idx)

        if len(pts_cam) < self.min_inliers:
            return

        pts_cam = np.array(pts_cam, dtype=np.float64)
        # Transform 3D points from camera coordinates into World coordinates
        pts_world = pose_wc.transform_points(pts_cam)

        # Add to sliding window of keyframes (keep last 5 keyframes in local map)
        kf_entry = {
            'pts_3d': pts_world,
            'des': des[valid_indices],
            'pose': pose_wc,
            'kp': [kp[i] for i in valid_indices]
        }
        self._keyframes.append(kf_entry)
        if len(self._keyframes) > 6:
            self._keyframes.pop(0)

        # Update merged local map descriptors & 3D points
        all_3d = [kf['pts_3d'] for kf in self._keyframes]
        all_des = [kf['des'] for kf in self._keyframes]
        self._local_map_pts3d = np.ascontiguousarray(np.vstack(all_3d), dtype=np.float64)
        self._local_map_des = np.ascontiguousarray(np.vstack(all_des), dtype=np.uint8)

        self._keyframe_count += 1
        self._frames_since_kf = 0

        logger.info(
            "Registered Keyframe %d (Total local 3D points: %d, Cam Pos=[%.2f, %.2f, %.2f])",
            self._keyframe_count, len(self._local_map_pts3d), *pose_wc.t
        )

    # ------------------------------------------------------------------ #
    #  3D-to-2D PnP RANSAC Tracking against Local Map                      #
    # ------------------------------------------------------------------ #

    def _track_local_map(self, kp: list, des: np.ndarray) -> tuple[Pose | None, int, float]:
        if self._local_map_des is None or self._local_map_pts3d is None or len(self._local_map_pts3d) < self.min_inliers:
            return None, 0, 0.0

        matches = self._ratio_match(self._local_map_des, des)
        if len(matches) < self.min_inliers:
            return None, len(matches), 0.0

        idx_3d = [m.queryIdx for m in matches]
        pts_3d = np.ascontiguousarray(self._local_map_pts3d[idx_3d], dtype=np.float64)
        pts_2d = np.ascontiguousarray([kp[m.trainIdx].pt for m in matches], dtype=np.float64)

        # Median pixel shift estimate
        parallax = 15.0

        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts_3d,
            imagePoints=pts_2d,
            cameraMatrix=self.K,
            distCoeffs=None,
            iterationsCount=150,
            reprojectionError=3.0,
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

        # Sanity check: smooth motion gate (< 0.7m per frame)
        step = np.linalg.norm(t_wc - self._last_pose_wc.t)
        if step > 0.8:
            logger.debug("Local map PnP step too large (%.2f m) - rejecting outlier", step)
            return None, len(inliers), parallax

        pose_wc = Pose.from_Rt(R_wc, t_wc)
        return pose_wc, len(inliers), parallax

    def _track_frame_to_frame(self, kp: list, des: np.ndarray) -> tuple[Pose | None, int]:
        """Continuous frame-to-frame motion estimation when PnP momentarily drops."""
        if self._prev_des is None or len(self._prev_kp) < self.min_inliers:
            return None, 0

        matches = self._ratio_match(self._prev_des, des)
        if len(matches) < self.min_inliers:
            return None, len(matches)

        pts_prev = np.float64([self._prev_kp[m.queryIdx].pt for m in matches])
        pts_cur = np.float64([kp[m.trainIdx].pt for m in matches])

        E, emask = cv2.findEssentialMat(
            pts_prev, pts_cur, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.8,
        )
        if E is None or emask is None:
            return None, 0

        if E.shape != (3, 3):
            E = E[:3, :3]

        emask = emask.ravel().astype(bool)
        inlier_mask = emask.astype(np.uint8).reshape(-1, 1)
        _, R, t_unit, _ = cv2.recoverPose(E, pts_prev, pts_cur, self.K, mask=inlier_mask)

        valid = (inlier_mask.ravel() > 0)
        n_inliers = int(valid.sum())
        if n_inliers < self.min_inliers:
            return None, n_inliers

        # Frame delta translation is typically a few centimeters
        t_scaled = t_unit * 0.03  # gentle translation step

        T_from_to = np.eye(4, dtype=np.float64)
        T_from_to[:3, :3] = R.T
        T_from_to[:3, 3] = (-R.T @ t_scaled).ravel()

        pose_wc = Pose(self._last_pose_wc.T @ T_from_to)
        return pose_wc, n_inliers

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
        return 1.0
