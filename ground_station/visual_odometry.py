"""
Visual Odometry — ORB features, essential-matrix RANSAC, keyframe management.

Tracking Pipeline
-----------------
1. Detect ORB features (using CLAHE to boost contrast in varied lighting).
2. Match against BOTH the immediate previous frame (for smooth short-baseline tracking)
   AND the reference keyframe (for parallax, triangulation & keyframe creation).
3. Compute essential matrix via RANSAC with robust inlier thresholds.
4. Scale recovery via metric depth network cross-triangulation.
5. Re-anchoring: gracefully bridges momentary tracking loss without runaway keyframe loops.
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
    pose_wc: Pose          # current camera-in-world
    is_keyframe: bool
    is_lost: bool
    num_inliers: int
    num_features: int
    keyframe_id: int       # id of the current / most-recent keyframe


class VisualOdometry:
    """Monocular ORB-based visual odometry with keyframe management.

    Parameters
    ----------
    K : 3×3 camera intrinsic matrix.
    n_features : ORB feature budget (default 2500).
    ratio_thresh : Lowe's ratio test threshold (default 0.78).
    min_inliers : below this → tracking lost (default 15).
    parallax_thresh : median pixel shift to trigger a new keyframe (default 25.0).
    min_kf_gap : minimum frames between keyframes (default 10).
    scale_smoothing : EMA weight for keeping old scale (default 0.8).
    """

    def __init__(
        self,
        K: np.ndarray,
        n_features: int = 2500,
        ratio_thresh: float = 0.78,
        min_inliers: int = 15,
        parallax_thresh: float = 25.0,
        min_kf_gap: int = 10,
        scale_smoothing: float = 0.8,
    ):
        self.K = K.astype(np.float64)
        self.ratio_thresh = ratio_thresh
        self.min_inliers = min_inliers
        self.parallax_thresh = parallax_thresh
        self.min_kf_gap = min_kf_gap
        self.scale_smoothing = scale_smoothing

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

        # ---- state ----
        self._initialized = False
        
        # Reference Keyframe (used for parallax, triangulation & mapping)
        self._ref_gray: np.ndarray | None = None
        self._ref_kp: list = []
        self._ref_des: np.ndarray | None = None
        self._ref_pose_wc = Pose.identity()

        # Previous Frame (used for continuous frame-to-frame tracking)
        self._prev_gray: np.ndarray | None = None
        self._prev_kp: list = []
        self._prev_des: np.ndarray | None = None
        self._prev_pose_wc = Pose.identity()

        self._current_scale: float = 1.0
        self._trajectory: list[Pose] = []
        self._keyframe_count: int = 0
        self._frames_since_kf: int = 0
        self._consecutive_lost: int = 0

    # ------------------------------------------------------------------ #
    #  Main entry                                                         #
    # ------------------------------------------------------------------ #

    def process(self, frame_bgr: np.ndarray) -> TrackingResult:
        """Process one frame and return tracking state."""
        t0 = time.perf_counter()
        gray_raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray_raw)
        
        kp, des = self.orb.detectAndCompute(gray, None)
        n_feat = len(kp) if kp else 0

        if des is None or n_feat < self.min_inliers:
            return self._handle_lost(gray, kp, des, n_feat)

        # First frame → initialise
        if not self._initialized:
            return self._init_first(gray, kp, des)

        # 1. First attempt: match against previous frame for continuous motion tracking
        pose_candidate, n_inliers_prev, inlier_matches_prev = self._estimate_motion(
            self._prev_kp, self._prev_des, kp, des, self._prev_pose_wc
        )

        # 2. Match against reference keyframe to determine parallax / keyframe creation
        _, n_inliers_ref, inlier_matches_ref = self._estimate_motion(
            self._ref_kp, self._ref_des, kp, des, self._ref_pose_wc
        )

        # Determine best pose estimate
        if pose_candidate is not None and n_inliers_prev >= self.min_inliers:
            pose_wc = pose_candidate
            n_inliers = n_inliers_prev
            self._consecutive_lost = 0
        else:
            # Fall back to reference keyframe direct match
            pose_ref_candidate, n_inl_ref, _ = self._estimate_motion(
                self._ref_kp, self._ref_des, kp, des, self._ref_pose_wc
            )
            if pose_ref_candidate is not None and n_inl_ref >= self.min_inliers:
                pose_wc = pose_ref_candidate
                n_inliers = n_inl_ref
                self._consecutive_lost = 0
            else:
                return self._handle_lost(gray, kp, des, n_feat)

        # Update previous frame tracking cache
        self._prev_gray = gray
        self._prev_kp = list(kp)
        self._prev_des = des
        self._prev_pose_wc = pose_wc
        self._trajectory.append(pose_wc)
        self._frames_since_kf += 1

        # Keyframe decision based on parallax relative to reference keyframe
        is_kf = False
        if inlier_matches_ref is not None and len(inlier_matches_ref) >= self.min_inliers:
            pts_ref = np.float64([self._ref_kp[m.queryIdx].pt for m in inlier_matches_ref])
            pts_cur = np.float64([kp[m.trainIdx].pt for m in inlier_matches_ref])
            par = float(np.median(np.linalg.norm(pts_cur - pts_ref, axis=1)))
            
            if (par >= self.parallax_thresh and self._frames_since_kf >= self.min_kf_gap) or (
                len(inlier_matches_ref) < self.min_inliers * 1.5 and self._frames_since_kf >= self.min_kf_gap
            ):
                is_kf = True
                self._set_keyframe(gray, kp, des, pose_wc)

        dt = time.perf_counter() - t0
        return TrackingResult(
            pose_wc=pose_wc,
            is_keyframe=is_kf,
            is_lost=False,
            num_inliers=n_inliers,
            num_features=n_feat,
            keyframe_id=self._keyframe_count,
        )

    # ------------------------------------------------------------------ #
    #  Motion Estimation Helper                                            #
    # ------------------------------------------------------------------ #

    def _estimate_motion(
        self,
        kp_from: list,
        des_from: np.ndarray | None,
        kp_to: list,
        des_to: np.ndarray | None,
        pose_from_wc: Pose,
    ) -> tuple[Pose | None, int, list[cv2.DMatch] | None]:
        if des_from is None or des_to is None or len(kp_from) < self.min_inliers:
            return None, 0, None

        matches = self._ratio_match(des_from, des_to)
        if len(matches) < self.min_inliers:
            return None, len(matches), None

        pts_from = np.float64([kp_from[m.queryIdx].pt for m in matches])
        pts_to = np.float64([kp_to[m.trainIdx].pt for m in matches])

        # Essential matrix via RANSAC with slightly wider threshold for mobile cameras
        E, emask = cv2.findEssentialMat(
            pts_from, pts_to, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.8,
        )
        if E is None or emask is None:
            return None, 0, None

        if E.shape != (3, 3):
            E = E[:3, :3]

        emask = emask.ravel().astype(bool)
        if int(emask.sum()) < self.min_inliers:
            return None, int(emask.sum()), None

        inlier_mask = emask.astype(np.uint8).reshape(-1, 1)
        _, R, t_unit, _ = cv2.recoverPose(
            E, pts_from, pts_to, self.K,
            mask=inlier_mask,
        )

        valid_inliers = (inlier_mask.ravel() > 0)
        n_inliers = int(valid_inliers.sum())
        if n_inliers < self.min_inliers:
            return None, n_inliers, None

        # Apply metric scale
        t_scaled = t_unit * self._current_scale

        # T_to<-from: maps points from 'from' frame into 'to' frame
        # Invert to map into 'from' coordinates and accumulate into world pose
        T_from_to = np.eye(4, dtype=np.float64)
        T_from_to[:3, :3] = R.T
        T_from_to[:3, 3] = (-R.T @ t_scaled).ravel()

        pose_wc = Pose(pose_from_wc.T @ T_from_to)
        inlier_matches = [matches[i] for i, v in enumerate(valid_inliers) if v]
        return pose_wc, n_inliers, inlier_matches

    # ------------------------------------------------------------------ #
    #  Scale Updates                                                       #
    # ------------------------------------------------------------------ #

    def update_scale(
        self,
        depth_network: np.ndarray,
        pts_3d_ref: np.ndarray,
        pts_2d_ref: np.ndarray,
    ) -> None:
        """Update metric scale from network depth vs triangulated depth."""
        if len(pts_3d_ref) == 0:
            return
        h, w = depth_network.shape[:2]
        net_d, tri_d = [], []
        for pt2, pt3 in zip(pts_2d_ref, pts_3d_ref):
            u, v = int(round(pt2[0])), int(round(pt2[1]))
            if 0 <= u < w and 0 <= v < h and pt3[2] > 0.05:
                nd = float(depth_network[v, u])
                if nd > 0.05:
                    net_d.append(nd)
                    tri_d.append(pt3[2])

        if len(net_d) < 8:
            return

        raw_scale = float(np.median(np.array(net_d) / np.array(tri_d)))
        if 0.1 <= raw_scale <= 10.0:  # ignore wild spikes
            self._current_scale = (
                self.scale_smoothing * self._current_scale
                + (1 - self.scale_smoothing) * raw_scale
            )
            logger.info("Scale updated: %.3f (raw: %.3f)", self._current_scale, raw_scale)

    def triangulate_against_ref(
        self, frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Triangulate matches against reference keyframe for scale estimation."""
        if not self._initialized or self._ref_des is None:
            return None

        gray = self.clahe.apply(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < self.min_inliers:
            return None

        matches = self._ratio_match(self._ref_des, des)
        if len(matches) < self.min_inliers:
            return None

        pts_ref = np.float64([self._ref_kp[m.queryIdx].pt for m in matches])
        pts_cur = np.float64([kp[m.trainIdx].pt for m in matches])

        E, emask = cv2.findEssentialMat(
            pts_ref, pts_cur, self.K,
            method=cv2.RANSAC, threshold=1.8,
        )
        if E is None or emask is None:
            return None

        if E.shape != (3, 3):
            E = E[:3, :3]

        emask = emask.ravel().astype(bool)
        inlier_mask = emask.astype(np.uint8).reshape(-1, 1)
        _, R, t, _ = cv2.recoverPose(
            E, pts_ref, pts_cur, self.K,
            mask=inlier_mask,
        )

        valid_inliers = (inlier_mask.ravel() > 0)
        if valid_inliers.sum() < self.min_inliers:
            return None

        P1 = self.K @ np.eye(3, 4, dtype=np.float64)
        P2 = self.K @ np.hstack([R, t])

        inl_ref = pts_ref[valid_inliers]
        inl_cur = pts_cur[valid_inliers]

        pts4 = cv2.triangulatePoints(P1, P2, inl_ref.T, inl_cur.T)
        w_coord = pts4[3]
        valid_w = np.abs(w_coord) > 1e-6
        if not np.any(valid_w):
            return None

        pts3 = np.zeros((pts4.shape[1], 3), dtype=np.float64)
        pts3[valid_w] = (pts4[:3, valid_w] / w_coord[valid_w]).T

        good = valid_w & (pts3[:, 2] > 0.05) & (pts3[:, 2] < 80.0)
        if not np.any(good):
            return None
        return pts3[good], inl_ref[good]

    # ------------------------------------------------------------------ #
    #  State & Queries                                                     #
    # ------------------------------------------------------------------ #

    def get_trajectory(self) -> list[Pose]:
        return list(self._trajectory)

    def get_keyframe_count(self) -> int:
        return self._keyframe_count

    @property
    def current_scale(self) -> float:
        return self._current_scale

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
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

    def _init_first(self, gray, kp, des) -> TrackingResult:
        pose = Pose.identity()
        self._set_keyframe(gray, kp, des, pose)
        self._prev_gray = gray
        self._prev_kp = list(kp)
        self._prev_des = des
        self._prev_pose_wc = pose
        self._initialized = True
        self._trajectory.append(pose)
        logger.info("VO initialised (%d features)", len(kp))
        return TrackingResult(
            pose_wc=pose, is_keyframe=True, is_lost=False,
            num_inliers=0, num_features=len(kp), keyframe_id=1,
        )

    def _set_keyframe(self, gray, kp, des, pose_wc: Pose) -> None:
        self._ref_gray = gray.copy()
        self._ref_kp = list(kp)
        self._ref_des = des.copy()
        self._ref_pose_wc = pose_wc
        self._keyframe_count += 1
        self._frames_since_kf = 0
        logger.info(
            "Keyframe %d: %d feats, t=[%.2f, %.2f, %.2f]",
            self._keyframe_count, len(kp), *pose_wc.t,
        )

    def _handle_lost(self, gray, kp, des, n_feat: int) -> TrackingResult:
        """Handle tracking loss without polluting keyframes or thrashing state."""
        self._consecutive_lost += 1
        last = self._trajectory[-1] if self._trajectory else Pose.identity()

        # If lost for more than 15 consecutive frames, re-anchor gently
        if self._consecutive_lost > 15 and des is not None and len(kp) >= self.min_inliers:
            logger.warning("Re-anchoring after sustained tracking loss")
            self._set_keyframe(gray, kp, des, last)
            self._prev_gray = gray
            self._prev_kp = list(kp)
            self._prev_des = des
            self._prev_pose_wc = last
            self._consecutive_lost = 0
            return TrackingResult(
                pose_wc=last, is_keyframe=True, is_lost=True,
                num_inliers=0, num_features=n_feat,
                keyframe_id=self._keyframe_count,
            )

        return TrackingResult(
            pose_wc=last, is_keyframe=False, is_lost=True,
            num_inliers=0, num_features=n_feat,
            keyframe_id=self._keyframe_count,
        )
