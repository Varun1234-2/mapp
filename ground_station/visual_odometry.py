"""
Visual Odometry — ORB features, essential-matrix RANSAC, keyframe management.

Pipeline per frame
------------------
1. Detect ORB features.
2. Match against the *reference keyframe* using BFMatcher + ratio test.
3. Compute the essential matrix with RANSAC and recover (R, t).
   ``cv2.recoverPose`` returns *unit* translation — metric scale is applied
   from ``self._current_scale`` (updated externally via ``update_scale``).
4. Accumulate pose: T_wc_cur = T_wc_ref @ T_ref→cur.
5. If the median pixel parallax of inlier matches exceeds a threshold *and*
   at least ``min_kf_gap`` frames have passed, mark the frame as a new
   keyframe and update the reference.

Tracking loss
-------------
When inlier count falls below ``min_inliers``, the tracker *re-anchors*:
it adopts the current frame as a new keyframe at the last known pose and
continues.  This avoids accumulating garbage, but introduces a gap — IMU
integration (future) will bridge it.

Scale estimation (called externally by slam_mapper)
---------------------------------------------------
Compare the depth-network depth at triangulated-point pixel locations with
the triangulated depth (z component in the reference camera frame).
    scale = median(depth_network / depth_triangulated)
Smoothed with exponential moving average.

Coordinate convention
---------------------
All poses are **T_wc** — world-from-camera (see ``utils.py``).
``cv2.recoverPose`` returns (R, t) such that
    x_2 = R · x_1 + t
i.e. it maps points *from frame 1 (ref) into frame 2 (cur)*.
Therefore T_cur←ref = [R | t], and we need its inverse to accumulate:
    T_wc_cur = T_wc_ref @ inv(T_cur←ref)
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
    """Monocular ORB-based visual odometry with essential-matrix RANSAC.

    Parameters
    ----------
    K : 3×3 camera intrinsic matrix.
    n_features : ORB feature budget.
    ratio_thresh : Lowe's ratio test threshold.
    min_inliers : below this → tracking lost.
    parallax_thresh : median pixel shift to trigger a new keyframe.
    min_kf_gap : minimum frames between keyframes.
    scale_smoothing : EMA weight for keeping old scale (0 = instant, 1 = frozen).
    """

    def __init__(
        self,
        K: np.ndarray,
        n_features: int = 2000,
        ratio_thresh: float = 0.75,
        min_inliers: int = 30,
        parallax_thresh: float = 20.0,
        min_kf_gap: int = 5,
        scale_smoothing: float = 0.7,
    ):
        self.K = K.astype(np.float64)
        self.ratio_thresh = ratio_thresh
        self.min_inliers = min_inliers
        self.parallax_thresh = parallax_thresh
        self.min_kf_gap = min_kf_gap
        self.scale_smoothing = scale_smoothing

        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        # ---- state ----
        self._initialized = False
        self._ref_gray: np.ndarray | None = None
        self._ref_kp: list = []
        self._ref_des: np.ndarray | None = None
        self._ref_pose_wc = Pose.identity()

        self._current_scale: float = 1.0
        self._trajectory: list[Pose] = []
        self._keyframe_count: int = 0
        self._frames_since_kf: int = 0

    # ------------------------------------------------------------------ #
    #  Main entry                                                         #
    # ------------------------------------------------------------------ #

    def process(self, frame_bgr: np.ndarray) -> TrackingResult:
        """Process one frame and return tracking state."""
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        n_feat = len(kp) if kp else 0

        if des is None or n_feat < self.min_inliers:
            return self._make_lost(n_feat)

        # First frame → initialise
        if not self._initialized:
            return self._init_first(gray, kp, des)

        # Match current features against the reference keyframe
        matches = self._ratio_match(self._ref_des, des)

        if len(matches) < self.min_inliers:
            logger.warning("Tracking lost (%d matches)", len(matches))
            return self._reanchor(gray, kp, des)

        # Matched pixel coords
        pts_ref = np.float64([self._ref_kp[m.queryIdx].pt for m in matches])
        pts_cur = np.float64([kp[m.trainIdx].pt for m in matches])

        # Essential matrix + RANSAC
        E, emask = cv2.findEssentialMat(
            pts_ref, pts_cur, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )
        if E is None or emask is None:
            return self._make_lost(n_feat)

        if E.shape != (3, 3):
            E = E[:3, :3]

        emask = emask.ravel().astype(bool)
        n_inliers = int(emask.sum())
        if n_inliers < self.min_inliers:
            logger.warning("Essential mat: only %d inliers", n_inliers)
            return self._reanchor(gray, kp, des)

        # Recover R, t  (t is *unit length*)
        inlier_mask = emask.astype(np.uint8).reshape(-1, 1)
        _, R, t_unit, _ = cv2.recoverPose(
            E, pts_ref, pts_cur, self.K,
            mask=inlier_mask,
        )

        valid_inliers = (inlier_mask.ravel() > 0)
        n_inliers = int(valid_inliers.sum())
        if n_inliers < self.min_inliers:
            logger.warning("RecoverPose: only %d cheirality inliers", n_inliers)
            return self._reanchor(gray, kp, des)

        # Apply metric scale
        t_scaled = t_unit * self._current_scale

        # T_cur←ref  (maps ref-frame points into cur-frame)
        T_cur_ref = np.eye(4, dtype=np.float64)
        T_cur_ref[:3, :3] = R
        T_cur_ref[:3, 3] = t_scaled.ravel()

        # Invert → T_ref←cur, then accumulate
        T_ref_cur = np.eye(4, dtype=np.float64)
        T_ref_cur[:3, :3] = R.T
        T_ref_cur[:3, 3] = (-R.T @ t_scaled).ravel()

        pose_wc = Pose(self._ref_pose_wc.T @ T_ref_cur)

        # Keyframe decision: parallax of inlier matches
        par = float(np.median(np.linalg.norm(
            pts_cur[valid_inliers] - pts_ref[valid_inliers], axis=1
        )))
        self._frames_since_kf += 1
        is_kf = (par > self.parallax_thresh) and (
            self._frames_since_kf >= self.min_kf_gap
        )

        if is_kf:
            self._set_keyframe(gray, kp, des, pose_wc)

        self._trajectory.append(pose_wc)

        dt = time.perf_counter() - t0
        logger.debug(
            "VO: %d inliers, parallax %.1f px, kf=%s, %.1f ms",
            n_inliers, par, is_kf, dt * 1000,
        )
        return TrackingResult(
            pose_wc=pose_wc,
            is_keyframe=is_kf,
            is_lost=False,
            num_inliers=n_inliers,
            num_features=n_feat,
            keyframe_id=self._keyframe_count,
        )

    # ------------------------------------------------------------------ #
    #  Scale                                                               #
    # ------------------------------------------------------------------ #

    def update_scale(
        self,
        depth_network: np.ndarray,
        pts_3d_ref: np.ndarray,
        pts_2d_ref: np.ndarray,
    ) -> None:
        """Update metric scale from network depth vs triangulated depth.

        Args:
            depth_network : H×W metric depth from the depth network
                (at the *reference* keyframe).
            pts_3d_ref : N×3 triangulated points **in reference camera frame**.
            pts_2d_ref : N×2 pixel coords in the reference keyframe.
        """
        if len(pts_3d_ref) == 0:
            return
        h, w = depth_network.shape[:2]
        net_d, tri_d = [], []
        for pt2, pt3 in zip(pts_2d_ref, pts_3d_ref):
            u, v = int(round(pt2[0])), int(round(pt2[1]))
            if 0 <= u < w and 0 <= v < h and pt3[2] > 0.01:
                nd = float(depth_network[v, u])
                if nd > 0.01:
                    net_d.append(nd)
                    tri_d.append(pt3[2])

        if len(net_d) < 10:
            return

        raw_scale = float(np.median(np.array(net_d) / np.array(tri_d)))
        self._current_scale = (
            self.scale_smoothing * self._current_scale
            + (1 - self.scale_smoothing) * raw_scale
        )
        logger.info("Scale update: %.3f  (raw %.3f)", self._current_scale, raw_scale)

    # ------------------------------------------------------------------ #
    #  Triangulation helper (for scale estimation by slam_mapper)          #
    # ------------------------------------------------------------------ #

    def triangulate_against_ref(
        self, frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Triangulate ORB matches between *frame_bgr* and the current
        reference keyframe.

        Returns ``(pts_3d_ref, pts_2d_ref)`` — 3-D points in the
        reference camera frame and matching pixel coordinates in the
        reference image.  Returns ``None`` if there are too few matches.
        """
        if not self._initialized or self._ref_des is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None:
            return None

        matches = self._ratio_match(self._ref_des, des)
        if len(matches) < self.min_inliers:
            return None

        pts_ref = np.float64([self._ref_kp[m.queryIdx].pt for m in matches])
        pts_cur = np.float64([kp[m.trainIdx].pt for m in matches])

        E, emask = cv2.findEssentialMat(
            pts_ref, pts_cur, self.K,
            method=cv2.RANSAC, threshold=1.0,
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

        # Projection matrices (ref = identity)
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

        # Keep points with physically plausible positive depth in front of the camera
        good = valid_w & (pts3[:, 2] > 0.01) & (pts3[:, 2] < 150.0)
        if not np.any(good):
            return None
        return pts3[good], inl_ref[good]

    # ------------------------------------------------------------------ #
    #  Queries                                                             #
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
        self._initialized = True
        self._trajectory.append(pose)
        logger.info("VO initialised (%d features)", len(kp))
        return TrackingResult(
            pose_wc=pose, is_keyframe=True, is_lost=False,
            num_inliers=0, num_features=len(kp), keyframe_id=0,
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

    def _reanchor(self, gray, kp, des) -> TrackingResult:
        """Adopt current frame as reference at the last known pose."""
        logger.warning("Re-anchoring at last known pose")
        last = self._trajectory[-1] if self._trajectory else Pose.identity()
        self._set_keyframe(gray, kp, des, last)
        self._trajectory.append(last)
        return TrackingResult(
            pose_wc=last, is_keyframe=True, is_lost=True,
            num_inliers=0, num_features=len(kp),
            keyframe_id=self._keyframe_count,
        )

    def _make_lost(self, n_feat: int) -> TrackingResult:
        last = self._trajectory[-1] if self._trajectory else Pose.identity()
        return TrackingResult(
            pose_wc=last, is_keyframe=False, is_lost=True,
            num_inliers=0, num_features=n_feat,
            keyframe_id=self._keyframe_count,
        )

