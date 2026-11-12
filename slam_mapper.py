#!/usr/bin/env python3
"""
slam_mapper.py — Real-time monocular SLAM with metric depth for live camera streams.

Ground-station entry point. Reads a live camera stream (phone IP camera /
webcam / drone UDP stream), runs ORB-based visual odometry, Depth Anything V2
metric depth, and builds a coloured 3-D point-cloud map in real time.

Live Stream Usage Examples
--------------------------
  # Phone camera via IP Webcam (Android)
  python slam_mapper.py --src "http://192.168.1.42:8080/video" --env indoor

  # Phone camera via RTSP (iOS / Android)
  python slam_mapper.py --src "rtsp://192.168.1.42:8554/live" --env indoor

  # Local USB/built-in webcam (index 0)
  python slam_mapper.py --src 0 --env indoor --hfov 66

  # Pi drone live stream over UDP
  python slam_mapper.py --src "udp://@:5600" --env outdoor --calib calib.npz

Outputs
-------
  Saved under: records/<TIMESTAMP_YYYYMMDD_HHMMSS>/
    - map.ply          (coloured 3D point cloud)
    - trajectory.txt   (KITTI-format 6-DoF camera poses)
    - occupancy.png    (top-down 2D floorplan occupancy grid)
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time

import cv2
import numpy as np

# ---- ground_station package ------------------------------------------------
from ground_station.video_reader import VideoReader
from ground_station.visual_odometry import VisualOdometry, TrackingResult
from ground_station.depth_estimator import DepthEstimator
from ground_station.dense_mapper import DenseMapper
from ground_station.visualizer import MapVisualizer
from ground_station.imu_interface import NullIMU, IMUInterface
from ground_station.utils import make_camera_matrix, load_calibration, Pose


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time monocular SLAM mapper for live camera streams",
    )
    p.add_argument(
        "--src", required=True,
        help="Live camera source: URL (HTTP/RTSP/UDP) or webcam index (e.g. 0).",
    )
    p.add_argument(
        "--env", choices=["indoor", "outdoor"], default="indoor",
        help="Environment — selects depth model and limits (default: indoor).",
    )
    p.add_argument(
        "--calib", default=None,
        help="Path to calib.npz (from calibrate.py). Overrides --hfov.",
    )
    p.add_argument(
        "--hfov", type=float, default=66.0,
        help="Horizontal FOV estimate in degrees (used if no calib, default 66).",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Base directory for output records (default: records/<timestamp>).",
    )
    p.add_argument(
        "--voxel-size", type=float, default=None,
        help="Voxel down-sample size in metres (default: 0.02 indoor, 0.05 outdoor).",
    )
    p.add_argument(
        "--depth-subsample", type=int, default=2,
        help="Take every N-th pixel when back-projecting depth (default 2).",
    )
    p.add_argument(
        "--device", default="cuda",
        help="Torch device for depth network (default: cuda).",
    )
    p.add_argument(
        "--no-vis", action="store_true",
        help="Disable Open3D / OpenCV visualisation (headless mode).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="DEBUG-level logging.",
    )
    return p.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_camera_matrix(
    args: argparse.Namespace, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (K, dist) — dist may be None when using HFOV fallback."""
    if args.calib and os.path.isfile(args.calib):
        K, dist = load_calibration(args.calib)
        logging.info("Loaded calibration from %s", args.calib)
        return K, dist
    K = make_camera_matrix(args.hfov, width, height)
    logging.info("Using HFOV estimate %.1f° → fx=fy=%.1f", args.hfov, K[0, 0])
    return K, None


def get_record_dir(custom_dir: str | None = None) -> str:
    """Generate timestamped session folder under records/ (e.g. records/20261112_143000)."""
    if custom_dir:
        os.makedirs(custom_dir, exist_ok=True)
        return custom_dir

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("records", timestamp)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def save_outputs(
    out_dir: str,
    mapper: DenseMapper,
    trajectory: list[Pose],
) -> None:
    """Write map.ply, trajectory.txt, and occupancy.png into the session directory."""
    os.makedirs(out_dir, exist_ok=True)
    ply_path = os.path.join(out_dir, "map.ply")
    traj_path = os.path.join(out_dir, "trajectory.txt")
    occ_path = os.path.join(out_dir, "occupancy.png")

    mapper.save_ply(ply_path)
    mapper.save_occupancy(occ_path)

    with open(traj_path, "w") as f:
        for pose in trajectory:
            f.write(pose.to_kitti_row() + "\n")
    logging.info("Saved %d poses → %s", len(trajectory), traj_path)
    logging.info("All session files stored in: %s", os.path.abspath(out_dir))


# --------------------------------------------------------------------------- #
#  Main loop                                                                   #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger("slam_mapper")

    # ---- initialize output session folder (records/YYYYMMDD_HHMMSS) ---------
    session_dir = get_record_dir(args.out_dir)
    log.info("Session records folder initialized: %s", os.path.abspath(session_dir))

    # ---- video reader -------------------------------------------------------
    reader = VideoReader(args.src)
    K, dist = build_camera_matrix(args, reader.width, reader.height)

    # ---- undistortion map (if calibrated) -----------------------------------
    undist_map1 = undist_map2 = None
    if dist is not None:
        K_mat = np.ascontiguousarray(K, dtype=np.float64)
        dist_vec = np.ascontiguousarray(dist, dtype=np.float64)
        undist_map1, undist_map2 = cv2.initUndistortRectifyMap(
            K_mat, dist_vec, None, K_mat,
            (reader.width, reader.height), cv2.CV_32FC1,
        )
        log.info("Undistortion maps built")

    # ---- pipeline components ------------------------------------------------
    voxel = args.voxel_size or (0.02 if args.env == "indoor" else 0.05)

    vo = VisualOdometry(K)
    depth_est = DepthEstimator(env=args.env, device=args.device)
    mapper = DenseMapper(K, voxel_size=voxel, subsample=args.depth_subsample)
    imu: IMUInterface = NullIMU()
    imu.start()

    # ---- visualiser ---------------------------------------------------------
    vis: MapVisualizer | None = None
    if not args.no_vis:
        vis = MapVisualizer()
        vis.setup()

    # ---- timing accumulators ------------------------------------------------
    t_vo_total = 0.0
    t_depth_total = 0.0
    n_frames = 0
    n_depth = 0

    log.info("=== Live Stream Pipeline running. Press 'q' to quit, 's' to save. ===")

    try:
        while True:
            frame, ts = reader.get_frame()
            if frame is None:
                time.sleep(0.002)
                continue

            # Undistort if calibrated
            if undist_map1 is not None:
                frame = cv2.remap(
                    frame, undist_map1, undist_map2, cv2.INTER_LINEAR,
                )

            n_frames += 1

            # ---- Visual Odometry --------------------------------------------
            t0 = time.perf_counter()
            result: TrackingResult = vo.process(frame)
            t_vo = time.perf_counter() - t0
            t_vo_total += t_vo

            # ---- Depth + mapping on keyframes --------------------------------
            if result.is_keyframe and not result.is_lost:
                t0 = time.perf_counter()
                depth = depth_est.predict(frame)
                t_depth = time.perf_counter() - t0
                t_depth_total += t_depth
                n_depth += 1

                # Scale update via triangulation
                tri = vo.triangulate_against_ref(frame)
                if tri is not None:
                    pts_3d, pts_2d = tri
                    vo.update_scale(depth, pts_3d, pts_2d)

                mapper.add_keyframe(frame, depth, result.pose_wc)

            # ---- Visualisation -----------------------------------------------
            if vis is not None:
                ok = vis.update(
                    frame,
                    num_inliers=result.num_inliers,
                    num_features=result.num_features,
                    is_lost=result.is_lost,
                    keyframe_count=vo.get_keyframe_count(),
                    scale=vo.current_scale,
                    cloud=mapper.get_cloud(),
                    trajectory=vo.get_trajectory(),
                )
                if not ok:
                    break

                key = vis.last_key()
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    log.info("Manual save triggered")
                    save_outputs(session_dir, mapper, vo.get_trajectory())

    except KeyboardInterrupt:
        log.info("Interrupted")

    # ---- Final save ---------------------------------------------------------
    save_outputs(session_dir, mapper, vo.get_trajectory())

    # ---- Timing report ------------------------------------------------------
    if n_frames > 0:
        log.info("--- Timing report ---")
        log.info("Frames processed : %d", n_frames)
        log.info("VO mean           : %.1f ms", t_vo_total / n_frames * 1000)
        if n_depth > 0:
            log.info("Depth mean        : %.1f ms", t_depth_total / n_depth * 1000)
        log.info("Keyframes         : %d", vo.get_keyframe_count())
        log.info("Cloud points      : %d", mapper.num_points)

    # ---- Cleanup ------------------------------------------------------------
    imu.stop()
    reader.release()
    if vis is not None:
        vis.destroy()


if __name__ == "__main__":
    main()
