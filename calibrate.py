#!/usr/bin/env python3
"""
calibrate.py — Camera calibration from a checkerboard pattern.

Captures frames from any OpenCV-compatible source (webcam, phone stream,
video file), detects a chessboard, collects corner sets, and computes the
intrinsic matrix K and distortion coefficients.  Results are saved to a
``.npz`` file that ``slam_mapper.py --calib`` can load.

Usage
-----
  # Live from webcam — press SPACE to capture, 'c' to calibrate, 'q' to quit
  python calibrate.py --src 0 --rows 9 --cols 6 --square-size 0.025

  # From phone
  python calibrate.py --src "http://192.168.1.42:8080/video" --rows 9 --cols 6

  # From a folder of images
  python calibrate.py --images "cal_imgs/*.jpg" --rows 9 --cols 6

The checkerboard size is *inner corners* (e.g. a 10×7-square board has
9×6 inner corners).
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("calibrate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Checkerboard camera calibration")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--src", help="Live source (webcam index or URL)")
    g.add_argument("--images", nargs="+", help="Glob/list of image files")
    p.add_argument("--rows", type=int, required=True, help="Inner corner rows")
    p.add_argument("--cols", type=int, required=True, help="Inner corner cols")
    p.add_argument(
        "--square-size", type=float, default=0.025,
        help="Physical square side in metres (default 0.025)",
    )
    p.add_argument(
        "--out", default="calib.npz",
        help="Output path (default: calib.npz)",
    )
    p.add_argument(
        "--min-caps", type=int, default=15,
        help="Minimum captures before calibration (default 15)",
    )
    return p.parse_args()


def calibrate_from_points(
    all_obj_pts: list[np.ndarray],
    all_img_pts: list[np.ndarray],
    img_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run OpenCV calibration.  Returns (K, dist, rms_error)."""
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        all_obj_pts, all_img_pts, img_size, None, None,
    )
    return K, dist.ravel(), rms


def main() -> None:
    args = parse_args()
    pattern = (args.cols, args.rows)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001,
    )

    # Object points for one board position
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    all_obj: list[np.ndarray] = []
    all_img: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None

    # ---- From image files ---------------------------------------------------
    if args.images:
        paths: list[str] = []
        for pat in args.images:
            paths.extend(glob.glob(pat))
        if not paths:
            log.error("No images found")
            sys.exit(1)

        for p in sorted(paths):
            img = cv2.imread(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(gray, pattern, None)
            if found:
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria,
                )
                all_obj.append(objp)
                all_img.append(corners2)
                log.info("  found  %s", p)
            else:
                log.warning("  MISS   %s — pattern not found", p)

    # ---- From live source ---------------------------------------------------
    else:
        try:
            src: str | int = int(args.src)  # type: ignore[assignment]
        except ValueError:
            src = args.src
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error("Cannot open %s", args.src)
            sys.exit(1)

        log.info("Press SPACE to capture, 'c' to calibrate, 'q' to quit")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(gray, pattern, None)
            vis = frame.copy()
            if found:
                cv2.drawChessboardCorners(vis, pattern, corners, found)

            cv2.putText(
                vis,
                f"Captures: {len(all_obj)} / {args.min_caps}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.imshow("Calibration", vis)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and found:
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria,
                )
                all_obj.append(objp)
                all_img.append(corners2)
                log.info("Captured #%d", len(all_obj))
            elif key == ord("c"):
                if len(all_obj) >= args.min_caps:
                    break
                else:
                    log.warning(
                        "Need at least %d captures (have %d)",
                        args.min_caps, len(all_obj),
                    )

        cap.release()
        cv2.destroyAllWindows()

    # ---- Calibrate ----------------------------------------------------------
    if len(all_obj) < 4:
        log.error("Too few captures (%d). Need at least 4.", len(all_obj))
        sys.exit(1)

    if img_size is None:
        log.error("No valid image size discovered during capture.")
        sys.exit(1)

    log.info("Calibrating with %d images ...", len(all_obj))
    K, dist, rms = calibrate_from_points(all_obj, all_img, img_size)

    log.info("RMS reprojection error: %.4f px", rms)
    log.info("K =\n%s", K)
    log.info("dist = %s", dist)

    np.savez(args.out, K=K, dist=dist, rms=rms)
    log.info("Saved -> %s", args.out)


if __name__ == "__main__":
    main()

