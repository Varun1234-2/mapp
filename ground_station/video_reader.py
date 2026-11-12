"""
Threaded video reader — always returns the *newest* frame.

For live streams (phone / drone) frames are grabbed continuously in a
background thread; ``get_frame()`` returns the latest one, discarding stale
frames so the pipeline never falls behind.

For video files ``get_frame()`` reads sequentially so every frame is
processed exactly once (useful for offline debugging / replay).

The ``--record`` option wraps any source and writes an MP4 sidecar.
"""

from __future__ import annotations

import time
import threading
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Unified video reader for streams (RTSP / HTTP / UDP) and files.

    Parameters
    ----------
    src : str
        OpenCV-compatible source string.  Examples:
        - ``"http://192.168.1.42:8080/video"``  (IP Webcam MJPEG)
        - ``"rtsp://192.168.1.42:8554/stream"`` (RTSP)
        - ``"udp://@:5600"``                    (GStreamer UDP)
        - ``"0"``                               (local webcam index)
        - ``"recording.mp4"``                   (video file)
    record_path : str | None
        If given, every grabbed frame is written to this MP4 path.
    """

    def __init__(self, src: str, record_path: str | None = None):
        # Try to parse as an integer index (webcam)
        try:
            self._src: str | int = int(src)
        except ValueError:
            self._src = src

        self._cap = cv2.VideoCapture(self._src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self._fps <= 0.0 or np.isnan(self._fps) or self._fps > 120.0:
            self._fps = 30.0

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Detect file vs. stream
        self._is_file = isinstance(self._src, str) and not self._src.startswith(
            ("http", "rtsp", "udp", "tcp", "rtmp")
        )
        try:
            # webcam index is also "live"
            int(src)
            self._is_file = False
        except ValueError:
            pass

        self._finished = False

        # Latest frame (live mode)
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._lock = threading.Lock()

        # If width or height is 0 (common with RTSP/HTTP MJPEG streams), read first frame
        if self._width <= 0 or self._height <= 0:
            ok, initial_frame = self._cap.read()
            if ok and initial_frame is not None:
                self._height, self._width = initial_frame.shape[:2]
                self._latest_frame = initial_frame.copy()
                self._latest_ts = time.monotonic()
            else:
                self._width, self._height = 640, 480  # fallback sensible default

        logger.info(
            "VideoReader opened: %s  (%dx%d @ %.1f fps)",
            src, self._width, self._height, self._fps,
        )

        # Optional recording
        self._writer: cv2.VideoWriter | None = None
        if record_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                record_path, fourcc, self._fps,
                (self._width, self._height),
            )
            logger.info("Recording to %s", record_path)
            # If we grabbed an initial frame for resolution, write it
            if self._latest_frame is not None:
                self._writer.write(self._latest_frame)

        # Start grabber thread for live sources
        self._running = True
        if not self._is_file:
            self._thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._thread.start()
        else:
            self._thread = None

    # ----- public API -------------------------------------------------------

    @property
    def is_file(self) -> bool:
        return self._is_file

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    def get_frame(self) -> tuple[np.ndarray | None, float]:
        """Return ``(frame_bgr, timestamp)`` or ``(None, 0.0)``.

        * Live: latest frame (may repeat if called faster than capture).
        * File: next sequential frame; ``(None, 0)`` when the file ends
          and ``self.finished`` becomes True.
        """
        if self._is_file:
            return self._read_next_file_frame()

        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            frame = self._latest_frame.copy()
            ts = self._latest_ts
        return frame, ts

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._writer is not None:
            self._writer.release()
        self._cap.release()
        logger.info("VideoReader released")

    # ----- internals --------------------------------------------------------

    def _grab_loop(self) -> None:
        """Continuously grab frames and keep only the latest."""
        while self._running:
            try:
                ok = self._cap.grab()
                if not ok:
                    time.sleep(0.005)
                    continue
                ok, frame = self._cap.retrieve()
                if not ok or frame is None:
                    continue
                ts = time.monotonic()
                if self._writer is not None:
                    self._writer.write(frame)
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = ts
            except Exception as exc:
                if not self._running:
                    break
                logger.debug("Error in grab loop: %s", exc)
                time.sleep(0.01)

    def _read_next_file_frame(self) -> tuple[np.ndarray | None, float]:
        if self._finished:
            return None, 0.0
        ok, frame = self._cap.read()
        if not ok:
            self._finished = True
            return None, 0.0
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if self._writer is not None:
            self._writer.write(frame)
        return frame, ts

