"""
Threaded live video reader — always returns the *newest* camera frame.

Frames are grabbed continuously in a background thread; ``get_frame()``
returns the latest one, discarding stale buffered frames so the SLAM
pipeline always tracks zero-latency real-time video.
"""

from __future__ import annotations

import time
import threading
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Zero-lag threaded camera stream reader (webcam, RTSP, HTTP MJPEG, UDP).

    Parameters
    ----------
    src : str
        Live camera source identifier. Examples:
        - ``"0"``                               (local USB/built-in webcam index)
        - ``"http://192.168.1.42:8080/video"``  (Phone camera IP Webcam)
        - ``"rtsp://192.168.1.42:8554/live"``   (RTSP live stream)
        - ``"udp://@:5600"``                    (Drone GStreamer UDP H.264 stream)
    """

    def __init__(self, src: str):
        # Parse numeric camera index if applicable
        try:
            self._src: str | int = int(src)
        except ValueError:
            self._src = src

        self._cap = cv2.VideoCapture(self._src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open live camera source: {src}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self._fps <= 0.0 or np.isnan(self._fps) or self._fps > 120.0:
            self._fps = 30.0

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Latest frame buffer
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._lock = threading.Lock()

        # If width or height is 0 (common with RTSP/HTTP streams before 1st decode)
        if self._width <= 0 or self._height <= 0:
            ok, initial_frame = self._cap.read()
            if ok and initial_frame is not None:
                self._height, self._width = initial_frame.shape[:2]
                self._latest_frame = initial_frame.copy()
                self._latest_ts = time.monotonic()
            else:
                self._width, self._height = 640, 480

        logger.info(
            "Live Camera Stream opened: %s  (%dx%d @ %.1f fps)",
            src, self._width, self._height, self._fps,
        )

        # Start continuous frame grabber thread
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    # ----- public API -------------------------------------------------------

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
        """Return the latest available camera frame and its timestamp."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            frame = self._latest_frame.copy()
            ts = self._latest_ts
        return frame, ts

    def release(self) -> None:
        """Stop background capture and release the camera stream."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()
        logger.info("Camera stream released")

    # ----- internals --------------------------------------------------------

    def _grab_loop(self) -> None:
        """Continuously grab frames in the background to prevent buffer latency."""
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
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = ts
            except Exception as exc:
                if not self._running:
                    break
                logger.debug("Error in stream grab loop: %s", exc)
                time.sleep(0.01)
