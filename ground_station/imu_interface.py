"""
IMU / sensor data interface.

Provides an abstract base class and concrete stubs so the pipeline can
accept inertial data from any source (MAVLink, phone, replay file) via
a uniform API.  For this first version the *NullIMU* is used — it returns
nothing but satisfies the interface.  When the drone hardware arrives,
*MAVLinkIMU* (or a UDP receiver) will be wired in.
"""

from __future__ import annotations

import time
import threading
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IMUSample:
    """Single time-stamped inertial / barometric / GPS reading."""
    timestamp: float                             # UNIX epoch (seconds)
    accel: np.ndarray                            # [ax, ay, az] m/s²
    gyro: np.ndarray                             # [gx, gy, gz] rad/s
    mag: np.ndarray | None = None                # [mx, my, mz] µT
    baro_alt: float | None = None                # barometric altitude (m)
    gps: tuple[float, float, float] | None = None  # (lat, lon, alt_msl)


# --------------------------------------------------------------------------- #
#  Abstract interface                                                          #
# --------------------------------------------------------------------------- #

class IMUInterface(ABC):
    """Common API for all inertial data sources."""

    @abstractmethod
    def start(self) -> None:
        """Begin listening / reading."""

    @abstractmethod
    def stop(self) -> None:
        """Clean shutdown."""

    @abstractmethod
    def get_samples(self, since: float = 0.0) -> list[IMUSample]:
        """Return samples with timestamp > *since*, oldest first."""

    @abstractmethod
    def get_latest(self) -> IMUSample | None:
        """Return the most recent sample, or None if nothing yet."""


# --------------------------------------------------------------------------- #
#  Null (no-op) implementation — used when there is no IMU source              #
# --------------------------------------------------------------------------- #

class NullIMU(IMUInterface):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_samples(self, since: float = 0.0) -> list[IMUSample]:
        return []

    def get_latest(self) -> IMUSample | None:
        return None


# --------------------------------------------------------------------------- #
#  MAVLink stub — ready to be filled in when the drone is available            #
# --------------------------------------------------------------------------- #

class MAVLinkIMU(IMUInterface):
    """Receive IMU / baro / GPS from a MAVLink UDP stream.

    Designed for the Radiolink CrossFlight FC forwarded by
    ``onboard/mavlink_forward.py``.

    Parameters
    ----------
    udp_addr : str
        ``host:port`` the forwarder sends to (e.g. ``0.0.0.0:14550``).
    """

    def __init__(self, udp_addr: str = "0.0.0.0:14550"):
        self._addr = udp_addr
        self._buf: list[IMUSample] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # -- public API ----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("MAVLinkIMU listener started on %s", self._addr)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("MAVLinkIMU listener stopped")

    def get_samples(self, since: float = 0.0) -> list[IMUSample]:
        with self._lock:
            out = [s for s in self._buf if s.timestamp > since]
        return out

    def get_latest(self) -> IMUSample | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    # -- background receiver (stub — uncomment pymavlink code when ready) ----

    def _run(self) -> None:
        """Read MAVLink messages in a loop.

        Un-comment and install pymavlink when deploying on the drone:
            pip install pymavlink
        """
        logger.warning(
            "MAVLinkIMU._run is a STUB — no pymavlink code executed.  "
            "Install pymavlink and see onboard/mavlink_forward.py for the "
            "real implementation."
        )
        # ------------------------------------------------------------------
        # from pymavlink import mavutil
        # conn = mavutil.mavlink_connection(f"udpin:{self._addr}")
        # while self._running:
        #     msg = conn.recv_match(blocking=True, timeout=0.1)
        #     if msg is None:
        #         continue
        #     mtype = msg.get_type()
        #     now = time.time()
        #     sample = IMUSample(
        #         timestamp=now,
        #         accel=np.zeros(3), gyro=np.zeros(3),
        #     )
        #     if mtype == "RAW_IMU":
        #         sample.accel = np.array([msg.xacc, msg.yacc, msg.zacc],
        #                                 dtype=np.float64) * 9.80665 / 1000.0
        #         sample.gyro = np.array([msg.xgyro, msg.ygyro, msg.zgyro],
        #                                dtype=np.float64) / 1000.0
        #     elif mtype == "SCALED_IMU2":
        #         sample.accel = np.array([msg.xacc, msg.yacc, msg.zacc],
        #                                 dtype=np.float64) / 1000.0 * 9.80665
        #         sample.gyro = np.array([msg.xgyro, msg.ygyro, msg.zgyro],
        #                                dtype=np.float64) / 1000.0
        #     elif mtype == "ATTITUDE":
        #         pass  # roll/pitch/yaw available as msg.roll, etc.
        #     elif mtype == "SCALED_PRESSURE":
        #         sample.baro_alt = msg.press_abs  # hPa → needs conversion
        #     elif mtype == "GPS_RAW_INT":
        #         if msg.fix_type >= 2:
        #             sample.gps = (msg.lat / 1e7, msg.lon / 1e7,
        #                           msg.alt / 1e3)
        #     else:
        #         continue
        #     with self._lock:
        #         self._buf.append(sample)
        #         if len(self._buf) > 5000:
        #             self._buf = self._buf[-2500:]
        # ------------------------------------------------------------------
        while self._running:
            time.sleep(0.5)

