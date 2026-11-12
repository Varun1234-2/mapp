#!/usr/bin/env python3
"""
mavlink_forward.py — Forward MAVLink from serial UART to UDP.

Runs on the Raspberry Pi.  Reads MAVLink messages from the flight
controller's UART (Radiolink CrossFlight TELEM2 port) and forwards
them over UDP to the ground station, where they can be consumed by
``MAVLinkIMU`` in the SLAM pipeline.

Usage
-----
  python3 mavlink_forward.py --serial /dev/ttyAMA0 --baud 57600 \
          --udp-target 192.168.1.100:14550

  # With verbose message logging
  python3 mavlink_forward.py --serial /dev/ttyAMA0 --udp-target 192.168.1.100:14550 -v

Prerequisites
-------------
  pip3 install pymavlink

  # Free UART0 from the Bluetooth modem (Pi 3B+):
  # Add to /boot/config.txt:
  #   dtoverlay=disable-bt
  # Then:
  #   sudo systemctl disable hciuart
  #   sudo reboot
  #
  # Also disable the serial console:
  #   sudo raspi-config -> Interface -> Serial Port
  #     "Would you like a login shell over serial?" -> No
  #     "Would you like the serial port hardware to be enabled?" -> Yes

Wiring
------
  Pi GPIO 14 (TXD, pin 8)  -->  FC TELEM2 RX
  Pi GPIO 15 (RXD, pin 10) <--  FC TELEM2 TX
  Pi GND     (pin 6 / 14)  ---  FC TELEM2 GND

  Both sides are 3.3 V — no level shifter needed.
  Do NOT connect the FC's 5 V output to the Pi on this cable.

Timestamp injection
-------------------
Each forwarded message is tagged with the Pi's monotonic clock.
Use Chrony/NTP between the Pi and the ground station to keep clocks
synchronised (see README.md → Time Synchronisation).

Message filtering
-----------------
By default only IMU / baro / GPS / attitude messages are forwarded
to keep bandwidth low.  Use ``--all`` to forward everything.
"""

from __future__ import annotations

import argparse
import logging
import struct
import socket
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mavlink_fwd")

# Messages we care about for SLAM
WANTED_TYPES = {
    "ATTITUDE",
    "RAW_IMU",
    "SCALED_IMU",
    "SCALED_IMU2",
    "SCALED_PRESSURE",
    "GPS_RAW_INT",
    "GLOBAL_POSITION_INT",
    "HEARTBEAT",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forward MAVLink from serial to UDP",
    )
    p.add_argument(
        "--serial", required=True,
        help="Serial device (e.g. /dev/ttyAMA0)",
    )
    p.add_argument(
        "--baud", type=int, default=57600,
        help="Baud rate (default 57600)",
    )
    p.add_argument(
        "--udp-target", required=True,
        help="Ground station host:port (e.g. 192.168.1.100:14550)",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Forward ALL message types (not just IMU/GPS/baro)",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Log every forwarded message",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)

    host, port_str = args.udp_target.rsplit(":", 1)
    port = int(port_str)

    # ---- pymavlink import (fail fast if not installed) ----------------------
    try:
        from pymavlink import mavutil
    except ImportError:
        log.error(
            "pymavlink is not installed.  Run:  pip3 install pymavlink"
        )
        sys.exit(1)

    # ---- Open serial MAVLink connection ------------------------------------
    log.info("Opening serial %s @ %d baud", args.serial, args.baud)
    mav = mavutil.mavlink_connection(
        args.serial, baud=args.baud, dialect="common",
    )

    # Wait for a heartbeat to confirm connection
    log.info("Waiting for heartbeat...")
    mav.wait_heartbeat(timeout=10)
    log.info(
        "Heartbeat received: system %d  component %d",
        mav.target_system, mav.target_component,
    )

    # ---- UDP output socket --------------------------------------------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    log.info("Forwarding to UDP %s:%d", host, port)

    # ---- Forward loop -------------------------------------------------------
    n_forwarded = 0
    n_dropped = 0
    t_start = time.monotonic()

    try:
        while True:
            msg = mav.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue

            mtype = msg.get_type()

            # Filter
            if not args.all and mtype not in WANTED_TYPES:
                n_dropped += 1
                continue

            # Serialise and send
            buf = msg.get_msgbuf()

            # Prepend an 8-byte monotonic timestamp (double, little-endian)
            ts = struct.pack("<d", time.monotonic())
            sock.sendto(ts + bytes(buf), (host, port))

            n_forwarded += 1
            if args.verbose:
                log.debug("FWD %-25s  (%d bytes)", mtype, len(buf))

            # Periodic stats
            if n_forwarded % 500 == 0:
                elapsed = time.monotonic() - t_start
                rate = n_forwarded / max(elapsed, 1e-3)
                log.info(
                    "Forwarded %d msgs (%.0f/s), dropped %d",
                    n_forwarded, rate, n_dropped,
                )

    except KeyboardInterrupt:
        log.info("Stopped. Forwarded %d messages total.", n_forwarded)
    finally:
        sock.close()


if __name__ == "__main__":
    main()

