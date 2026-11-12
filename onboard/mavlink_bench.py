#!/usr/bin/env python3
"""
mavlink_bench.py — Probe which MAVLink messages the flight controller sends.

Connects to the Radiolink CrossFlight (or any Pixhawk-derived FC) over
a serial UART, listens for 10 seconds (configurable), and reports every
distinct message type received along with its rate.  This answers the
question "does this FC actually send ATTITUDE / RAW_IMU / SCALED_PRESSURE
/ GPS_RAW_INT?" before you rely on them in the SLAM pipeline.

Usage
-----
  python3 mavlink_bench.py --serial /dev/ttyAMA0 --baud 57600
  python3 mavlink_bench.py --serial /dev/ttyAMA0 --baud 57600 --duration 20

Expected output for a standard Pixhawk-derived FC with defaults:
  HEARTBEAT          :   1.0 Hz   (10 msgs)
  SYS_STATUS         :   1.0 Hz   (10 msgs)
  ATTITUDE           :  10.0 Hz  (100 msgs)   <-- needed
  RAW_IMU            :  50.0 Hz  (500 msgs)   <-- needed (or SCALED_IMU)
  SCALED_PRESSURE    :   1.0 Hz   (10 msgs)   <-- needed
  GPS_RAW_INT        :   1.0 Hz   (10 msgs)   <-- needed (outdoor only)

If a needed message is missing, check:
  - MAVLink stream rates:  use MAVProxy or QGroundControl to set
    SR*_RAW_SENS, SR*_POSITION, SR*_EXTRA1 to non-zero Hz.
  - UART baud rate / wiring.
  - MAVLink version (1 vs 2) — CrossFlight should default to v2.

Prerequisites
-------------
  pip3 install pymavlink
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
)
log = logging.getLogger("mavlink_bench")

# Messages the SLAM pipeline needs
NEEDED = {"ATTITUDE", "RAW_IMU", "SCALED_IMU", "SCALED_IMU2",
          "SCALED_PRESSURE", "GPS_RAW_INT"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probe MAVLink message types from a flight controller",
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
        "--duration", type=int, default=10,
        help="Listen duration in seconds (default 10)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from pymavlink import mavutil
    except ImportError:
        log.error("pymavlink not installed.  Run:  pip3 install pymavlink")
        sys.exit(1)

    log.info("Opening %s @ %d baud ...", args.serial, args.baud)
    mav = mavutil.mavlink_connection(args.serial, baud=args.baud)

    log.info("Waiting for heartbeat ...")
    hb = mav.wait_heartbeat(timeout=15)
    if hb is None:
        log.error(
            "No heartbeat received in 15 s.  Check wiring, baud rate, "
            "and that the FC is powered."
        )
        sys.exit(1)

    log.info(
        "Heartbeat OK — system %d  component %d  type %d",
        mav.target_system, mav.target_component, hb.type,
    )

    # ---- Collect messages for --duration seconds ---------------------------
    counts: dict[str, int] = defaultdict(int)
    log.info("Listening for %d seconds ...", args.duration)

    t_start = time.monotonic()
    deadline = t_start + args.duration

    while time.monotonic() < deadline:
        msg = mav.recv_match(blocking=True, timeout=0.5)
        if msg is None:
            continue
        counts[msg.get_type()] += 1

    elapsed = time.monotonic() - t_start

    # ---- Report ------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"MAVLink message report  ({elapsed:.1f} s)")
    print("=" * 60)
    print(f"{'Message Type':<30} {'Count':>8}  {'Rate (Hz)':>10}  {'Status'}")
    print("-" * 60)

    found_any_imu = False
    for mtype in sorted(counts, key=lambda k: -counts[k]):
        cnt = counts[mtype]
        rate = cnt / max(elapsed, 0.1)
        if mtype in NEEDED:
            status = "<-- NEEDED"
            if "IMU" in mtype:
                found_any_imu = True
        else:
            status = ""
        print(f"  {mtype:<28} {cnt:>8}  {rate:>9.1f}  {status}")

    print("-" * 60)

    # Check for missing needed messages
    missing = NEEDED - set(counts.keys())
    # RAW_IMU and SCALED_IMU are alternatives; only flag if BOTH are missing
    imu_types = {"RAW_IMU", "SCALED_IMU", "SCALED_IMU2"}
    if not (imu_types & set(counts.keys())):
        imu_missing = True
    else:
        imu_missing = False
        missing -= imu_types  # don't double-report

    if missing or imu_missing:
        print("\nWARNING — missing messages:")
        if imu_missing:
            print("  - No IMU message (RAW_IMU / SCALED_IMU / SCALED_IMU2)")
            print("    → Set SR*_RAW_SENS > 0 in the FC parameters")
        for m in sorted(missing):
            if m in imu_types:
                continue
            print(f"  - {m}")
            if m == "ATTITUDE":
                print("    → Set SR*_EXTRA1 > 0")
            elif m == "SCALED_PRESSURE":
                print("    → Set SR*_RAW_SENS > 0")
            elif m == "GPS_RAW_INT":
                print("    → Only available outdoors with GPS fix. Set SR*_POSITION > 0")
    else:
        print("\nAll needed message types are present.")

    print("=" * 60)


if __name__ == "__main__":
    main()

