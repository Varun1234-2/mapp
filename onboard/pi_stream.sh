#!/usr/bin/env bash
# ============================================================================
# pi_stream.sh — Raspberry Pi camera H.264 streaming (drone onboard)
#
# Captures from the CSI camera, hardware-encodes to H.264 with low-latency
# settings, and streams over UDP to the ground station.
#
# Usage:
#   ./pi_stream.sh <GROUND_STATION_IP> [PORT] [RESOLUTION] [BITRATE]
#
# Examples:
#   ./pi_stream.sh 192.168.1.100              # 640x480 @ 30fps, 4 Mbps, port 5600
#   ./pi_stream.sh 192.168.1.100 5600 1280x720 8000000
#
# The script auto-detects whether to use:
#   libcamera-vid  (Bullseye / Bookworm — Pi OS >= 2021-11-08)
#   raspivid       (Buster and earlier — legacy camera stack)
#
# Requirements:
#   - Pi Camera Module v1 or v2 connected via CSI ribbon cable
#   - Camera enabled:  sudo raspi-config -> Interface -> Camera
#   - For libcamera:   libcamera-apps installed (default on Bullseye+)
#   - For raspivid:    legacy camera stack enabled in raspi-config
# ============================================================================

set -euo pipefail

# ----- Arguments ------------------------------------------------------------
GS_IP="${1:?Usage: $0 <GROUND_STATION_IP> [PORT] [RESOLUTION] [BITRATE]}"
PORT="${2:-5600}"
RES="${3:-640x480}"
BITRATE="${4:-4000000}"    # bits/sec  (4 Mbps default; use 8000000 for 720p)
FPS=30

WIDTH="${RES%x*}"
HEIGHT="${RES#*x}"

echo "=== Pi Camera Streamer ==="
echo "Target : ${GS_IP}:${PORT}"
echo "Res    : ${WIDTH}x${HEIGHT} @ ${FPS} fps"
echo "Bitrate: ${BITRATE} bps"
echo ""

# ----- Detect camera stack ---------------------------------------------------
if command -v libcamera-vid &>/dev/null; then
    BACKEND="libcamera"
elif command -v raspivid &>/dev/null; then
    BACKEND="raspivid"
else
    echo "ERROR: Neither libcamera-vid nor raspivid found."
    echo "       Enable the camera in raspi-config and install libcamera-apps."
    exit 1
fi
echo "Backend: ${BACKEND}"

# ----- Low-latency H.264 settings -------------------------------------------
# - intra (keyframe) period = 30 frames (1 s at 30 fps)  -- short GOP
# - profile = baseline  -- no B-frames
# - inline headers      -- SPS/PPS with every IDR (needed for UDP)

# ----- Stream ----------------------------------------------------------------
if [ "$BACKEND" = "libcamera" ]; then
    # libcamera-vid (Bullseye / Bookworm)
    # Pipe raw H.264 NALUs into GStreamer for UDP/RTP packetisation.
    echo "Starting libcamera-vid -> GStreamer RTP/UDP pipeline..."

    libcamera-vid \
        --width  "$WIDTH" \
        --height "$HEIGHT" \
        --framerate "$FPS" \
        --bitrate "$BITRATE" \
        --profile baseline \
        --level  4 \
        --intra  "$FPS" \
        --inline \
        --nopreview \
        --timeout 0 \
        --output - \
    | gst-launch-1.0 -v \
        fdsrc fd=0 \
        ! h264parse \
        ! rtph264pay config-interval=1 pt=96 \
        ! udpsink host="$GS_IP" port="$PORT" sync=false

else
    # raspivid (Buster / legacy)
    echo "Starting raspivid -> GStreamer RTP/UDP pipeline..."

    raspivid \
        -w "$WIDTH" \
        -h "$HEIGHT" \
        -fps "$FPS" \
        -b "$BITRATE" \
        -pf baseline \
        -ih \
        -g "$FPS" \
        -t 0 \
        -n \
        -o - \
    | gst-launch-1.0 -v \
        fdsrc fd=0 \
        ! h264parse \
        ! rtph264pay config-interval=1 pt=96 \
        ! udpsink host="$GS_IP" port="$PORT" sync=false
fi

# ============================================================================
# GStreamer receive pipeline on the GROUND STATION (for reference):
#
#   gst-launch-1.0 \
#       udpsrc port=5600 \
#       ! application/x-rtp,encoding-name=H264,payload=96 \
#       ! rtph264depay \
#       ! h264parse \
#       ! avdec_h264 \
#       ! videoconvert \
#       ! autovideosink sync=false
#
# Or to use with slam_mapper.py (if OpenCV has GStreamer backend):
#
#   python slam_mapper.py --src \
#     "udpsrc port=5600 ! application/x-rtp,encoding-name=H264,payload=96 \
#      ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert \
#      ! appsink sync=false"
#
# ============================================================================
# Wiring notes (Pi <-> Flight Controller UART):
#
#   Pi GPIO 14 (TXD) --> FC TELEM2 RX
#   Pi GPIO 15 (RXD) <-- FC TELEM2 TX
#   Pi GND           --- FC GND
#
#   The Pi UART is 3.3 V -- the CrossFlight TELEM ports are also 3.3 V,
#   so NO level shifter is needed.  Do NOT connect 5 V.
#
# ============================================================================
# Pi resource checklist (1 GB RAM target: <= 400 MB, <= 60% CPU):
#
#   - Boot to CLI, no desktop:
#       sudo raspi-config -> System -> Boot -> Console
#   - Disable Bluetooth (frees UART0 and some RAM):
#       Add to /boot/config.txt:  dtoverlay=disable-bt
#       sudo systemctl disable hciuart
#   - GPU memory split: at least 128 MB for the camera encoder:
#       Add to /boot/config.txt:  gpu_mem=128
#   - Disable unused services:
#       sudo systemctl disable avahi-daemon triggerhappy
#   - Monitor:
#       htop                      -- CPU / RAM
#       vcgencmd measure_temp     -- throttling check
#       vcgencmd get_throttled    -- voltage / thermal warnings
# ============================================================================

