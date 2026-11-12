# Drone Spatial Mapping System

Real-time monocular SLAM with metric depth estimation for drone mapping.  
The **ground station** runs ORB visual odometry + Depth Anything V2 metric depth + coloured point-cloud fusion.  
The **onboard Pi** captures, encodes and streams; all heavy processing happens on the laptop GPU.

---

## Architecture

```
┌─────────────────────────────┐          Wi-Fi (5 GHz)          ┌──────────────────────────────────────┐
│     DRONE  (Raspberry Pi)   │  ─────────────────────────────► │       GROUND STATION  (Laptop)       │
│                             │   H.264 / UDP  :5600            │                                      │
│  ┌──────────┐  ┌──────────┐ │   MAVLink / UDP :14550          │  ┌────────────────────────────────┐  │
│  │ Pi Camera│→ │pi_stream │ │ ───────────────────────────────►│  │        slam_mapper.py          │  │
│  │  (CSI)   │  │  .sh     │ │                                 │  │                                │  │
│  └──────────┘  └──────────┘ │                                 │  │  VideoReader  (threaded)       │  │
│                             │                                 │  │       │                        │  │
│  ┌──────────┐  ┌──────────┐ │                                 │  │       ▼                        │  │
│  │ Flight   │→ │ mavlink  │ │                                 │  │  VisualOdometry  (ORB+E-mat)   │  │
│  │Controller│  │_forward  │ │                                 │  │       │                        │  │
│  │(MAVLink) │  │  .py     │ │                                 │  │       ├──► DepthEstimator      │  │
│  └──────────┘  └──────────┘ │                                 │  │       │    (DA-V2 Metric GPU)  │  │
│                             │                                 │  │       │         │              │  │
│  Chrony NTP client          │                                 │  │       ▼         ▼              │  │
└─────────────────────────────┘                                 │  │  DenseMapper (Open3D fusion)   │  │
                                                                │  │       │                        │  │
  ┌───────────────────────┐                                     │  │       ▼                        │  │
  │  PHONE  (dev proxy)   │ ── HTTP/RTSP ──────────────────────►│  │  MapVisualizer                 │  │
  │  IP Webcam / DroidCam │                                     │  │  (Open3D 3D + OpenCV HUD)      │  │
  └───────────────────────┘                                     │  └────────────────────────────────┘  │
                                                                │                                      │
                                                                │  Outputs: map.ply, trajectory.txt,   │
                                                                │           occupancy.png              │
                                                                └──────────────────────────────────────┘
```

### Data flow

1. **Capture** — Phone (now) or Pi camera (later) produces 640×480 / 720p frames at 30 fps.
2. **Stream** — Frames reach the ground station as an MJPEG HTTP stream, RTSP, or H.264-over-UDP.
3. **Threaded reader** — `VideoReader` grabs frames in a background thread and always returns the *newest* one (no buffer lag). For video files it reads sequentially.
4. **Visual odometry** — ORB features detected per frame, matched against a reference keyframe with Lowe's ratio test, essential matrix computed via RANSAC, pose (R, t) recovered. Translation is unit-length; metric scale comes from the depth network.
5. **Keyframe selection** — A new keyframe is declared when median pixel parallax of inlier matches exceeds 20 px and at least 5 frames have elapsed.
6. **Depth estimation** — On each keyframe, Depth Anything V2 Metric (Small) runs on the GPU and produces a metric depth map in metres.
7. **Scale fusion** — Matched features are triangulated between keyframes; the ratio `median(network_depth / triangulated_depth)` gives the metric scale, smoothed via EMA.
8. **Dense mapping** — The depth map is back-projected to 3D, flying pixels are filtered, points are coloured from the RGB frame, transformed to world frame, and voxel-downsampled into a global Open3D point cloud.
9. **Visualisation** — Open3D shows the growing 3D map and green trajectory; OpenCV shows the camera feed with FPS, tracking status, keyframe count, and scale.
10. **Output** — On quit (or `s` key): `map.ply`, `trajectory.txt` (KITTI format), `occupancy.png`.

### Coordinate convention

| Frame   | X     | Y    | Z       |
|---------|-------|------|---------|
| Camera  | right | down | forward |
| World   | right | down | forward |

World frame = first camera frame. Poses `T_wc` (4×4) transform points FROM camera TO world: `p_world = T_wc @ [p_cam; 1]`.

---

## Installation

### Ground station (Windows / Linux with NVIDIA GPU)

```bash
# 1. Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

# 2. Install PyTorch with CUDA (adjust cu121 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Onboard Pi (later, when hardware arrives)

```bash
sudo apt update && sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-good
pip3 install pymavlink
```

---

## Quick start

### 1. Stream from Android phone (IP Webcam)

1. Install **IP Webcam** from the Play Store.
2. Open the app → scroll down → **Start server**.
3. Note the URL shown (e.g. `http://192.168.1.42:8080`).
4. Run:

```bash
python slam_mapper.py --src "http://192.168.1.42:8080/video" --env indoor
```

### 2. Stream from iPhone

Use **DroidCam** (iOS version) or any RTSP-server app:

```bash
# DroidCam
python slam_mapper.py --src "http://192.168.1.42:4747/video" --env indoor

# Generic RTSP app
python slam_mapper.py --src "rtsp://192.168.1.42:8554/live" --env indoor
```

### 3. Local webcam

```bash
python slam_mapper.py --src 0 --env indoor --hfov 60
```

### 4. Recorded video (offline replay)

```bash
# Record a session first
python slam_mapper.py --src 0 --env indoor --record session.mp4

# Replay it
python slam_mapper.py --src session.mp4 --env indoor
```

### 5. Pi drone stream (future)

```bash
# On the Pi:
./onboard/pi_stream.sh 192.168.1.100 5600

# On the ground station:
python slam_mapper.py --src "udp://@:5600" --env outdoor --calib calib.npz
```

---

## Camera calibration

Print a checkerboard pattern (e.g. 10×7 squares).  Measure the square side in metres.

```bash
# Live calibration — hold the board in different poses, press SPACE to capture
python calibrate.py --src 0 --rows 9 --cols 6 --square-size 0.025

# Then use it
python slam_mapper.py --src 0 --calib calib.npz --env indoor
```

If you don't have a checkerboard, `--hfov 66` (typical phone) or `--hfov 73` (Pi Camera v2) gives a reasonable estimate.

---

## CLI reference

### slam_mapper.py

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | *required* | Video source (URL, file, or webcam index) |
| `--env` | `indoor` | `indoor` or `outdoor` — selects depth model and limits |
| `--calib` | — | Path to `calib.npz` from `calibrate.py` |
| `--hfov` | `66.0` | Horizontal FOV in degrees (fallback if no calib) |
| `--record` | — | Record incoming stream to this MP4 path |
| `--out-dir` | `.` | Directory for output files |
| `--voxel-size` | auto | Voxel size in metres (0.02 indoor, 0.05 outdoor) |
| `--depth-subsample` | `2` | Back-project every N-th pixel |
| `--device` | `cuda` | PyTorch device (`cuda` or `cpu`) |
| `--no-vis` | off | Headless mode (no GUI windows) |
| `-v` | off | Verbose / DEBUG logging |

---

## Outputs

| File | Format | Description |
|------|--------|-------------|
| `map.ply` | PLY (binary) | Coloured 3D point cloud — open in MeshLab, CloudCompare, or Open3D |
| `trajectory.txt` | KITTI | One 3×4 pose matrix per line (12 floats, space-separated) |
| `occupancy.png` | Grayscale PNG | Top-down occupancy grid: 255 = occupied, 128 = unknown |

---

## Time synchronisation (Pi ↔ Ground station)

Install Chrony on both machines:

```bash
# Pi (client)
sudo apt install chrony
# Add to /etc/chrony/chrony.conf:
#   server 192.168.1.100 iburst prefer
sudo systemctl restart chrony

# Laptop (server)
# Add to chrony.conf:
#   allow 192.168.0.0/16
#   local stratum 10
sudo systemctl restart chrony
```

Check sync: `chronyc tracking` — offset should be < 5 ms.

---

## Onboard Pi scripts

### pi_stream.sh

Auto-detects `libcamera-vid` (Bullseye+) or `raspivid` (Buster) and streams H.264 over UDP.

```bash
chmod +x onboard/pi_stream.sh
./onboard/pi_stream.sh 192.168.1.100          # 640x480, 4 Mbps
./onboard/pi_stream.sh 192.168.1.100 5600 1280x720 8000000   # 720p, 8 Mbps
```

### mavlink_forward.py

Forwards MAVLink from the FC serial port to UDP:

```bash
python3 onboard/mavlink_forward.py \
    --serial /dev/ttyAMA0 --baud 57600 \
    --udp-target 192.168.1.100:14550
```

### mavlink_bench.py

Discovers which MAVLink messages the FC actually sends:

```bash
python3 onboard/mavlink_bench.py --serial /dev/ttyAMA0 --duration 15
```

---

## Test plan

### Stage 1: Offline video replay

1. Record 30–60 s of walking through a room with your phone/webcam.
2. Run: `python slam_mapper.py --src recording.mp4 --env indoor`
3. **Success criteria:**
   - The trajectory in Open3D traces a plausible path (no wild jumps).
   - The point cloud looks like the room — walls, furniture visible.
   - Scale is plausible (room dimensions ≈ correct order of magnitude).
   - `map.ply` opens in MeshLab and looks coherent.
   - `trajectory.txt` has as many rows as processed frames.

### Stage 2: Live phone stream

1. Start IP Webcam on your phone.
2. Run: `python slam_mapper.py --src "http://<phone-ip>:8080/video" --env indoor`
3. Walk slowly through a room. Move smoothly — sudden rotations will cause tracking loss.
4. **Success criteria:**
   - FPS in the HUD stays ≥ 10.
   - Status shows "TRACKING" (not "LOST") most of the time.
   - The 3D map grows in real time.
   - Press `s` to save mid-session; verify `map.ply` is non-empty.

### Stage 3: Pi stream (when hardware arrives)

1. Set up Pi streaming and MAVLink forwarding.
2. Run `mavlink_bench.py` to verify message types.
3. Run `slam_mapper.py --src "udp://@:5600"` — verify same behaviour.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot open video source` | Check the URL. For IP Webcam, use `http://ip:8080/video` (not `/videofeed`). Make sure phone and laptop are on the same Wi-Fi. |
| Black window / no frames | Phone screen may have locked. Keep IP Webcam in foreground. |
| `No depth model could be loaded` | Check internet for first-time download. The model is ~100 MB. Or pre-download: `huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf` |
| `CUDA out of memory` | Reduce `--depth-subsample` to 4. Or use `--device cpu` (slower). |
| Tracking lost frequently | Move slower. Ensure good lighting. Avoid textureless walls. |
| Open3D window blank | Some systems need `--no-vis` and rely on the OpenCV window only. Open3D on WSL/remote may not work. |
| Import error for `open3d` | `pip install open3d`. On ARM Linux: build from source or use `pip install open3d-cpu`. |
| Scale looks wrong | Calibrate the camera (`calibrate.py`). A bad HFOV estimate causes scale drift. |
| UDP stream won't connect | Check firewall rules. On Windows: allow Python through the firewall. |

---

## Known limitations

1. **No loop closure** — Drift accumulates. Revisiting a place does not correct the map. → Next step: bag-of-words loop detection + pose-graph optimisation (g2o / GTSAM).

2. **No IMU fusion** — The interface is wired but the IMU is not used for pose estimation. → Next step: pre-integrate IMU between keyframes and add IMU factors to the graph.

3. **Rolling-shutter distortion** — The Pi Camera v1/v2 and phone cameras have rolling shutters. Fast motion causes skew not modelled by the VO. → Next step: rolling-shutter-aware essential matrix, or switch to a global-shutter camera.

4. **Monocular scale ambiguity** — Scale comes from the depth network, which has its own error (especially at range). True scale will improve with IMU + barometer fusion.

5. **No multi-session / map reuse** — Each run starts a fresh map. → Next step: save/load keyframe database for relocalisation.

6. **Feature-based only** — ORB features fail in textureless environments. → Next step: direct or semi-direct VO (DSO, SVO), or a learned frontend.

7. **No dynamic-object filtering** — Moving people / cars corrupt the map. → Next step: semantic segmentation mask to exclude dynamic classes.

## Recommended next steps (priority order)

1. **Wire IMU** — Fill in `MAVLinkIMU._run()`, fuse with VO for robust scale and handling of fast rotations.
2. **Loop closure** — DBoW2 or NetVLAD place recognition + pose-graph backend.
3. **Swap VO backend** — Replace the hand-written ORB pipeline with ORB-SLAM3 or DPVO for better accuracy.
4. **Mesh reconstruction** — Use Poisson / TSDF reconstruction from the depth-fused cloud for cleaner geometry.
5. **Path planning** — Convert the occupancy grid to a navigation grid for autonomous flight.

---

## File tree

```
mapping/
├── README.md               ← this file
├── requirements.txt         ← pip dependencies
├── calibrate.py             ← checkerboard calibration → calib.npz
├── slam_mapper.py           ← main ground-station entry point
├── ground_station/
│   ├── __init__.py
│   ├── utils.py             ← Pose class, camera model, back-projection
│   ├── video_reader.py      ← threaded capture, newest-frame, recording
│   ├── visual_odometry.py   ← ORB + essential matrix + keyframes + scale
│   ├── depth_estimator.py   ← Depth Anything V2 Metric (HuggingFace)
│   ├── dense_mapper.py      ← point-cloud fusion, occupancy grid
│   ├── visualizer.py        ← Open3D 3D window + OpenCV HUD
│   └── imu_interface.py     ← ABC + NullIMU + MAVLinkIMU stub
└── onboard/
    ├── pi_stream.sh         ← GStreamer H.264 streaming (raspivid + libcamera)
    ├── mavlink_forward.py   ← MAVLink serial → UDP forwarder
    └── mavlink_bench.py     ← probe which MAVLink messages the FC sends
```

---

## License

MIT — use freely, attribution appreciated.

