# Drone Real-Time Spatial Mapping System (Live Camera Stream)

Real-time monocular SLAM system with metric depth estimation designed for **live camera streams** (Phone camera, USB/built-in webcam, or Drone CSI camera over UDP/Wi-Fi).

All heavy SLAM processing runs on the ground station (GPU laptop / desktop).

---

## Hardware & Architecture

```
┌─────────────────────────────────┐           Wi-Fi (5 GHz)         ┌──────────────────────────────────────┐
│     DRONE  (Raspberry Pi)       │  ─────────────────────────────► │       GROUND STATION (GPU Laptop)    │
│                                 │   H.264 / UDP :5600             │                                      │
│  ┌───────────┐   ┌────────────┐ │   MAVLink / UDP :14550          │  ┌────────────────────────────────┐  │
│  │ Pi Camera │ → │ pi_stream  │ │ ───────────────────────────────►│  │        slam_mapper.py          │  │
│  │ (CSI/Raw) │   │   .sh      │ │                                 │  │                                │  │
│  └───────────┘   └────────────┘ │                                 │  │  VideoReader                   │  │
│  ┌───────────┐   ┌────────────┐ │                                 │  │  (Zero-lag threaded grabber)   │  │
│  │ Flight    │ → │ mavlink    │ │                                 │  │       │                        │  │
│  │Controller │   │ _forward.py│ │                                 │  │       ▼                        │  │
│  └───────────┘   └────────────┘ │                                 │  │  VisualOdometry (ORB + E-mat)  │  │
└─────────────────────────────────┘                                 │  │       │                        │  │
                                                                    │  │       ├──► DepthEstimator      │  │
  ┌───────────────────────────────┐                                 │  │       │    (DA-V2 Metric GPU)  │  │
  │     PHONE / LIVE CAMERA       │  ── HTTP / RTSP ───────────────►│  │       ▼         ▼              │  │
  │   IP Webcam / DroidCam / USB  │                                 │  │  DenseMapper (Open3D fusion)   │  │
  └───────────────────────────────┘                                 │  │       │                        │  │
                                                                    │  │       ▼                        │  │
                                                                    │  │  MapVisualizer                 │  │
                                                                    │  │  (Open3D 3D + OpenCV HUD)      │  │
                                                                    │  └────────────────────────────────┘  │
                                                                    │                                      │
                                                                    │  Outputs: map.ply, trajectory.txt,   │
                                                                    │           occupancy.png              │
                                                                    └──────────────────────────────────────┘
```

---

## Project Structure

```
mapping/
├── README.md               # Setup & usage guide
├── requirements.txt         # Python dependencies
├── slam_mapper.py           # Main ground station live SLAM pipeline
├── calibrate.py             # Live camera checkerboard calibration
├── ground_station/
│   ├── __init__.py
│   ├── video_reader.py      # Zero-lag threaded live camera stream reader
│   ├── visual_odometry.py   # Monocular ORB visual odometry + scale recovery
│   ├── depth_estimator.py   # Depth Anything V2 Metric model inference
│   ├── dense_mapper.py      # Colored 3D point cloud fusion & 2D occupancy grid
│   ├── visualizer.py        # Non-blocking Open3D 3D view + OpenCV HUD
│   ├── imu_interface.py     # Abstract interface for flight-controller telemetry
│   └── utils.py             # Poses, camera models, and back-projection
└── onboard/
    ├── pi_stream.sh         # Drone Pi camera hardware H.264 encode & stream
    ├── mavlink_forward.py   # Drone FC telemetry forwarder
    └── mavlink_bench.py     # Hardware telemetry probe script
```

---

## Live Stream Run Commands

### 1. From an Android Phone Camera (IP Webcam)
1. Install **IP Webcam** from Google Play.
2. Open the app, scroll to the bottom, and tap **Start server**.
3. Point `slam_mapper.py` to the stream URL:
```powershell
python slam_mapper.py --src "http://192.168.1.42:8080/video" --env indoor
```

### 2. From an iPhone Camera (RTSP / DroidCam)
```powershell
# DroidCam URL
python slam_mapper.py --src "http://192.168.1.42:4747/video" --env indoor

# RTSP Stream
python slam_mapper.py --src "rtsp://192.168.1.42:8554/live" --env indoor
```

### 3. From a USB / Built-in Webcam
```powershell
python slam_mapper.py --src 0 --env indoor --hfov 66
```

### 4. From the Drone Camera (UDP Stream)
```powershell
# On the drone Raspberry Pi:
./onboard/pi_stream.sh <GROUND_STATION_IP> 5600

# On the ground station:
python slam_mapper.py --src "udp://@:5600" --env outdoor
```

---

## Controls & Outputs

While the SLAM window is active:
- **`s`**: Manually saves the current 3D map and trajectory to disk.
- **`q`**: Quits the pipeline and saves final outputs.

Generated outputs:
- **`map.ply`**: Coloured 3D point cloud map (view in MeshLab, CloudCompare, or Open3D).
- **`trajectory.txt`**: KITTI-format 6-DoF camera pose history.
- **`occupancy.png`**: Top-down 2D floorplan occupancy map.
