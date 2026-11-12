# Commands Reference Guide

Complete reference of commands for running the real-time drone spatial mapping system, from setup and calibration to live phone streaming, drone streaming, and map inspection.

---

## 1. Environment Setup

Run these commands in PowerShell inside the workspace root (`z:\coding\mapping`):

```powershell
# 1. Create Python virtual environment (if not already created)
python -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\activate

# 3. Install PyTorch with CUDA acceleration (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install remaining ground-station dependencies
pip install -r requirements.txt
```

---

## 2. Camera Calibration (Optional)

Using a standard printed checkerboard pattern (e.g. 9×6 inner corners, 25 mm square size):

```powershell
# Calibrate from local USB / Laptop Webcam
python calibrate.py --src 0 --rows 9 --cols 6 --square-size 0.025 --out calib.npz

# Calibrate from Phone IP Webcam Stream
python calibrate.py --src "http://<PHONE_IP>:8080/video" --rows 9 --cols 6 --square-size 0.025 --out calib.npz
```

**Calibration Controls**:
- Press **`SPACE`**: Capture current frame (capture 15–20 angles).
- Press **`c`**: Calculate intrinsics matrix `K` and distortion coefficients, saving to `calib.npz`.
- Press **`q`**: Exit calibration.

---

## 3. Live SLAM Mapping Commands

All runs automatically generate a unique timestamped session folder under `records/` (`records/YYYYMMDD_HHMMSS/`) to store the 3D point cloud, trajectory poses, and occupancy grid.

### A. Android Phone Camera (IP Webcam)
```powershell
# Default indoor mapping
python slam_mapper.py --src "http://<PHONE_IP>:8080/video" --env indoor

# With custom camera calibration file
python slam_mapper.py --src "http://<PHONE_IP>:8080/video" --env indoor --calib calib.npz

# Verbose logging
python slam_mapper.py --src "http://<PHONE_IP>:8080/video" --env indoor -v
```

### B. iPhone Camera (DroidCam / RTSP)
```powershell
# DroidCam stream
python slam_mapper.py --src "http://<PHONE_IP>:4747/video" --env indoor

# Generic RTSP live stream app
python slam_mapper.py --src "rtsp://<PHONE_IP>:8554/live" --env indoor
```

### C. Local USB / Built-in Webcam
```powershell
# Using default webcam index 0 with 66 degree HFOV
python slam_mapper.py --src 0 --env indoor --hfov 66
```

### D. Drone Raspberry Pi Live Stream (UDP)
```powershell
# Ground station listener on port 5600
python slam_mapper.py --src "udp://@:5600" --env outdoor
```

---

## 4. Onboard Drone Commands (Raspberry Pi 3B+)

Run these commands on the drone via SSH:

### A. Probe Flight Controller Telemetry
```bash
# Discover which MAVLink telemetry messages the flight controller broadcasts
python3 onboard/mavlink_bench.py --serial /dev/ttyAMA0 --baud 57600 --duration 15
```

### B. Forward Telemetry over UDP
```bash
# Forward MAVLink packets to laptop ground station over UDP port 14550
python3 onboard/mavlink_forward.py --serial /dev/ttyAMA0 --baud 57600 --udp-target <LAPTOP_IP>:14550
```

### C. Hardware-Accelerated Video Stream (H.264 over UDP)
```bash
chmod +x onboard/pi_stream.sh

# Stream 640x480 @ 30 fps (~4 Mbps low-latency)
./onboard/pi_stream.sh <LAPTOP_IP> 5600

# Stream 1280x720 @ 30 fps (~8 Mbps)
./onboard/pi_stream.sh <LAPTOP_IP> 5600 1280x720 8000000
```

---

## 5. Live Runtime Keybindings

While the mapping GUI windows are active:
- **`s`**: Manually saves current intermediate mapping progress to `records/<TIMESTAMP>/`.
- **`q`** or **`Ctrl + C`**: Gracefully stops the stream, outputs final timing reports, and saves:
  - `records/<TIMESTAMP>/map.ply` (Dense 3D point cloud)
  - `records/<TIMESTAMP>/trajectory.txt` (6-DoF KITTI poses)
  - `records/<TIMESTAMP>/occupancy.png` (2D top-down occupancy grid)

---

## 6. Inspecting Generated Maps

### Visualizing the 3D Point Cloud via Open3D
```powershell
python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('records/<FOLDER_NAME>/map.ply'); o3d.visualization.draw_geometries([pcd])"
```
*(You can also drag and drop `map.ply` directly into [CloudCompare](https://www.cloudcompare.org/) or [MeshLab](https://www.meshlab.net/))*.

### Viewing the 2D Occupancy Grid
Open `records/<FOLDER_NAME>/occupancy.png` in any image viewer.

---

## 7. Advanced CLI Options

| Argument | Options / Default | Description |
|---|---|---|
| `--src` | *Required* | Camera stream URL (`http://...`, `rtsp://...`, `udp://...`) or webcam index (`0`). |
| `--env` | `indoor` (default), `outdoor` | Selects depth model scale & range ($10\text{ m}$ indoor vs $80\text{ m}$ outdoor). |
| `--calib` | `None` | Path to calibration file (e.g. `calib.npz`). Overrides `--hfov`. |
| `--hfov` | `66.0` (degrees) | Horizontal Field of View estimate when no calibration file is supplied. |
| `--voxel-size` | `0.02` indoor, `0.05` outdoor | Voxel downsampling size in meters ($0.02 = 2\text{ cm}$). |
| `--depth-subsample` | `2` (default) | Subsampling factor for back-projecting depth pixels (use `4` for lower GPU load). |
| `--device` | `cuda` (default), `cpu` | PyTorch execution device for the depth neural network. |
| `--no-vis` | Flag | Headless execution without GUI display windows. |
| `--out-dir` | `records/<timestamp>` | Specify a custom destination path for output files. |
| `-v`, `--verbose` | Flag | Enables detailed DEBUG-level logging. |

