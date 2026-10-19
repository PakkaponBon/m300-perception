# Setup

These are user-run installation commands. Launchers never configure the network, install dependencies, build packages, or download weights automatically.

The driver build in section 1 is **optional for non-M300 sensors**. If your
own driver already publishes PointCloud2, use the
[driver-independent entry point](ANY_LIDAR.md) instead; no M300 workspace is
required for the core perception nodes.

## 1. Clone and build the LiDAR workspace

Prerequisites: Ubuntu 22.04, ROS 2 Humble desktop, colcon, and rosdep. The ROS packages and SDK are included under `ros2_ws/src`.

```bash
git clone https://github.com/PakkaponBon/m300-perception.git
cd m300-perception
source /opt/ros/humble/setup.bash
rosdep install --from-paths ros2_ws/src --ignore-src -r -y
cd ros2_ws
colcon build --packages-select pacecat_m300_inter pacecat_m300_driver
source install/setup.bash
```

Inspect `ros2_ws/src/pacecat_m300_driver/params/LDS-M300-E.yaml` **before running**. The saved setup uses:

- sensor IP `192.168.158.98`, command port `6543`, listen port `6668`;
- adapter `eno1` (change for your Ethernet interface);
- frame `m300_lidar`; point cloud, custom message, and IMU enabled;
- `timemode: 1`; matrix transform disabled (`mr_enable: 0`).

These are example device settings, not a command to change your host network. The host Ethernet connection must already be configured appropriately. Do not enable an unexpected coordinate transform: trained boxes and filters assume the recorded sensor frame.

Start the driver:

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash  # from repository root
ros2 launch pacecat_m300_driver LDS-M300-E.launch
```

Check real samples, not just publisher discovery:

```bash
ros2 topic echo /pointcloud --qos-reliability best_effort --field header --once
ros2 topic hz /pointcloud
```

## 2. Geometric obstacle monitor only

Requires the driver plus ROS Python message packages and NumPy. No CUDA or model:

```bash
./scripts/start_m300_obstacle.sh
```

The launcher sources Humble automatically. Its default current ROI is 4 m deep. For the earlier 1.5 m demonstration geometry, override **all three** limits:

```bash
./scripts/start_m300_obstacle.sh --ros-args \
  -p red_zone_max_m:=0.5 -p yellow_zone_max_m:=1.0 -p distance_m:=1.5
```

The driver must be running separately. Do not interpret no data as clear space.

## 3. OpenPCDet for AI

Use a **dedicated** checkout and a compatible PyTorch/CUDA/spconv environment. An existing working OpenPCDet environment can be used, but the overlay installer will refuse conflicting files. It never resets your changes.

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git ../OpenPCDet
git -C ../OpenPCDet checkout 233f849829b6ac19afb8af8837a0246890908755

python3 tools/install_openpcdet_overlay.py --openpcdet ../OpenPCDet
python3 tools/install_openpcdet_overlay.py --openpcdet ../OpenPCDet --apply
```

Install/build OpenPCDet following its [installation guide](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/docs/INSTALL.md). Python must match ROS Humble's Python 3.10 ABI for live inference. A typical environment starts with `python3 -m venv --system-site-packages ../openpcdet_env`; install the GPU-compatible PyTorch and spconv packages before building the CUDA extensions. GPU architecture/toolkit compatibility must be checked on the target machine; this repo does not prescribe one universal CUDA wheel.

The overlay:

- retains valid empty-GT negative frames instead of resampling them away;
- parses empty label files as zero boxes with shape `(0, 7)`;
- avoids NumPy string truncation when mapping `person` to `Pedestrian`;
- offers overlap evaluation through the installed OpenPCDet CUDA op;
- disables the optional Argo2 dataset registration in this pinned installation;
- explicitly enables legacy PyTorch checkpoint loading with `weights_only=False`.

**Only load checkpoints you trust.** Legacy checkpoint loading can execute pickle payloads.

## 4. Configure local paths and launch AI

```bash
cp .env.example .env
# Edit .env with your OpenPCDet directory, Python executable, and checkpoint.
set -a
source .env
set +a
./scripts/start_m300_detection.sh --deadzone 1.0 --warning 2.0
```

The AI launcher starts the driver too. Stop any separately running copy first. The bundled workspace is the default; `M300_WS_SETUP` can select another already-built workspace. `.env` is ignored by Git and is not loaded automatically.

Weights and data are not bundled. Use a checkpoint trained with the supplied class mapping, point features, coordinates, model extent, and 0.3-second input policy.

Both launchers stop only their own process groups when Ctrl+C is pressed, RViz closes, or another launched process exits. RViz's child environment is cleaned of inherited Snap GTK/Qt paths to avoid mixing host and Snap libc libraries.

## Validation scope

Offline tests check temporal accumulation, packing, labels, split gaps, and tracker behavior. Source/shell syntax is checked separately. This is not a clean-machine CUDA build test or a live hardware test. The copied driver packages are not rebuilt automatically as part of publication.

An **inactive** GitHub Actions template is included at [ci/offline.yml](ci/offline.yml). To enable automatic checks, publish it at `.github/workflows/offline.yml` using GitHub authorization that permits workflow updates. It is not active in this initial publication.
