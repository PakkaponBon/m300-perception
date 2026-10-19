# M300 Perception

ROS 2 LiDAR perception for indoor person detection, short-term tracking, and class-agnostic front-obstacle monitoring.

Built around a **Pacecat LDS-M300-E**, **OpenPCDet PointPillars**, and **RViz 2**. The project covers the complete workflow: record MCAP bags, accumulate point clouds, label in CVAT, prepare a custom dataset, train, and visualize live detections.

**The runtime is not locked to M300.** It can consume compatible ROS 2
`PointCloud2` from another driver's topic/frame. M300 remains the bundled
development example; using another sensor does not guarantee that the M300
checkpoint will generalize. See [using a different LiDAR](docs/ANY_LIDAR.md).

This is a portfolio project by [PakkaponBon](https://github.com/PakkaponBon). The companion project, [ros2-warehouse-traffic-monitor](https://github.com/PakkaponBon/ros2-warehouse-traffic-monitor), focuses on warehouse vehicle localization, traffic recording, congestion analysis, and historical heatmaps. The two repositories demonstrate complementary perception and monitoring work; **this repository does not claim an implemented cross-repository integration**.

## What is implemented

- **3D person detection:** raw XYZI input, a trailing 0.3-second accumulation window, and a custom single-class PointPillars model.
- **Lightweight tracking:** nearest-centre association, confirmation across detections, stable marker IDs, and a short hold through missed detections.
- **Sensor-centred visualization:** person boxes, distance/confidence labels, red/yellow/green person point clouds, sparse scene context, and a LiDAR origin indicator.
- **Front-obstacle monitoring without AI:** point-count occupancy inside a configurable triangular prism; reports occupied, clear, invalid, and no-data states.
- **Dataset tooling:** timestamp-aware extraction, multi-source CVAT ZIP packing, verified source matching, empty-label negatives, temporal train/validation/test splits, and OpenPCDet info generation.
- **LiDAR integration:** a snapshot of the ROS 2 driver and custom-message packages, with upstream attribution.
- **Reproducible dependency changes:** a patch against a pinned OpenPCDet revision rather than a second copy of the entire framework.

## System overview

```text
Pacecat M300 ROS 2 driver
  ├── /pointcloud ──┬── 0.3 s XYZI accumulation → PointPillars → tracking → RViz
  │                ├── latest-frame geometric ROI → /front_obstacle/* → RViz
  │                └── MCAP → extract → CVAT ZIP → label → prepare → train
  ├── /custommsg
  └── /imu
```

AI detection and geometric occupancy are **independent consumers**, not a fused detector. RViz's top-down view is sensor-relative visualization, **not SLAM or an occupancy map**.

## Repository layout

```text
runtime/                    Live AI detector, tracker, geometric obstacle node
tools/dataset/              Extract, pack, prepare, build infos
tools/training/             Optional model/data smoke check
tools/visualization/        Optional BEV and Open3D label/prediction viewers
tools/install_openpcdet_overlay.py
configs/openpcdet/          PointPillars and custom dataset YAML
scripts/                   AI + LiDAR + RViz / obstacle + RViz launchers
                           + driver-independent start_perception.sh
rviz/                      Two ready-to-use RViz presets
ros2_ws/src/                pacecat_m300_driver + pacecat_m300_inter
patches/                   Pinned OpenPCDet compatibility changes
tests/                     Offline tests; no ROS nodes, GPU, or hardware
docs/                      Setup, dataset/training, topics, results, limitations
third_party/               Upstream attribution and license notices
```

## Quick start

Environment used: Ubuntu 22.04, ROS 2 Humble, Python 3.10, and an NVIDIA CUDA-capable GPU for AI inference. Geometric obstacle monitoring does **not** need OpenPCDet or a GPU.

Already have another driver publishing a 3D point cloud?

```bash
PERCEPTION_INPUT_TOPIC=/vendor/points PERCEPTION_FRAME_ID=lidar_front \
  ./scripts/start_perception.sh obstacle
```

This entry point starts no LiDAR driver. For AI, select the input field,
model, and weights as described in [the generic setup](docs/ANY_LIDAR.md).
The original M300-specific commands below are retained.

1. Follow [setup](docs/SETUP.md) to build the ROS packages and, for AI, install OpenPCDet plus the overlay.
2. For AI, export your own paths. Weights are intentionally not published:

   ```bash
   export M300_OPENPCDET_ROOT=/absolute/path/to/OpenPCDet
   export M300_PYTHON=/absolute/path/to/openpcdet_env/bin/python
   export M300_CHECKPOINT=/absolute/path/to/trusted/checkpoint_epoch_200.pth
   ./scripts/start_m300_detection.sh
   ```

   This starts the driver, detector, and RViz. Do not start a second driver simultaneously.

3. For obstacle-only monitoring, start the driver separately, then:

   ```bash
   ./scripts/start_m300_obstacle.sh
   ```

   This starts only the obstacle publisher and RViz; it leaves the existing driver alone.

[Dataset and training commands](docs/DATASET_AND_TRAINING.md) · [Topics and settings](docs/RUNTIME.md) · [Results](docs/RESULTS.md) · [Limitations](docs/LIMITATIONS.md)

## Current settings

These are the source snapshot's settings, not validated safety thresholds:

| Component | Setting |
|---|---|
| AI input | Raw XYZI; 0.3 s trailing window; no motion compensation |
| AI display score threshold | 0.30 |
| Tracking | 2-hit confirmation; 0.3 s hold; 0.8 m XY association gate |
| Person zones | Box-centre XY distance: red <1 m, yellow <2 m, otherwise green |
| Front-obstacle zones | Forward depth: red ≤2 m, yellow ≤3 m, green up to 4 m |
| Front-obstacle ROI | +X; 60° full opening; near cutoff 0.3 m; sensor Z −0.2…1.5 m; ≥5 points |
| Model input extent | X [−12.8, 8.0], Y [−6.4, 12.8], Z [−1.0, 4.0] metres |

The model input range is **rectangular and asymmetric**, not a 20 m detection guarantee. Person zones use box-centre range; obstacle zones use forward depth. See [runtime details](docs/RUNTIME.md).

## Evaluation snapshot

A 200-epoch scratch run on the combined dataset improved validation **3D AP_R40 at IoU 0.5 from 43.71% to 55.02%** relative to an earlier 80-epoch run on the same validation set. Multiple training settings changed, so this is not an isolated epoch-count experiment.

A later small-dataset fine-tune reported 71.41% on its **different, 32-frame validation split**. These scores are not directly comparable, are not classification accuracy, and do not establish generalization. Dataset sizes, metrics, and caveats are recorded in [results](docs/RESULTS.md).

## Offline checks

```bash
python3 -m pip install -r requirements-offline.txt
python3 -m unittest discover -s tests -v
bash -n scripts/start_perception.sh scripts/start_m300_detection.sh scripts/start_m300_obstacle.sh
```

Checks do not launch ROS nodes, train a model, or access a sensor. An optional
[GitHub Actions template](docs/ci/offline.yml) is included but is not activated.

## Scope and attribution

PointPillars/OpenPCDet and the M300 vendor driver are existing upstream work. This project's contribution is the custom-data workflow, runtime integration, tracking/display logic, obstacle monitor, configurations, and deployment tooling. See [third-party notices](third_party/README.md).

Raw recordings, private annotations, CVAT exports, checkpoints, virtual environments, build products, and local credentials are excluded. A fresh clone can run offline tests and build the driver, but **AI inference requires separately supplied compatible weights**.

**Research/demo prototype only. No vehicle control, certified protective function, collision-avoidance guarantee, or substitute for safety-rated sensing.**
