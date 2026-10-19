# Using a different LiDAR

The runtime consumes **ROS 2 sensor_msgs/PointCloud2**, not M300 UDP packets.
M300 is the development sensor and bundled driver example, not a required input
device. Compatibility with a message format is **not** proof of model accuracy
on every sensor.

## Input contract

| Path | Required fields | Other requirements |
|---|---|---|
| Geometric obstacle monitor | Scalar numeric x, y, z | Coordinates in metres; correct frame/axes; enough valid returns |
| AI and dataset extractor | Scalar numeric x, y, z plus an explicitly selected intensity field | Raw values; positive header timestamps; 4-feature model/config/weights |

Extra fields such as ring, time, channel, and return type are ignored. Point
padding, row padding, scalar numeric datatypes, and either byte order are
decoded. Nonfinite XYZI points are dropped after float32 conversion by the
AI/extraction caller. Obstacle decoding also excludes all-zero XYZ returns.

`LaserScan`, vendor `CustomMsg`, and raw UDP data are **not** accepted directly.
Use the chosen vendor's driver or a deliberate conversion adapter first.
A 2D scan converted into a flat PointCloud2 does not supply the 3D body geometry
needed by this person detector.

No missing intensity is invented, and an alias such as `reflectivity` is not
silently normalized into another sensor's intensity scale. A different field
name can be selected explicitly, but its meaning and training distribution
still matter.

## Driver-independent launch

Start your sensor's ROS 2 driver separately. Use the actual cloud topic and
`header.frame_id`, not a renamed label for untransformed coordinates:

```bash
export PERCEPTION_INPUT_TOPIC=/vendor/points
export PERCEPTION_FRAME_ID=lidar_front

./scripts/start_perception.sh obstacle
```

This launches only the geometric obstacle node and RViz. It does not source
the M300 workspace or start a driver. Optional `PERCEPTION_WS_SETUP` sources
your already-built workspace if needed.

For AI, install the OpenPCDet overlay, supply compatible weights, then:

```bash
export PERCEPTION_OPENPCDET_ROOT=/absolute/path/to/OpenPCDet
export PERCEPTION_PYTHON=/absolute/path/to/openpcdet_env/bin/python
export PERCEPTION_CHECKPOINT=/absolute/path/to/trusted/checkpoint.pth
export PERCEPTION_INTENSITY_FIELD=intensity
export PERCEPTION_DECAY_TIME_SEC=0.3
export PERCEPTION_SCORE_THRESHOLD=0.3

./scripts/start_perception.sh ai
```

`PERCEPTION_MODEL_CONFIG` selects another compatible OpenPCDet YAML. If
omitted, the **M300-trained PointPillars preset** is used; that file's name,
anchors, crop, and temporal policy describe its original training setup, not a
universal sensor calibration. Its `_BASE_CONFIG_` must resolve relative to the
OpenPCDet tools directory or use an absolute path.

The launcher keeps RViz's fixed frame and raw-cloud topic aligned with the
selected input. RViz Humble supports the fixed-frame command-line override in
its [upstream implementation](https://github.com/ros2/rviz/blob/humble/rviz_common/src/rviz_common/visualizer_app.cpp).
No TF, axis conversion, deskew, or pose compensation is performed. Check axes,
mounting orientation, sensor height, point density, and intensity semantics
before reusing a checkpoint.

Startup-only AI ROS parameters:

`input_topic`, `expected_frame_id`, `intensity_field`, `decay_time_sec`,
`score_threshold`, `model_config`, and `checkpoint`. The generic launcher
uses the environment settings above; use those for topic/frame so the node
and RViz agree. Existing zone/tracking parameters remain available.

## Matching offline extraction

For a driver whose scalar signal is named `reflectivity`:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 tools/dataset/extract_mcap_dataset.py \
  --bag /path/to/mcap_bag --output /path/to/new_extraction \
  --topic /vendor/points --frame-id lidar_front \
  --intensity-field reflectivity \
  --decay-time 0.3 --allow-uncompensated --stride 5
```

Use the same selected field and temporal policy in live inference. The output
is still Nx4 `[x, y, z, intensity]`; the manifest records the actual source
frame and selected input-field name. Existing M300 defaults preserve the old
manifest standard/hash. Mixing different frames/aliases/standards is rejected
by the multi-source tools rather than silently treated as equivalent.

Some sensors represent reflectivity differently from intensity. Merely
selecting a field does not make its distribution equivalent to M300 training
data. Validate in held-out scenes, and retrain/fine-tune with reviewed examples
when needed. The supplied info builder remains a **0.3-second training profile**;
using another accumulation time also requires adapting the training workflow.

## Existing M300 setup

The driver packages under `ros2_ws/src` and the older
`scripts/start_m300_detection.sh` / `start_m300_obstacle.sh` remain for the
original machine. Their defaults and M300-prefixed environment variables are
kept for compatibility. The generic launcher is an additional entry point.

Keeping that vendor snapshot does not resolve its upstream license ambiguity;
the [third-party notices](../third_party/README.md) still apply.
