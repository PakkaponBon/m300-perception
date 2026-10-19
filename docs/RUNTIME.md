# Runtime interfaces and settings

All measurements are sensor-relative metres. `m300_lidar` is the original
default, not a fixed requirement; topic/frame and AI field/model settings are
configurable as described in [ANY_LIDAR.md](ANY_LIDAR.md). No TF transform,
global localization, or persistent map is supplied.

## AI path

`runtime/live_warehouse_detector_ros2.py` consumes `/pointcloud` with sensor-data QoS (BEST_EFFORT). It concatenates XYZI rows from an inclusive 0.3-second header-time window, drops nonfinite XYZI points, and feeds the custom PointPillars model. Older points retain their original intensity; they are not averaged, added, deduplicated, or exponentially weighted.

The model's YAML crops to X [−12.8, 8.0], Y [−6.4, 12.8], Z [−1.0, 4.0]. Detection outside this asymmetric extent is not supported just because RViz draws a point there.

| Topic | Type | Meaning |
|---|---|---|
| `/ai_detections/boxes` | visualization_msgs/MarkerArray | Person cuboids, labels, origin, and display-zone guides |
| `/ai_detections/person_points` | sensor_msgs/PointCloud2 | XYZI + RGB points cropped by current matched person boxes |
| `/ai_detections/background_points` | sensor_msgs/PointCloud2 | Sparse scene context; not semantic walls or an edge detector |

Box labels show model score and horizontal **box-centre** distance. Scores are not calibrated probabilities. Point crops may include chair/background returns inside the predicted cuboid; this is not semantic segmentation. Held boxes do not synthesize held person points.

Current source constants: `SCORE_THRESH=0.3`, `DECAY_TIME_SEC=0.3`, `TRACKING_HOLD_SEC=0.3`, `TRACKING_MIN_HITS=2`, `TRACKING_MATCH_DISTANCE_M=0.8`. The tracker default constructor's 0.6 is overridden by the live detector's current threshold.

Person zones are red for centre range <1 m, yellow for <2 m, and green otherwise. The 3 m green guide is visual only, not the model's outer detection limit. Override red/yellow via the launcher flags `--deadzone` and `--warning`, or startup ROS parameters `deadzone_radius_m` and `warning_radius_m`.

Lightweight greedy tracking can swap identities and can preserve a false detection briefly. It does not distinguish an unmoving person from a chair or compensate for a moving LiDAR.

## Geometric front-obstacle path

`runtime/m300_front_obstacle_publisher.py` uses only the **latest** cloud, not the AI temporal window. It tests finite, nonzero XYZ points in a clipped triangular prism, with +X forward by default:

- forward 0.3…4.0 m, full opening 60°;
- lateral limit `abs(lateral) <= forward * tan(30°)`;
- sensor-relative Z −0.2…1.5 m;
- at least 5 selected points for `OCCUPIED`.

The 4 m maximum is **forward depth**, not radial XY range. The prism's far corners have greater radial distance. No plane fit, ground removal, clustering, object identification, or AI confidence is used.

Edit the current zone values near the top of the file and restart:

```python
RED_ZONE_MAX_M = 2
YELLOW_ZONE_MAX_M = 3
GREEN_ZONE_MAX_M = 4
```

Require `near < red < yellow < max distance`. Startup ROS parameters override them: `red_zone_max_m`, `yellow_zone_max_m`, `distance_m`, `full_angle_deg`, `forward_yaw_deg`, `near_m`, `min_z_m`, `max_z_m`, `min_points`, `input_timeout_sec` (see declarations in source for the full set).

| Topic suffix under `/front_obstacle` | Type | Meaning |
|---|---|---|
| `state` | std_msgs/String | CLEAR / OCCUPIED / NO_DATA / INVALID_FRAME / INVALID_CLOUD |
| `data_valid` | std_msgs/Bool | Whether the current input is usable and fresh |
| `detected` | std_msgs/Bool | Emitted only on valid clouds; never use alone |
| `point_count` | std_msgs/UInt32 | Points selected inside the ROI |
| `distance_m` | std_msgs/Float32 | Nearest selected XY radial distance |
| `zone_distance_m` | std_msgs/Float32 | Nearest selected forward depth used for zone classification |
| `zone` | std_msgs/String | RED / YELLOW / GREEN / UNKNOWN |
| `points` | sensor_msgs/PointCloud2 | Original selected XYZ when occupied; empty otherwise |
| `markers` | visualization_msgs/MarkerArray | Zone-coloured ROI wireframe, forward arrow, state text |

Exactly 2 m is red; exactly 3 m is yellow. CLEAR also uses green; green **occupied** is not clear. With fewer than the minimum number of points, a valid cloud is CLEAR. Distance is +inf when clear and NaN when invalid/no data. The 0.5-second input watchdog reports NO_DATA, UNKNOWN, and invalid data; it does not assert a false clear.

The RViz ROI wireframe changes by zone. Selected obstacle points remain red in the preset. Its raw cloud is grey. Both are display settings, independent of the published state.

**Neither the AI zones nor geometric zones issue a vehicle stop or constitute a safety-rated protective field.**
