# Third-party provenance

## OpenPCDet

- Upstream: https://github.com/open-mmlab/OpenPCDet
- Pinned base: `233f849829b6ac19afb8af8837a0246890908755`
- License: Apache-2.0; retained in [OpenPCDet-LICENSE](OpenPCDet-LICENSE).
- The framework is not vendored. `patches/openpcdet-m300.patch` contains local changes to the dataset registry, empty-ground-truth handling, custom evaluation, and trusted legacy-checkpoint loading, plus the added overlap adapter.
- `configs/openpcdet/custom_models/pointpillar_m300_decay03.yaml` adapts the upstream PointPillars configuration for one-class M300 data. This is an upstream-based model configuration, not a new model architecture.
- PointPillars paper: https://arxiv.org/abs/1812.05784

## Pacecat / BlueSeaLiDAR M300

- Upstream: https://github.com/BlueSeaLiDAR/m300
- Local base revision: `70bf9bc72f9b56d9b0bc818be32ba5beceddbd46`
- `ros2_ws/src/pacecat_m300_driver` and `pacecat_m300_inter` are a snapshot of the locally used ROS 2 packages, including local configuration/launch adjustments.
- Driver code, SDK, custom messages, and interfaces are vendor work, not claimed as original work by this portfolio.
- The inspected upstream snapshot has **no top-level license file**, and both ROS package manifests contain `TODO: License declaration`. No license grant is inferred or replaced here; downstream redistribution/use should clarify the vendor's terms.
- The embedded readerwriterqueue library retains its own [license](../ros2_ws/src/pacecat_m300_driver/3rdparty/readerwriterqueue/LICENSE.md).
- Original upstream README is retained as [m300-upstream-README.md](m300-upstream-README.md); its repository-relative links refer to upstream, not this reduced snapshot.

## Project license status

No new blanket license has been assigned to this repository. Third-party licenses and notices remain applicable to their respective files. Public visibility does not itself grant an open-source license for all project or vendor code.
