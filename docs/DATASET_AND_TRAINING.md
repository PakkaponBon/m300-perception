# Dataset preparation and training

Run these examples from the repository root unless noted. Replace paths with your own; outputs should be **new** paths. No datasets or model weights are in Git.

These examples retain the original M300 training profile. For another sensor,
configure `--topic`, `--frame-id`, and `--intensity-field` during extraction
and the matching live input settings; see [ANY_LIDAR.md](ANY_LIDAR.md).

## 1. Record raw MCAP

Start the LiDAR driver first. Record raw inputs, not just person crops or front-ROI points:

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
export M300_DATA_ROOT=/absolute/path/to/dataset_storage

ros2 bag record -s mcap \
  -o "$M300_DATA_ROOT/site_$(date +%Y%m%d_%H%M%S)" \
  /pointcloud /custommsg /imu
```

Ctrl+C closes the recording. Use `ros2 bag info /path/to/bag` to inspect actual topics and counts. ROS 2 needs its MCAP storage plugin installed.

## 2. Extract a trailing 0.3-second input window

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 tools/dataset/extract_mcap_dataset.py \
  --bag "$M300_DATA_ROOT/site_A" \
  --output "$M300_DATA_ROOT/extracted_A" \
  --decay-time 0.3 --allow-uncompensated --stride 5
```

For another recording, keep exactly the same standard:

```bash
/usr/bin/python3 tools/dataset/extract_mcap_dataset.py \
  --bag "$M300_DATA_ROOT/site_B" \
  --output "$M300_DATA_ROOT/extracted_B" \
  --decay-time 0.3 --allow-uncompensated --stride 5 \
  --reference-manifest "$M300_DATA_ROOT/extracted_A/manifest.json"
```

Outputs: `points/*.bin` (little-endian float32, **N×4** XYZI), `frames.csv`, and a completion/standard manifest. `stride` reduces exported samples, not the frames used inside each temporal window.

Defaults retain raw intensity including zero, retain finite zero-XYZ points, remove a point if any XYZI field is NaN/Inf, and apply no crop unless `--bounds` is supplied. No voxelization, ground/robot removal, deduplication, intensity averaging, or motion compensation is performed. The model's later range crop and voxelization are separate.

The inclusive window is `[latest_stamp − 0.3 s, latest_stamp]`; count varies with timestamps and is **not guaranteed to be exactly three messages**. Warmup emits partial windows. Use `--decay-time 0` for single-message samples, but do not mix that standard into 0.3-second training.

## 3. Pack one or more sources for CVAT

```bash
python3 tools/dataset/pack_cvat_dataset.py \
  --input "$M300_DATA_ROOT/extracted_A" \
  --input "$M300_DATA_ROOT/extracted_B" \
  --sample-every 6 \
  --output "$M300_DATA_ROOT/cvat_A_B_decay03.zip"
```

With stride-5 extraction, `--sample-every 6` selects every 30th original message per source. It does not change the 0.3-second clouds. Omit this option to pack all extracted samples.

ZIP entries use `velodyne_points/data/*.bin`. Multi-source names receive numeric suffixes `_00`, `_01`, etc.; the adjacent `.manifest.json` records provenance. **Retain the original extracted directories and manifests** for annotation matching. The packer never joins temporal windows across bags.

## 4. Label in CVAT

Upload the archive as a point-cloud task. Use a `person` cuboid class and label every visible target person, including seated people. Empty frames with no person are valid negatives; do not label standalone chairs as people. Keep cuboids upright (roll/pitch zero), tight and consistent. Use a consistent front-direction/yaw convention and verify exported dimensions; do not infer object orientation only from a scene-axis colour.

Export as **Sly Point Cloud Format 1.0**, including the point-cloud data, and unzip into a local directory. The preparation script expects `*/ann/*.pcd.json` and matching `*/pointcloud/*.pcd` with uncompressed binary float32 XYZI. Unsupported formats, ambiguous source matches, or incompatible standards fail rather than guessing.

## 5. Prepare and split

Dry-run first; repeat with `--write` to create the dataset:

```bash
python3 tools/dataset/prepare_m300_dataset.py \
  --export "$M300_DATA_ROOT/cvat_export" \
  --source "$M300_DATA_ROOT/extracted_A" \
  --source "$M300_DATA_ROOT/extracted_B" \
  --output "$M300_DATA_ROOT/openpcdet_m300" \
  --ratios 0.70 0.15 0.15 --gap-seconds 1.0
```

```bash
python3 tools/dataset/prepare_m300_dataset.py \
  --export "$M300_DATA_ROOT/cvat_export" \
  --source "$M300_DATA_ROOT/extracted_A" \
  --source "$M300_DATA_ROOT/extracted_B" \
  --output "$M300_DATA_ROOT/openpcdet_m300" \
  --ratios 0.70 0.15 0.15 --gap-seconds 1.0 --write

python3 tools/dataset/build_m300_infos.py \
  --data "$M300_DATA_ROOT/openpcdet_m300"
```

Matching checks frame candidates against actual XYZI content across sources. Negative samples receive empty label files; they still have valid points. Labels are `cx cy cz dx dy dz yaw person`. Splits are chronological **within each source**, with a boundary gap accounting for accumulated windows, not shuffled adjacent frames. Very short recordings can fail because no samples remain in a split.

If correcting/deleting annotations from an already-prepared dataset, use `--reference-splits /old/prepared/samples.csv` to preserve held-out identities. It accepts only a corrected subset, not new recordings. For generalization evaluation, reserve entirely new scenes/recordings separately; temporal splits alone are not enough.

## 6. Train 200 epochs from scratch

Install the overlay as in [setup](SETUP.md), then use an absolute fresh run directory:

```bash
export M300_PROJECT_ROOT="$(pwd)"
export M300_OPENPCDET_ROOT=/absolute/path/to/OpenPCDet
export M300_PYTHON=/absolute/path/to/openpcdet_env/bin/python
export M300_DATASET="$M300_DATA_ROOT/openpcdet_m300"
M300_RUN="$M300_DATA_ROOT/runs/scratch_$(date +%Y%m%d_%H%M%S)"

cd "$M300_OPENPCDET_ROOT/tools"
"$M300_PYTHON" train.py \
  --cfg_file cfgs/custom_models/pointpillar_m300_decay03.yaml \
  --epochs 200 --batch_size 4 --workers 2 --fix_random_seed \
  --max_ckpt_save_num 5 --num_epochs_to_eval 5 \
  --extra_tag "$M300_RUN" \
  --set DATA_CONFIG.DATA_PATH "$M300_DATASET"
```

A fresh output path matters: OpenPCDet may resume an existing run automatically. The absolute `--extra_tag` is intentionally used with the pinned implementation to place generated runs on your selected storage disk.

Augmentation is applied **during training**, not duplicated into the prepared dataset: random world flip along X, rotation −0.35…0.35 rad, scale 0.95…1.05. GT-database sampling is disabled. Defaults in YAML are 80 epochs/batch 2/LR 0.003; the command overrides epochs and batch to 200/4.

For weights-initialized fine-tuning, add `--pretrained_model /trusted/checkpoint.pth` and set a lower LR:

```bash
# Add to the train.py invocation; use a different fresh --extra_tag:
--pretrained_model /trusted/checkpoint.pth
# In the final --set section:
DATA_CONFIG.DATA_PATH "$M300_DATASET" OPTIMIZATION.LR 0.0003
```

These last lines illustrate arguments, not a standalone shell command. `--pretrained_model` initializes weights; `--ckpt` resumes optimizer/epoch state and is a different operation. Mix reviewed older training data if retaining older scenes matters.

## 7. Evaluate and visualize training

Default evaluation is **validation**. Use validation to choose a checkpoint; evaluate held-out test only after selection:

```bash
# From OpenPCDet/tools, using the exported paths above:
"$M300_PYTHON" test.py \
  --cfg_file cfgs/custom_models/pointpillar_m300_decay03.yaml \
  --ckpt "$M300_RUN/ckpt/checkpoint_epoch_200.pth" --batch_size 4 \
  --extra_tag "$M300_RUN" \
  --set DATA_CONFIG.DATA_PATH "$M300_DATASET"

"$M300_PYTHON" -m tensorboard.main --logdir "$M300_DATA_ROOT/runs" --port 6006
```

Open `http://localhost:6006` on the training machine. Install TensorBoard in the selected environment if absent. For a final test run, override `DATA_CONFIG.DATA_SPLIT.test test` and `DATA_CONFIG.INFO_PATH.test custom_infos_test.pkl` in `--set`, using a separate evaluation tag to avoid mixing reports.

An optional **GPU forward/backward smoke check** (no optimizer step or saved checkpoint) is available:

```bash
"$M300_PYTHON" "$M300_PROJECT_ROOT/tools/training/check_m300_training.py" \
  --openpcdet "$M300_OPENPCDET_ROOT" --data "$M300_DATASET"
```

It intentionally requires at least two negative training frames and positive examples. It is not part of the offline tests and does not report model accuracy.

Finally set `M300_CHECKPOINT` to your selected checkpoint and run the AI launcher from this repo. Do not treat the last epoch as automatically best.
## Optional saved-prediction viewers

Install Matplotlib for BEV PNG output or Open3D for the interactive 3D viewer.
These older development viewers are included with portable path arguments:

```bash
python3 tools/visualization/visualize_warehouse_bev.py \
  --data /path/to/prepared_dataset --results /path/to/result.pkl \
  --frame-index 0 --score 0.3 --output /path/to/plots

python3 tools/visualization/visualize_warehouse_predictions.py \
  --data /path/to/prepared_dataset --results /path/to/result.pkl \
  --frame-index 0 --score 0.3
```

The YOLO-style BEV alternative takes the same arguments as the first viewer.
Green boxes are labels, red boxes are predictions. Pickle loading is unsafe for
untrusted input; use only your own evaluation output. No images or private
point clouds are uploaded by these tools.
