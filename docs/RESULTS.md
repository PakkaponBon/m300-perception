# Recorded experiments

These are transcribed experiment logs from development, not a fresh benchmark reproduced by this repository's offline tests. Private samples and weights are not published. AP is not classification accuracy.

## Combined dataset, scratch training

Prepared split counts: train 706 (12 negative frames), validation 147 (0 negatives), test 150 (0 negatives); 6 of 1,009 prepared samples excluded at temporal boundaries. Validation contains 180 labelled objects.

| Experiment | Validation 3D AP_R40 @ IoU 0.5 | BEV AP_R40 @ 0.5 | 3D AP_R40 @ 0.25 |
|---|---:|---:|---:|
| Earlier scratch run, epoch 80 | 43.7108% | 46.8542% | 58.8791% |
| Scratch run, epoch 200, batch 4 | 55.0221% | 65.2196% | 78.0385% |

These logs use the same reported 147-frame validation split. Training duration/batch/schedule changed together, so the result is not evidence that increasing epochs alone caused the improvement. KITTI-format custom evaluation repeats difficulty columns because the custom labels do not provide genuine KITTI visibility/difficulty metadata. Do not interpret them as distinct easy/moderate/hard benchmarks. The synthetic 2D bbox AP is not a meaningful camera-based detection score here.

## Later small-dataset fine-tune

Experiment ID: `finetune_fix_13_10_20261013_205930`; initialized from an earlier checkpoint at epoch 196, trained for 200 fine-tuning epochs with batch 4 and learning rate 0.0003.

Prepared subset: train 160 (18 negatives), validation 32 (0 negatives), test 33 (0 negatives), 4 temporal-gap exclusions. This experiment used the corrected new recording, not a combined replay of all older training data.

Epoch 200 validation:

- 3D AP_R40 @ IoU 0.5: **71.4063%**
- BEV AP_R40 @ IoU 0.5: **74.1406%**
- 3D AP_R40 @ IoU 0.25: **77.5%**
- recall @ IoU 0.3: **1.0** (32/32 ground-truth objects)
- recall @ IoU 0.5: **0.9375** (30/32)
- recall @ IoU 0.7: **0.25**

These results use a different, very small validation subset and **cannot be directly ranked against the combined-dataset experiment**. Recall 1.0 at IoU 0.3 does not mean zero false positives, 100% accuracy, or correct chair rejection.

## What remains unverified

- No held-out test-set result is claimed.
- Validation scenes without any people were absent in these splits; false alarms per minute in empty scenes were not established by AP.
- Same-recording temporal splits can retain scene similarity despite boundary gaps.
- New rooms, shelf/chair lookalikes, sensor-height changes, and moving-platform accumulation can cause failures.
- A lower training loss does not prove generalization or absence of overfitting.
- No collision-avoidance or safety performance claim is made.
