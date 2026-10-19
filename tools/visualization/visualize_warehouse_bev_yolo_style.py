import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DATASET_DIR = None
RESULT_PKL = None
OUT_DIR = None

# Use your current PointPillars range
X_MIN, X_MAX = -12.8, 8.0
Y_MIN, Y_MAX = -6.4, 12.8

# Tiny test model, so keep low, but not too low
SCORE_THRESH = 0.20
MAX_PRED_BOXES = 30


def box_corners_xy(box):
    x, y, z, l, w, h, yaw = box

    corners = np.array([
        [ l / 2,  w / 2],
        [ l / 2, -w / 2],
        [-l / 2, -w / 2],
        [-l / 2,  w / 2],
        [ l / 2,  w / 2],
    ])

    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)],
    ])

    return corners @ R.T + np.array([x, y])


def load_gt_boxes(frame_id):
    label_path = DATASET_DIR / "labels" / f"{frame_id}.txt"
    names, boxes = [], []

    if not label_path.exists():
        return names, boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue

            # OpenPCDet label format:
            # x y z l w h yaw class
            try:
                box = [float(v) for v in parts[:7]]
                name = parts[7]
            except ValueError:
                name = parts[0]
                box = [float(v) for v in parts[1:]]

            names.append(name)
            boxes.append(np.array(box, dtype=float))

    return names, boxes


def valid_box(box):
    box = np.asarray(box, dtype=float)

    if box.shape[0] != 7:
        return False

    if not np.all(np.isfinite(box)):
        return False

    x, y, z, l, w, h, yaw = box

    if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX):
        return False

    if not (0.05 <= l <= 8.0 and 0.05 <= w <= 8.0 and 0.05 <= h <= 5.0):
        return False

    if abs(yaw) > 20.0:
        return False

    return True


def main():
    global DATASET_DIR, RESULT_PKL, SCORE_THRESH, OUT_DIR
    parser = argparse.ArgumentParser(description="Compare labels with TRUSTED OpenPCDet result.pkl")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True,
                        help="Trusted local pickle only; unpickling can execute code")
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--score", type=float, default=SCORE_THRESH)
    parser.add_argument("--output", type=Path, default=Path("visual_outputs"))
    args = parser.parse_args()
    DATASET_DIR, RESULT_PKL, SCORE_THRESH = args.data, args.results, args.score
    OUT_DIR = args.output
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PKL, "rb") as f:
        results = pickle.load(f)

    print("Available frames:")
    for i, r in enumerate(results):
        frame_id = r.get("frame_id", f"index_{i}")
        print(f"[{i}] {frame_id} raw_detections={len(r.get('name', []))}")

    idx = args.frame_index
    if idx is None:
        idx = int(input("Choose frame index: ").strip())
    result = results[idx]

    frame_id = result["frame_id"]
    points = np.load(DATASET_DIR / "points" / f"{frame_id}.npy")
    xyz = points[:, :3]

    # Keep only visible range points
    mask = (
        (xyz[:, 0] >= X_MIN) & (xyz[:, 0] <= X_MAX) &
        (xyz[:, 1] >= Y_MIN) & (xyz[:, 1] <= Y_MAX) &
        (xyz[:, 2] >= -1.0) & (xyz[:, 2] <= 4.0)
    )
    xyz = xyz[mask]

    print("\nFrame:", frame_id)
    print("Visible points:", xyz.shape)

    plt.figure(figsize=(12, 10))

    # Point cloud
    plt.scatter(xyz[:, 0], xyz[:, 1], s=0.25, c="lightgray", label="LiDAR points")

    # Ground truth boxes: green
    gt_names, gt_boxes = load_gt_boxes(frame_id)
    gt_count = 0

    for name, box in zip(gt_names, gt_boxes):
        if not valid_box(box):
            continue

        c = box_corners_xy(box)
        plt.plot(c[:, 0], c[:, 1], color="limegreen", linewidth=2.5)
        plt.text(
            box[0], box[1],
            f"GT {name}",
            color="green",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="green")
        )
        gt_count += 1

    # Prediction boxes: red
    pred_names = result.get("name", [])
    pred_scores = result.get("score", [])
    pred_boxes = result.get("boxes_lidar", result.get("box3d_lidar", []))

    pred_items = []

    for name, score, box in zip(pred_names, pred_scores, pred_boxes):
        box = np.asarray(box, dtype=float)

        if not np.isfinite(score):
            continue

        if score < SCORE_THRESH:
            continue

        if not valid_box(box):
            continue

        pred_items.append((name, float(score), box))

    pred_items = sorted(pred_items, key=lambda x: x[1], reverse=True)
    pred_items = pred_items[:MAX_PRED_BOXES]

    for name, score, box in pred_items:
        c = box_corners_xy(box)
        plt.plot(c[:, 0], c[:, 1], color="red", linewidth=2.0)
        plt.text(
            box[0], box[1],
            f"{name} {score:.2f}",
            color="red",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="red")
        )

    plt.title(f"YOLO-style 3D LiDAR BEV Detection | {frame_id}\nGreen = CVAT labels, Red = AI predictions")
    plt.xlabel("X forward/backward")
    plt.ylabel("Y left/right")
    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="upper right")

    out_path = OUT_DIR / f"{frame_id}_bev_yolo_style.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")

    print("\nSaved:", out_path)
    print("GT boxes shown:", gt_count)
    print("Prediction boxes shown:", len(pred_items))

    if len(pred_items) == 0:
        print("\nNo prediction boxes shown.")
        print("This means the tiny 20-frame model is not predicting valid boxes confidently yet.")
        print("The green CVAT boxes should still appear.")

    plt.show()


if __name__ == "__main__":
    main()
