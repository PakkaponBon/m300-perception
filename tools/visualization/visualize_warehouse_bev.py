import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DATASET_DIR = None
RESULT_PKL = None
OUT_DIR = None

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

            # OpenPCDet format:
            # x y z l w h yaw class
            if parts[0].replace(".", "", 1).replace("-", "", 1).isdigit():
                box = [float(v) for v in parts[:7]]
                name = parts[7]
            else:
                name = parts[0]
                box = [float(v) for v in parts[1:]]

            names.append(name)
            boxes.append(box)

    return names, boxes


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
        print(f"[{i}] {frame_id} detections={len(r.get('name', []))}")

    idx = args.frame_index
    if idx is None:
        idx = int(input("Choose frame index: ").strip())
    result = results[idx]

    frame_id = result["frame_id"]
    points = np.load(DATASET_DIR / "points" / f"{frame_id}.npy")

    print("Frame:", frame_id)
    print("Points shape:", points.shape)
    print("XYZ min:", points[:, :3].min(axis=0))
    print("XYZ max:", points[:, :3].max(axis=0))

    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, s=0.2, c="gray")

    # Ground truth: green
    gt_names, gt_boxes = load_gt_boxes(frame_id)
    for name, box in zip(gt_names, gt_boxes):
        c = box_corners_xy(box)
        plt.plot(c[:, 0], c[:, 1], "g-", linewidth=2)
        plt.text(box[0], box[1], f"GT {name}", color="green")

    # Predictions: red
    pred_names = result.get("name", [])
    pred_scores = result.get("score", [])
    pred_boxes = result.get("boxes_lidar", result.get("box3d_lidar", []))

    shown = 0

    pred_items = []
    for name, score, box in zip(pred_names, pred_scores, pred_boxes):
        box = np.asarray(box, dtype=float)

        if not np.isfinite(score):
            continue
        if not np.all(np.isfinite(box)):
            continue
        if score < SCORE_THRESH:
            continue

        pred_items.append((name, float(score), box))

    pred_items = sorted(pred_items, key=lambda x: x[1], reverse=True)
    pred_items = pred_items[:MAX_PRED_BOXES]

    for name, score, box in pred_items:
        c = box_corners_xy(box)
        plt.plot(c[:, 0], c[:, 1], "r-", linewidth=2)
        plt.text(box[0], box[1], f"PRED {name} {score:.2f}", color="red")
        shown += 1

    plt.title(f"{frame_id} | green=CVAT labels, red=AI predictions")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)

    out_path = OUT_DIR / f"{frame_id}_bev.png"
    plt.savefig(out_path, dpi=200)
    print("Saved visual:", out_path)
    print("Prediction boxes shown:", shown)


if __name__ == "__main__":
    main()
