import argparse
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d


DATASET_DIR = None
RESULT_PKL = None

# Change this to 0.01 if no red prediction boxes appear
SCORE_THRESH = 0.01


def make_box(box, color):
    """
    box format: x, y, z, length, width, height, yaw
    """
    x, y, z, l, w, h, yaw = box

    center = np.array([x, y, z], dtype=float)
    extent = np.array([l, w, h], dtype=float)

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,          0.0,         1.0],
    ])

    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    line_set.paint_uniform_color(color)
    return line_set


def load_gt_boxes(frame_id):
    label_path = DATASET_DIR / "labels" / f"{frame_id}.txt"

    boxes = []
    names = []

    if not label_path.exists():
        print("No GT label file found:", label_path)
        return names, boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue

            # Supports both formats:
            # 1) class x y z l w h yaw
            # 2) x y z l w h yaw class   <-- your OpenPCDet format
            try:
                float(parts[0])
                nums = [float(x) for x in parts[:7]]
                cls = parts[7]
            except ValueError:
                cls = parts[0]
                nums = [float(x) for x in parts[1:]]

            boxes.append(nums)
            names.append(cls)

    return names, boxes

def main():
    global DATASET_DIR, RESULT_PKL, SCORE_THRESH
    parser = argparse.ArgumentParser(description="Compare labels with TRUSTED OpenPCDet result.pkl")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True,
                        help="Trusted local pickle only; unpickling can execute code")
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--score", type=float, default=SCORE_THRESH)
    args = parser.parse_args()
    DATASET_DIR, RESULT_PKL, SCORE_THRESH = args.data, args.results, args.score
    with open(RESULT_PKL, "rb") as f:
        results = pickle.load(f)

    print("Available prediction frames:")
    for i, r in enumerate(results):
        frame_id = r.get("frame_id", f"index_{i}")
        names = r.get("name", [])
        scores = r.get("score", [])
        print(f"  [{i}] {frame_id} detections={len(names)} scores={np.round(scores, 3) if len(scores) else []}")

    idx = args.frame_index
    if idx is None:
        idx = int(input("Choose frame index: ").strip())
    result = results[idx]

    frame_id = result.get("frame_id", None)
    if frame_id is None:
        raise RuntimeError("No frame_id found in result.pkl")

    print("\nShowing frame:", frame_id)

    points_path = DATASET_DIR / "points" / f"{frame_id}.npy"
    points = np.load(points_path)

    xyz = points[:, :3]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.paint_uniform_color([0.55, 0.55, 0.55])

    geometries = [cloud]

    # Ground truth boxes from CVAT labels: green
    gt_names, gt_boxes = load_gt_boxes(frame_id)
    print("\nGround truth boxes:")
    for name, box in zip(gt_names, gt_boxes):
        print("  GT:", name, np.round(box, 3))
        geometries.append(make_box(box, [0.0, 1.0, 0.0]))

    # Prediction boxes from AI model: red
    pred_names = result.get("name", [])
    pred_scores = result.get("score", [])
    pred_boxes = result.get("boxes_lidar", result.get("box3d_lidar", []))

    print("\nPrediction boxes:")
    shown = 0
    for name, score, box in zip(pred_names, pred_scores, pred_boxes):
        if score < SCORE_THRESH:
            continue

        print(f"  PRED: {name} score={score:.3f} box={np.round(box, 3)}")
        geometries.append(make_box(box, [1.0, 0.0, 0.0]))
        shown += 1

    if shown == 0:
        print("\nNo prediction boxes above threshold.")
        print("Try changing SCORE_THRESH to 0.01 near the top of this script.")

    print("\nLegend:")
    print("  gray points = lidar point cloud")
    print("  green boxes = your CVAT labels")
    print("  red boxes = AI predictions")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Warehouse 3D Detection - {frame_id}",
        width=1400,
        height=900,
    )


if __name__ == "__main__":
    main()
