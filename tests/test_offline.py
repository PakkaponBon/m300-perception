"""Hardware-free regression tests. All fixture data is created in temp dirs."""
import ast
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from dataclasses import dataclass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


extract = module_at("m300_extract_test", "tools/dataset/extract_mcap_dataset.py")
prepare = module_at("m300_prepare_test", "tools/dataset/prepare_m300_dataset.py")
tracker = module_at("m300_tracker_test", "runtime/m300_person_tracker.py")


def geometry_only():
    # Compile only pure geometry definitions; never import ROS or create a node.
    tree = ast.parse((ROOT / "runtime/m300_front_obstacle_publisher.py").read_text())
    wanted = {"RED_ZONE_MAX_M", "YELLOW_ZONE_MAX_M", "GREEN_ZONE_MAX_M",
              "ZONE_RGB", "zone_for_distance", "FrontRegion"}
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            nodes.append(node)
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in wanted for target in node.targets):
            nodes.append(node)
    module = types.ModuleType("m300_geometry_test")
    sys.modules[module.__name__] = module
    module.__dict__.update(np=np, math=math, dataclass=dataclass)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<geometry-only>", "exec"),
         module.__dict__)
    return module


geometry = geometry_only()


def run_tool(name, *args, check=True):
    return subprocess.run([sys.executable, str(ROOT / "tools/dataset" / name),
                           *map(str, args)], capture_output=True, text=True, check=check)


def source_fixture(base, name, offset, count=12):
    source = base / name
    (source / "points").mkdir(parents=True)
    standard = extract.standard(None, 0.3)
    manifest = dict(status="complete", bag=str(base / ("bag_" + name)),
                    features=["x", "y", "z", "intensity"], dtype="<f4",
                    standard=standard, accumulation_seconds=0.3, stride=5)
    (source / "manifest.json").write_text(json.dumps(manifest))
    rows = []
    for i in range(count):
        sid = f"{i * 5:06d}"
        points = np.array([[offset + i + 1, 0.1, 0.5, 7],
                           [offset + i + 1, 0.2, 0.6, 9]], dtype="<f4")
        points.tofile(source / "points" / (sid + ".bin"))
        stamp = (offset + i * 3 + 10) * 1_000_000_000
        rows.append(dict(sample_id=sid, header_timestamp_ns=stamp,
                         window_start_ns=stamp - 200_000_000, saved_points=2))
    with (source / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return source


def sly_fixture(base, sources):
    export = base / "export"
    ann_dir = export / "ds0/ann"
    cloud_dir = export / "ds0/pointcloud"
    ann_dir.mkdir(parents=True)
    cloud_dir.mkdir()
    for index, source in enumerate(sources):
        for i, binary in enumerate(sorted((source / "points").glob("*.bin"))):
            sid = f"{binary.stem}_{index:02d}"
            payload = binary.read_bytes()
            header = ("VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
                      "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\n"
                      "POINTS 2\nDATA binary\n")
            (cloud_dir / (sid + ".pcd")).write_bytes(header.encode() + payload)
            annotation = dict(objects=[dict(key="p", classTitle="person")], figures=[])
            if i:
                annotation["figures"] = [dict(objectKey="p", geometryType="cuboid_3d",
                    geometry=dict(position=dict(x=1, y=0, z=0.5),
                                  dimensions=dict(x=0.6, y=0.7, z=1.7),
                                  rotation=dict(x=0, y=0, z=0.2)))]
            (ann_dir / (sid + ".pcd.json")).write_text(json.dumps(annotation))
    return export


class TemporalTests(unittest.TestCase):
    def test_concatenate_not_add_or_average(self):
        buffer = extract.TemporalBuffer(0.3)
        a = np.array([[1, 2, 3, 4]], dtype="<f4")
        b = np.array([[1, 2, 3, 10]], dtype="<f4")
        buffer.append(1_000_000_000, a, 1, 0, 0)
        buffer.append(1_300_000_000, b, 1, 0, 0)
        points, counts = buffer.merged()
        np.testing.assert_array_equal(points, np.vstack([a, b]))
        self.assertEqual(points.shape, (2, 4))
        self.assertEqual(counts, [2, 0, 0])
        buffer.append(1_300_000_001, b, 1, 0, 0)
        self.assertEqual(len(buffer.frames), 2)

    def test_zero_decay_and_nonincreasing_stamp(self):
        buffer = extract.TemporalBuffer(0)
        a = np.zeros((2, 4), dtype="<f4")
        buffer.append(10, a, 2, 0, 0)
        buffer.append(11, a, 2, 0, 0)
        self.assertEqual(len(buffer.frames), 1)
        with self.assertRaises(ValueError):
            buffer.append(11, a, 2, 0, 0)


class TrackingTests(unittest.TestCase):
    def test_confirmation_hold_expiry_and_score_gate(self):
        track = tracker.PersonTracker(start_score=0.6, keep_score=0.6)
        box = np.array([1, 0, 0, 0.6, 0.7, 1.7, 0])
        detection = [(0.8, "person", box)]
        self.assertEqual(track.update(detection, 1_000_000_000), [])
        confirmed = track.update(detection, 1_100_000_000)
        self.assertEqual(len(confirmed), 1)
        held = track.update([(0.5, "person", box)], 1_200_000_000)
        self.assertEqual(held[0].track_id, confirmed[0].track_id)
        self.assertFalse(held[0].matched)
        self.assertEqual(track.update([], 1_400_000_001), [])

    def test_one_frame_flash_not_confirmed(self):
        track = tracker.PersonTracker()
        box = [1, 0, 0, 0.6, 0.7, 1.7, 0]
        self.assertEqual(track.update([(0.9, "person", box)], 10), [])
        self.assertEqual(track.update([], 20), [])
        self.assertEqual(track.tracks, {})


class GeometryTests(unittest.TestCase):
    def test_depth_slope_height_and_current_defaults(self):
        region = geometry.FrontRegion()
        self.assertEqual(region.distance_m, 4)
        xyz = np.array([[0.3, 0, 0], [4, 0, 0], [4.01, 0, 0],
                        [1, 1, 0], [1, 0, 1.6], [np.nan, 0, 0]])
        np.testing.assert_array_equal(region.select(xyz), xyz[:2])
        rotated = geometry.FrontRegion(forward_yaw_deg=90)
        np.testing.assert_allclose(rotated.select(np.array([[0, 1, 0], [1, 0, 0]])),
                                   [[0, 1, 0]])

    def test_inclusive_zone_boundaries(self):
        self.assertEqual(geometry.zone_for_distance(2, 2, 3), "RED")
        self.assertEqual(geometry.zone_for_distance(3, 2, 3), "YELLOW")
        self.assertEqual(geometry.zone_for_distance(3.1, 2, 3), "GREEN")
        self.assertEqual(geometry.zone_for_distance(math.nan, 2, 3), "UNKNOWN")

    def test_invalid_geometry(self):
        for kwargs in [dict(distance_m=0), dict(full_angle_deg=180),
                       dict(min_z_m=2, max_z_m=1)]:
            with self.assertRaises(ValueError):
                geometry.FrontRegion(**kwargs)


class DatasetTests(unittest.TestCase):
    def test_pack_multiple_sources_preserves_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            a = source_fixture(base, "a", 0)
            b = source_fixture(base, "b", 100)
            output = base / "samples.zip"
            run_tool("pack_cvat_dataset.py", "--input", a, "--input", b,
                     "--sample-every", 2, "--output", output)
            mapping = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(mapping["samples"], 12)
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(len(set(archive.namelist())), 12)
                for item in mapping["files"]:
                    self.assertEqual(archive.read(item["archive_name"]),
                                     Path(item["source_file"]).read_bytes())
            self.assertTrue(all(s["effective_source_stride"] == 10 for s in mapping["sources"]))
            result = run_tool("pack_cvat_dataset.py", "--input", a, "--output", output, check=False)
            self.assertNotEqual(result.returncode, 0)

    def test_prepare_then_infos_and_empty_negative(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            sources = [source_fixture(base, "a", 0), source_fixture(base, "b", 100)]
            export = sly_fixture(base, sources)
            output = base / "prepared"
            args = ["--export", export, "--source", sources[0], "--source", sources[1],
                    "--output", output]
            run_tool("prepare_m300_dataset.py", *args)
            self.assertFalse(output.exists())
            run_tool("prepare_m300_dataset.py", *args, "--write")
            report = json.loads((output / "preparation_report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["splits"]["train"]["negatives"], 2)
            self.assertEqual((output / "labels/src00_000000.txt").read_text(), "")
            self.assertEqual(np.load(output / "points/src00_000000.npy").shape, (2, 4))
            split_ids = [set((output / "ImageSets" / (s + ".txt")).read_text().split())
                         for s in ("train", "val", "test")]
            self.assertFalse(split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2]
                             or split_ids[1] & split_ids[2])
            run_tool("build_m300_infos.py", "--data", output)
            for split in ("train", "val", "test"):
                self.assertTrue((output / f"custom_infos_{split}.pkl").is_file())

    def test_split_gap_and_empty_box_shape(self):
        records = [dict(start_ns=2_000_000_000 + i * 500_000_000,
                        end_ns=2_300_000_000 + i * 500_000_000) for i in range(50)]
        prepare.split_records(records, [0.7, 0.15, 0.15], 1_000_000_000)
        last_end = None
        for split in ("train", "val", "test"):
            rows = [r for r in records if r["split"] == split]
            self.assertTrue(rows)
            if last_end is not None:
                self.assertGreater(min(r["start_ns"] for r in rows), last_end + 1_000_000_000)
            last_end = max(r["end_ns"] for r in rows)
        self.assertTrue(any(r["split"] == "excluded_gap" for r in records))
        with tempfile.TemporaryDirectory() as temp:
            ann = Path(temp) / "negative.json"
            ann.write_text(json.dumps(dict(objects=[], figures=[])))
            self.assertEqual(prepare.read_boxes(ann).shape, (0, 7))


if __name__ == "__main__":
    unittest.main()
