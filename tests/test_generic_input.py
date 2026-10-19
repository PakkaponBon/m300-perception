"""Portable sensor-input tests; no ROS imports, nodes, models, or device access."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.lidar_input import InputSettings, decode_xyzi
from tools.dataset.extract_mcap_dataset import extract_points, standard


def cloud_fixture(big=False, intensity_name="reflectivity"):
    endian = ">" if big else "<"
    # Deliberately unordered fields, point padding, row padding, integer intensity.
    dtype = np.dtype(dict(names=["z", "signal", "x", "y"],
                          formats=[endian+"f4", endian+"u2", endian+"f4", endian+"f4"],
                          offsets=[0, 4, 8, 12], itemsize=20))
    payload = bytearray(48 * 2)
    rows = np.ndarray((2, 2), dtype=dtype, buffer=payload, strides=(48, 20))
    rows["x"] = [[1, 2], [3, 4]]
    rows["y"] = 0.5
    rows["z"] = 1.5
    rows["signal"] = [[7, 10], [20, 30]]
    fields = [types.SimpleNamespace(name=name, datatype=kind, offset=offset, count=1)
              for name, kind, offset in [("z", 7, 0), (intensity_name, 4, 4),
                                         ("x", 7, 8), ("y", 7, 12)]]
    return types.SimpleNamespace(fields=fields, is_bigendian=big, point_step=20,
                                 width=2, height=2, row_step=48, data=payload)


class InputTests(unittest.TestCase):
    def test_settings_allow_other_sensor(self):
        settings = InputSettings("/vendor/points", "lidar_front", "reflectivity", 0.1, 0.6)
        self.assertEqual(settings.expected_frame_id, "lidar_front")
        for kwargs in [dict(input_topic=""), dict(expected_frame_id="bad frame"),
                       dict(intensity_field="x"), dict(decay_time_sec=math.nan),
                       dict(score_threshold=1.1)]:
            with self.assertRaises(ValueError):
                InputSettings(**kwargs)

    def test_alias_endian_and_padding_live_equals_extract(self):
        for big in (False, True):
            cloud = cloud_fixture(big)
            live = decode_xyzi(cloud, "reflectivity")
            offline = extract_points(cloud, "reflectivity")
            np.testing.assert_array_equal(live, offline)
            np.testing.assert_array_equal(live[:, 0], [1, 2, 3, 4])
            np.testing.assert_array_equal(live[:, 3], [7, 10, 20, 30])
            self.assertEqual(live.shape, (4, 4))
            self.assertEqual(live.dtype, np.dtype("<f4"))

    def test_no_silent_intensity_fabrication(self):
        with self.assertRaisesRegex(ValueError, "missing required"):
            decode_xyzi(cloud_fixture())
        cloud = cloud_fixture()
        cloud.fields.append(cloud.fields[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            decode_xyzi(cloud, "reflectivity")

    def test_truncated_or_out_of_bounds_field(self):
        cloud = cloud_fixture()
        cloud.data = cloud.data[:-1]
        with self.assertRaisesRegex(ValueError, "Truncated"):
            decode_xyzi(cloud, "reflectivity")
        cloud = cloud_fixture()
        cloud.fields[0].offset = 20
        with self.assertRaisesRegex(ValueError, "outside"):
            decode_xyzi(cloud, "reflectivity")

    def test_default_manifest_hash_unchanged_and_alias_recorded(self):
        default = standard(None, 0.3)
        digest = hashlib.sha256(json.dumps(default, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, "a096d5000fa072852084660f6df6fa2362073f2035f54585baf3664d02867517")
        other = standard(None, 0.3, "lidar_front", "reflectivity")
        self.assertEqual(other["coordinate_frame"], "lidar_front")
        self.assertEqual(other["source_intensity_field"], "reflectivity")
        self.assertNotEqual(default, other)

    def test_origin_and_zone_markers_use_selected_frame(self):
        class Object:
            def __getattr__(self, name):
                value = Object()
                setattr(self, name, value)
                return value

        class Marker(Object):
            CYLINDER, TEXT_VIEW_FACING, LINE_STRIP, ADD = range(4)

        tree = ast.parse((ROOT/"runtime/live_warehouse_detector_ros2.py").read_text())
        names = {"make_lidar_origin_markers", "make_zone_markers", "set_zone_marker_colour"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = dict(Marker=Marker, MarkerArray=lambda **kw: types.SimpleNamespace(**kw),
                         Time=Object, Point=lambda **kw: types.SimpleNamespace(**kw),
                         EXPECTED_FRAME_ID="m300_lidar", math=math, np=np,
                         ZONE_RGB=((255, 60, 60), (255, 210, 0), (80, 255, 120)))
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<marker-test>", "exec"), namespace)
        origin = namespace["make_lidar_origin_markers"]("lidar_front")
        zones = namespace["make_zone_markers"](1, 2, "lidar_front")
        self.assertEqual(len(origin.markers), 2)
        self.assertEqual(len(zones), 6)
        self.assertTrue(all(m.header.frame_id == "lidar_front" for m in origin.markers + zones))


class LauncherTests(unittest.TestCase):
    def launch_stubbed(self, mode, score="0.6"):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            bindir = base/"bin"
            bindir.mkdir()
            setup = base/"setup.bash"
            setup.write_text("# Stub only: do not source actual ROS\ntrue\n")
            node = bindir/"stub-python"
            node.write_text(f"#!{sys.executable}\n" + """
import json, os, pathlib, sys, time
if sys.argv[1] == '-':
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
pathlib.Path(os.environ['STUB_NODE_LOG']).write_text(json.dumps(sys.argv[1:]))
time.sleep(0.4)
""")
            rviz = bindir/"rviz2"
            rviz.write_text(f"#!{sys.executable}\n" + """
import json, os, pathlib, signal, sys, time
pathlib.Path(os.environ['STUB_RVIZ_LOG']).write_text(json.dumps(sys.argv[1:]))
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
time.sleep(20)
""")
            node.chmod(0o755)
            rviz.chmod(0o755)
            checkpoint, config = base/"trusted.pth", base/"model.yaml"
            checkpoint.write_text("stub - never loaded")
            config.write_text("stub - never loaded")
            env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ["PATH"],
                       DISPLAY=":stub-only", PERCEPTION_ROS_SETUP=str(setup),
                       PERCEPTION_WS_SETUP="", PERCEPTION_INPUT_TOPIC="/vendor/points",
                       PERCEPTION_FRAME_ID="lidar_front", PERCEPTION_PYTHON=str(node),
                       PERCEPTION_CHECKPOINT=str(checkpoint), PERCEPTION_MODEL_CONFIG=str(config),
                       PERCEPTION_OPENPCDET_ROOT=str(base),
                       PERCEPTION_SCORE_THRESHOLD=score, PERCEPTION_DECAY_TIME_SEC="0.3",
                       PERCEPTION_INTENSITY_FIELD="reflectivity",
                       STUB_NODE_LOG=str(base/"node.json"), STUB_RVIZ_LOG=str(base/"rviz.json"))
            result = subprocess.run(["bash", str(ROOT/"scripts/start_perception.sh"), mode],
                                    env=env, capture_output=True, text=True, timeout=10)
            if score == "-0.1" and mode == "ai":
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((base/"node.json").exists())
                self.assertFalse((base/"rviz.json").exists())
                return
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            node_args = json.loads((base/"node.json").read_text())
            rviz_args = json.loads((base/"rviz.json").read_text())
            self.assertIn("input_topic:=/vendor/points", node_args)
            self.assertIn("expected_frame_id:=lidar_front", node_args)
            self.assertIn("-f", rviz_args)
            self.assertEqual(rviz_args[rviz_args.index("-f")+1], "lidar_front")
            self.assertIn("/pointcloud:=/vendor/points", rviz_args)
            if mode == "ai":
                self.assertIn("intensity_field:=reflectivity", node_args)
                self.assertIn("score_threshold:=0.6", node_args)

    def test_ai_launcher_without_ros_or_gpu(self):
        self.launch_stubbed("ai")

    def test_obstacle_launcher_without_ros_or_driver(self):
        self.launch_stubbed("obstacle")

    def test_invalid_score_fails_before_children(self):
        self.launch_stubbed("ai", "-0.1")


if __name__ == "__main__":
    unittest.main()
