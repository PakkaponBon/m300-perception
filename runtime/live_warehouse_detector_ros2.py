#!/usr/bin/env python3

import sys
import os
import math
import logging
import time
from pathlib import Path

import numpy as np
import torch
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rcl_interfaces.msg import ParameterDescriptor
from lidar_input import InputSettings, decode_xyzi
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Time
from m300_person_tracker import PersonTracker

# ================================================================
# OpenPCDet paths
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = Path(os.environ.get(
    "PERCEPTION_OPENPCDET_ROOT", os.environ.get(
        "M300_OPENPCDET_ROOT", str(PROJECT_ROOT.parent / "OpenPCDet"))
)).expanduser().resolve()

sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu


CFG_FILE = Path(os.environ.get(
    "PERCEPTION_MODEL_CONFIG",
    str(OPENPCDET_ROOT / "tools/cfgs/custom_models/pointpillar_m300_decay03.yaml")
)).expanduser().resolve()

CKPT_FILE = Path(os.environ.get(
    "PERCEPTION_CHECKPOINT", os.environ.get(
        "M300_CHECKPOINT", str(PROJECT_ROOT / "models" / "model.pth"))
)).expanduser().resolve()

DATA_ROOT = OPENPCDET_ROOT / "data/custom"


# ================================================================
# ROS topics
# ================================================================

POINT_TOPIC = "/pointcloud"
MARKER_TOPIC = "/ai_detections/boxes"
PERSON_POINTS_TOPIC = "/ai_detections/person_points"
BACKGROUND_POINTS_TOPIC = "/ai_detections/background_points"


# ================================================================
# Detection settings
# ================================================================

SCORE_THRESH = 0.3
DECAY_TIME_SEC = 0.3
EXPECTED_FRAME_ID = "m300_lidar"
MAX_BOXES = 10

# Use 1 for smoother detection.
PROCESS_EVERY_N_FRAMES = 1

MIN_POINT_COUNT = 100

# Tracking is display persistence, not a safety guarantee or model improvement.
TRACKING_HOLD_SEC = 0.3
# Refresh uses the same threshold as new detections. HOLD expires after the
# last valid match; it cannot identify chairs or correct model predictions.
TRACKING_KEEP_SCORE = SCORE_THRESH
TRACKING_MATCH_DISTANCE_M = 0.8
TRACKING_MIN_HITS = 2

# Marker automatically disappears if detector stops publishing.
MARKER_LIFETIME_SEC = 0.40

# Visualization only: sensor-relative height slice and one original XYZI row
# per XY cell. These settings do not filter or modify the detector input.
BACKGROUND_CELL_SIZE_M = 0.20
BACKGROUND_MIN_RANGE_M = 0.30
BACKGROUND_MAX_RANGE_M = 20.0
BACKGROUND_MIN_Z_M = -0.20
BACKGROUND_MAX_Z_M = 1.50

# Illustrative display defaults only, NOT validated vehicle safety distances.
# Override at startup with ROS parameters deadzone_radius_m / warning_radius_m.
DEFAULT_DEADZONE_RADIUS_M = 1.0
DEFAULT_WARNING_RADIUS_M = 2.0
ZONE_NAMES = ("DEADZONE", "WARNING", "NORMAL")
ZONE_RGB = ((255, 60, 60), (255, 210, 0), (80, 255, 120))


def get_marker_lifetime():
    return Duration(
        nanoseconds=int(MARKER_LIFETIME_SEC * 1e9)
    ).to_msg()


def detection_zone(box, deadzone_radius, warning_radius):
    """Classify horizontal box-centre distance, not nearest body-point distance."""
    distance = math.hypot(float(box[0]), float(box[1]))
    if distance < deadzone_radius:
        return 0
    if distance < warning_radius:
        return 1
    return 2


def set_zone_marker_colour(marker, zone_index):
    red, green, blue = ZONE_RGB[zone_index]
    marker.color.r = red / 255.0
    marker.color.g = green / 255.0
    marker.color.b = blue / 255.0
    marker.color.a = 1.0


def select_person_points(points, detections, deadzone_radius, warning_radius):
    """Return cropped XYZI rows and packed RGB for each person's centre zone.

    This is box cropping, not semantic segmentation: background points inside
    a person box can remain. Overlapping boxes do not duplicate input rows;
    the highest-danger zone wins when boxes overlap. Intensity is unchanged.
    """
    point_zones = np.full(len(points), 3, dtype=np.uint8)
    for _, class_name, box in detections:
        if class_name.casefold() not in {"person", "pedestrian"}:
            continue
        x, y, z, length, width, height, yaw = map(float, box)
        if min(length, width, height) <= 0:
            continue
        dx = points[:, 0] - x
        dy = points[:, 1] - y
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        inside = (
            (np.abs(local_x) <= length / 2.0)
            & (np.abs(local_y) <= width / 2.0)
            & (np.abs(points[:, 2] - z) <= height / 2.0)
        )
        zone_index = detection_zone(box, deadzone_radius, warning_radius)
        point_zones[inside] = np.minimum(point_zones[inside], zone_index)
    keep = point_zones < 3
    packed_palette = np.array(
        [(red << 16) | (green << 8) | blue for red, green, blue in ZONE_RGB],
        dtype=np.uint32,
    )
    return points[keep], packed_palette[point_zones[keep]]


def select_background_points(points):
    """Sparse top-down context, not wall classification or edge extraction.

    The Z limits are relative to the sensor, not the floor. Keep an original
    point (and its intensity) per XY cell, with no flattening or averaging.
    """
    range_squared = points[:, 0] ** 2 + points[:, 1] ** 2
    keep = (
        (range_squared >= BACKGROUND_MIN_RANGE_M ** 2)
        & (range_squared <= BACKGROUND_MAX_RANGE_M ** 2)
        & (points[:, 2] >= BACKGROUND_MIN_Z_M)
        & (points[:, 2] <= BACKGROUND_MAX_Z_M)
    )
    selected = points[keep]
    if len(selected) == 0:
        return selected
    cells = np.floor(selected[:, :2] / BACKGROUND_CELL_SIZE_M).astype(np.int64)
    _, first_indices = np.unique(cells, axis=0, return_index=True)
    return selected[np.sort(first_indices)]


def make_xyzi_cloud(header, points, packed_rgb=None):
    """Serialize XYZI, optionally with RGB; empty clouds keep the same schema."""
    points = np.ascontiguousarray(points, dtype="<f4").reshape(-1, 4)
    fields = [
        PointField(name=name, offset=4 * index,
                   datatype=PointField.FLOAT32, count=1)
        for index, name in enumerate(("x", "y", "z", "intensity"))
    ]
    point_step = 16
    data = points.tobytes()
    if packed_rgb is not None:
        packed_rgb = np.asarray(packed_rgb, dtype="<u4").reshape(-1)
        if len(packed_rgb) != len(points):
            raise ValueError("Each point must have exactly one packed RGB colour")
        records = np.empty(len(points), dtype=[("xyzi", "<f4", (4,)), ("rgb", "<u4")])
        records["xyzi"] = points
        records["rgb"] = packed_rgb
        fields.append(PointField(name="rgb", offset=16, datatype=PointField.UINT32, count=1))
        point_step = 20
        data = records.tobytes()
    return PointCloud2(
        header=header,
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * len(points),
        data=data,
        is_dense=True,
    )


# ================================================================
# OpenPCDet live dataset
# ================================================================

class LiveDataset(DatasetTemplate):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        logger,
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=DATA_ROOT,
            logger=logger,
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise NotImplementedError

    def prepare_live_points(self, points):
        data_dict = {
            "points": points,
            "frame_id": "live",
        }

        return self.prepare_data(
            data_dict=data_dict
        )


# ================================================================
# Marker creation
# ================================================================

def make_wire_box_marker(
    marker_id,
    frame_id,
    stamp,
    box,
    class_name,
    score,
    zone_index=None,
    held=False,
):
    x, y, z, length, width, height, yaw = [
        float(value) for value in box
    ]

    dx = length / 2.0
    dy = width / 2.0
    dz = height / 2.0

    corners = np.array(
        [
            [dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],

            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
        ],
        dtype=np.float32,
    )

    rotation = np.array(
        [
            [
                math.cos(yaw),
                -math.sin(yaw),
                0.0,
            ],
            [
                math.sin(yaw),
                math.cos(yaw),
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float32,
    )

    corners = (
        corners @ rotation.T
        + np.array(
            [x, y, z],
            dtype=np.float32,
        )
    )

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),

        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),

        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    marker = Marker()

    marker.header.frame_id = frame_id
    marker.header.stamp = Time()

    marker.ns = "ai_3d_boxes"
    marker.id = marker_id

    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.04

    marker.color.r = 1.0
    marker.color.g = 0.75
    marker.color.b = 0.0
    marker.color.a = 1.0

    if zone_index is not None:
        set_zone_marker_colour(marker, zone_index)
    if held:
        marker.color.a = 0.45
    marker.lifetime = get_marker_lifetime()

    for start_index, end_index in edges:
        start_point = Point()

        start_point.x = float(
            corners[start_index, 0]
        )
        start_point.y = float(
            corners[start_index, 1]
        )
        start_point.z = float(
            corners[start_index, 2]
        )

        end_point = Point()

        end_point.x = float(
            corners[end_index, 0]
        )
        end_point.y = float(
            corners[end_index, 1]
        )
        end_point.z = float(
            corners[end_index, 2]
        )

        marker.points.append(start_point)
        marker.points.append(end_point)

    return marker


def make_text_marker(
    marker_id,
    frame_id,
    stamp,
    box,
    class_name,
    score,
    zone_index=None,
    track_id=None,
    hold_age=0.0,
):
    x, y, z, length, width, height, yaw = [
        float(value) for value in box
    ]

    marker = Marker()

    marker.header.frame_id = frame_id
    marker.header.stamp = Time()

    marker.ns = "ai_3d_labels"
    marker.id = marker_id

    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD

    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = (
        z + height / 2.0 + 0.25
    )

    marker.pose.orientation.w = 1.0

    marker.scale.z = 0.175

    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 1.0

    # Horizontal distance from the LiDAR origin to the detected box centre.
    distance_m = math.hypot(x, y)
    marker.text = (
        f"{class_name} {distance_m:.1f} m (score {score:.2f})"
    )
    if track_id is not None:
        marker.text = f"{class_name} #{track_id} {distance_m:.1f} m "
        if hold_age > 0:
            marker.text += f"HOLD {hold_age:.1f}s (last score {score:.2f})"
        else:
            marker.text += f"(score {score:.2f})"
    if zone_index is not None:
        set_zone_marker_colour(marker, zone_index)
        marker.text += f" {ZONE_NAMES[zone_index]}"
    if hold_age > 0:
        marker.color.a = 0.45

    marker.lifetime = get_marker_lifetime()

    return marker


def make_delete_marker(
    marker_id,
    frame_id,
    stamp,
    namespace,
):
    marker = Marker()

    marker.header.frame_id = frame_id
    marker.header.stamp = Time()

    marker.ns = namespace
    marker.id = marker_id

    marker.action = Marker.DELETE

    return marker


# ================================================================
# Detector node
# ================================================================

def make_lidar_origin_markers(frame_id=EXPECTED_FRAME_ID):
    """A schematic sensor icon and label, always at the LiDAR frame origin."""
    markers = []
    for marker_id, marker_type in enumerate((Marker.CYLINDER, Marker.TEXT_VIEW_FACING)):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = Time()
        marker.ns = "lidar_origin"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0
        if marker_type == Marker.CYLINDER:
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.075
        else:
            marker.pose.position.y = -0.3
            marker.pose.position.z = 0.25
            marker.scale.z = 0.1
            marker.text = "LiDAR (0, 0)"
        markers.append(marker)
    return MarkerArray(markers=markers)


def make_zone_markers(deadzone_radius, warning_radius, frame_id=EXPECTED_FRAME_ID):
    """Display three labelled rings, independent of whether anyone is detected.

    The green ring is a visual guide, not a maximum detection range. NORMAL
    classification continues beyond it. No stop commands or safety guarantees.
    """
    # Match the yellow band's width: defaults give rings at 1, 2 and 3 m.
    normal_guide_radius = warning_radius + (warning_radius - deadzone_radius)
    radii = (deadzone_radius, warning_radius, normal_guide_radius)
    labels = (
        f"DEADZONE < {deadzone_radius:g} m",
        f"WARNING {deadzone_radius:g}-{warning_radius:g} m",
        f"NORMAL >= {warning_radius:g} m",
    )
    markers = []
    for zone_index, radius in enumerate(radii):
        ring = Marker()
        ring.header.frame_id = frame_id
        ring.header.stamp = Time()
        ring.ns = "detection_zones"
        ring.id = 2 * zone_index
        ring.type = Marker.LINE_STRIP
        ring.action = Marker.ADD
        ring.pose.orientation.w = 1.0
        ring.scale.x = 0.025
        set_zone_marker_colour(ring, zone_index)
        ring.color.a = 0.65
        ring.points = [
            Point(x=radius * math.cos(angle), y=radius * math.sin(angle), z=-0.1)
            for angle in np.linspace(0.0, 2.0 * math.pi, 97)
        ]
        markers.append(ring)
        text = Marker()
        text.header.frame_id = frame_id
        text.header.stamp = Time()
        text.ns = "detection_zones"
        text.id = 2 * zone_index + 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.orientation.w = 1.0
        text.pose.position.y = -radius - 0.15
        text.scale.z = 0.12
        text.text = labels[zone_index]
        set_zone_marker_colour(text, zone_index)
        markers.append(text)
    return markers


class WarehouseDetectorNode(Node):
    def __init__(self):
        super().__init__(
            "warehouse_3d_detector"
        )
        descriptor = ParameterDescriptor(read_only=True)

        def param(name, default):
            return self.declare_parameter(name, default, descriptor).value

        self.input_settings = InputSettings(
            input_topic=param("input_topic", POINT_TOPIC),
            expected_frame_id=param("expected_frame_id", EXPECTED_FRAME_ID),
            intensity_field=param("intensity_field", "intensity"),
            decay_time_sec=float(param("decay_time_sec", DECAY_TIME_SEC)),
            score_threshold=float(param("score_threshold", SCORE_THRESH)),
        )
        self.config_file = Path(param("model_config", str(CFG_FILE))).expanduser().resolve()
        self.checkpoint_file = Path(param("checkpoint", str(CKPT_FILE))).expanduser().resolve()
        self.deadzone_radius = float(self.declare_parameter(
            "deadzone_radius_m", DEFAULT_DEADZONE_RADIUS_M,
        ).value)
        self.warning_radius = float(self.declare_parameter(
            "warning_radius_m", DEFAULT_WARNING_RADIUS_M,
        ).value)
        if not (
            math.isfinite(self.deadzone_radius)
            and math.isfinite(self.warning_radius)
            and 0.0 < self.deadzone_radius < self.warning_radius
        ):
            raise ValueError("Zone radii must be finite and satisfy 0 < deadzone < warning")

        self.tracking_enabled = self.declare_parameter("tracking_enabled", True).value
        hold_seconds = float(self.declare_parameter("tracking_hold_seconds", TRACKING_HOLD_SEC).value)
        self.tracker = PersonTracker(
            start_score=self.input_settings.score_threshold,
            keep_score=float(self.declare_parameter("tracking_keep_score", self.input_settings.score_threshold).value),
            hold_seconds=hold_seconds,
            match_distance=float(self.declare_parameter("tracking_match_distance_m", TRACKING_MATCH_DISTANCE_M).value),
            min_hits=self.declare_parameter("tracking_min_hits", TRACKING_MIN_HITS).value,
            max_tracks=MAX_BOXES,
        )
        self.tracking_input_timeout = min(hold_seconds, MARKER_LIFETIME_SEC)
        self.last_tracking_update = None

        if not self.config_file.is_file():
            raise FileNotFoundError(
                f"Config file not found: "
                f"{self.config_file}"
            )

        if not self.checkpoint_file.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{self.checkpoint_file}"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available"
            )

        self.publisher = self.create_publisher(
            MarkerArray,
            MARKER_TOPIC,
            10,
        )
        # Independent of detections: show the sensor even with no person/input.
        # Repeat so RViz instances opened later also receive the origin markers.
        self.lidar_origin_timer = self.create_timer(0.5, self.publish_lidar_origin)
        self.person_points_publisher = self.create_publisher(
            PointCloud2, PERSON_POINTS_TOPIC, 1,
        )
        self.background_points_publisher = self.create_publisher(
            PointCloud2, BACKGROUND_POINTS_TOPIC, 1,
        )
        self.last_background_cloud_time = None
        self.last_person_cloud_time = None
        self.person_cloud_timeout_timer = self.create_timer(
            0.1, self.clear_stale_person_cloud,
        )

        self.subscription = (
            self.create_subscription(
                PointCloud2,
                self.input_settings.input_topic,
                self.points_callback,
                qos_profile_sensor_data,
            )
        )

        self.frame_count = 0
        self.busy = False
        self.point_buffer = deque()
        self.decay_time_ns = round(self.input_settings.decay_time_sec * 1_000_000_000)

        # Track IDs stay stable even when prediction score ordering changes.
        self.previous_marker_ids = set()

        logging.basicConfig(
            level=logging.INFO
        )

        self.logger_py = logging.getLogger(
            "warehouse_detector"
        )

        self.get_logger().info(
            f"Loading config: {self.config_file}"
        )

        # OpenPCDet resolves _BASE_CONFIG_ relative to its tools directory.
        previous_directory = Path.cwd()
        try:
            os.chdir(OPENPCDET_ROOT / "tools")
            cfg_from_yaml_file(str(self.config_file), cfg)
        finally:
            os.chdir(previous_directory)

        self.class_names = cfg.CLASS_NAMES

        self.dataset = LiveDataset(
            cfg.DATA_CONFIG,
            self.class_names,
            self.logger_py,
        )

        self.get_logger().info(
            "Building PointPillars model..."
        )

        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(
                self.class_names
            ),
            dataset=self.dataset,
        )

        self.get_logger().info(
            f"Loading checkpoint: "
            f"{self.checkpoint_file}"
        )

        self.model.load_params_from_file(
            filename=str(self.checkpoint_file),
            logger=self.logger_py,
            to_cpu=False,
        )

        self.model.cuda()
        self.model.eval()

        self.get_logger().info(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        self.get_logger().info(
            "Live 3D detector is ready"
        )

        self.get_logger().info(
            f"Point topic: {self.input_settings.input_topic}"
        )
        self.get_logger().info(
            f"Input frame: {self.input_settings.expected_frame_id}; "
            f"intensity field: {self.input_settings.intensity_field}; "
            f"window: {self.input_settings.decay_time_sec:g}s; no automatic transform. "
            "A different sensor may require different training data and weights."
        )

        self.get_logger().info(
            f"Marker topic: {MARKER_TOPIC}"
        )
        self.get_logger().info(
            f"Person-only point cloud: {PERSON_POINTS_TOPIC}"
        )
        self.get_logger().info(
            f"Sparse background point cloud: {BACKGROUND_POINTS_TOPIC}"
        )
        self.get_logger().info(
            f"Display-only zones (box-centre distance): red < {self.deadzone_radius:g} m; "
            f"yellow < {self.warning_radius:g} m; otherwise green. "
            "Not a certified safety system or automatic stop controller."
        )

        self.get_logger().info(
            f"Score threshold: "
            f"{self.input_settings.score_threshold:.2f}"
        )
        self.get_logger().info(
            f"Tracking {'enabled' if self.tracking_enabled else 'disabled'}: "
            f"confirm {self.tracker.min_hits} consecutive detections >= {self.input_settings.score_threshold:.2f}; "
            f"keep >= {self.tracker.keep_score:.2f}; match within {self.tracker.match_distance:g} m; "
            f"hold <= {hold_seconds:g}s. HOLD uses the last box/score, not a new detection. "
            "No ego-motion correction or velocity prediction."
        )

    def publish_lidar_origin(self):
        marker_array = make_lidar_origin_markers(self.input_settings.expected_frame_id)
        marker_array.markers.extend(make_zone_markers(self.deadzone_radius, self.warning_radius,
                                                    self.input_settings.expected_frame_id))
        self.publisher.publish(marker_array)

    def publish_person_points(self, header, points, detections):
        selected, packed_rgb = select_person_points(
            points, detections, self.deadzone_radius, self.warning_radius,
        )
        self.person_points_publisher.publish(make_xyzi_cloud(header, selected, packed_rgb))
        self.last_person_cloud_time = time.monotonic() if len(selected) else None

    def publish_background_points(self, header, points):
        selected = select_background_points(points)
        self.background_points_publisher.publish(make_xyzi_cloud(header, selected))
        self.last_background_cloud_time = time.monotonic() if len(selected) else None

    def clear_stale_person_cloud(self):
        # No sensor/inference updates: stop all tracks, not a repeated HOLD loop.
        if (self.last_tracking_update is not None
                and time.monotonic() - self.last_tracking_update >= self.tracking_input_timeout):
            self.tracker.reset()
            self.last_tracking_update = None
            self.publish_detections(PointCloud2(header=Header(frame_id=self.input_settings.expected_frame_id)), [])
            self.publish_person_points(
                Header(frame_id=self.input_settings.expected_frame_id), np.empty((0, 4), dtype=np.float32), [],
            )
        # Clear both visualization clouds independently if their updates stop.
        if (
            self.last_background_cloud_time is not None
            and time.monotonic() - self.last_background_cloud_time >= MARKER_LIFETIME_SEC
        ):
            self.background_points_publisher.publish(make_xyzi_cloud(
                Header(frame_id=self.input_settings.expected_frame_id), np.empty((0, 4), dtype=np.float32),
            ))
            self.last_background_cloud_time = None
        if self.last_person_cloud_time is None:
            return
        if time.monotonic() - self.last_person_cloud_time < MARKER_LIFETIME_SEC:
            return
        self.person_points_publisher.publish(make_xyzi_cloud(
            Header(frame_id=self.input_settings.expected_frame_id), np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.uint32),
        ))
        self.last_person_cloud_time = None

    def clear_detection_output(self, msg):
        self.tracker.reset()
        self.last_tracking_update = None
        self.publish_detections(msg, [])
        self.publish_person_points(
            msg.header, np.empty((0, 4), dtype=np.float32), [],
        )

    def convert_point_cloud(self, msg):
        points = decode_xyzi(msg, self.input_settings.intensity_field)
        # Same raw XYZI conversion/filter as dataset extraction; no scaling.
        return points[np.isfinite(points).all(axis=1)]

    def publish_detections(
        self,
        msg,
        detections,
        tracks=None,
    ):
        marker_array = MarkerArray()

        if tracks is None:
            entries = [(i, score, name, box, 0.0, None)
                       for i, (score, name, box) in enumerate(detections)]
        else:
            entries = [(t.track_id, t.score, t.class_name, t.box, t.hold_age, t.track_id)
                       for t in tracks]
        current_marker_ids = {entry[0] for entry in entries}

        # Add or update current markers.
        for marker_id, score, class_name, box, hold_age, track_id in entries:
            zone_index = detection_zone(box, self.deadzone_radius, self.warning_radius)

            marker_array.markers.append(
                make_wire_box_marker(
                    marker_id=marker_id,
                    frame_id=msg.header.frame_id,
                    stamp=msg.header.stamp,
                    box=box,
                    class_name=class_name,
                    score=score,
                    zone_index=zone_index,
                    held=hold_age > 0,
                )
            )

            marker_array.markers.append(
                make_text_marker(
                    marker_id=marker_id,
                    frame_id=msg.header.frame_id,
                    stamp=msg.header.stamp,
                    box=box,
                    class_name=class_name,
                    score=score,
                    zone_index=zone_index,
                    track_id=track_id,
                    hold_age=hold_age,
                )
            )
            if tracks is not None:
                # Re-publishing a held box must NOT extend its last-seen TTL.
                remaining = max(1e-9, min(
                    MARKER_LIFETIME_SEC, self.tracker.hold_ns / 1e9 - hold_age,
                ))
                for marker in marker_array.markers[-2:]:
                    marker.lifetime = Duration(
                        nanoseconds=max(1, round(remaining * 1e9)),
                    ).to_msg()

        # Delete markers that existed previously but
        # are not present in the current result.
        for stale_id in sorted(self.previous_marker_ids - current_marker_ids):
            marker_array.markers.append(
                make_delete_marker(
                    marker_id=stale_id,
                    frame_id=msg.header.frame_id,
                    stamp=msg.header.stamp,
                    namespace="ai_3d_boxes",
                )
            )

            marker_array.markers.append(
                make_delete_marker(
                    marker_id=stale_id,
                    frame_id=msg.header.frame_id,
                    stamp=msg.header.stamp,
                    namespace="ai_3d_labels",
                )
            )

        self.previous_marker_ids = current_marker_ids

        self.publisher.publish(
            marker_array
        )

    def accumulate_points(self, msg, current_points):
        if msg.header.frame_id != self.input_settings.expected_frame_id:
            self.point_buffer.clear()
            raise ValueError(f"Expected {self.input_settings.expected_frame_id}, got {msg.header.frame_id!r}")
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp_ns <= 0:
            raise ValueError("PointCloud header timestamp must be positive")
        if self.point_buffer:
            last_stamp = self.point_buffer[-1][0]
            if stamp_ns == last_stamp:
                raise ValueError("Duplicate PointCloud timestamp; frame ignored")
            if stamp_ns < last_stamp:
                # Bag replay/clock reset: never combine different time sequences.
                self.point_buffer.clear()
                self.tracker.reset()
                self.last_tracking_update = None
                self.get_logger().warn("PointCloud time moved backwards; cleared accumulation")

        self.point_buffer.append((stamp_ns, current_points))
        # Inclusive [latest - decay, latest], matching dataset extraction.
        while self.point_buffer[0][0] < stamp_ns - self.decay_time_ns:
            self.point_buffer.popleft()

        # No motion correction, averaging or voxelization here; model handles voxels.
        return np.concatenate(
            [points for _, points in self.point_buffer],
            axis=0,
        )

    def points_callback(self, msg):
        if self.busy:
            return

        self.busy = True

        try:
            current_points = self.convert_point_cloud(
                msg
            )

            points = self.accumulate_points(
                msg,
                current_points,
            )
            # Current raw frame only: do not accumulate background into a map.
            self.publish_background_points(msg.header, current_points)

            # Accumulate every received frame; stride only selects inference outputs.
            index = self.frame_count
            self.frame_count += 1
            if index % PROCESS_EVERY_N_FRAMES != 0:
                return

            if len(points) < MIN_POINT_COUNT:
                self.get_logger().warn(
                    f"Too few points: "
                    f"{len(points)}"
                )

                self.clear_detection_output(msg)
                return

            data_dict = (
                self.dataset.prepare_live_points(
                    points
                )
            )

            batch_dict = (
                self.dataset.collate_batch(
                    [data_dict]
                )
            )

            load_data_to_gpu(
                batch_dict
            )

            with torch.inference_mode():
                prediction_dicts, _ = (
                    self.model(batch_dict)
                )

            prediction = prediction_dicts[0]

            predicted_boxes = (
                prediction["pred_boxes"]
                .detach()
                .cpu()
                .numpy()
            )

            predicted_scores = (
                prediction["pred_scores"]
                .detach()
                .cpu()
                .numpy()
            )

            predicted_labels = (
                prediction["pred_labels"]
                .detach()
                .cpu()
                .numpy()
            )

            detections = []

            for box, score, label_id in zip(
                predicted_boxes,
                predicted_scores,
                predicted_labels,
            ):
                score = float(score)

                threshold = self.tracker.keep_score if self.tracking_enabled else self.input_settings.score_threshold
                if not math.isfinite(score) or score < threshold:
                    continue

                if not np.all(
                    np.isfinite(box)
                ):
                    continue

                class_index = (
                    int(label_id) - 1
                )

                if not (
                    0
                    <= class_index
                    < len(self.class_names)
                ):
                    continue

                class_name = (
                    self.class_names[
                        class_index
                    ]
                )

                detections.append(
                    (
                        score,
                        class_name,
                        box,
                    )
                )

            detections.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            detections = detections[
                :MAX_BOXES
            ]

            if self.tracking_enabled:
                stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                tracks = self.tracker.update(detections, stamp_ns)
                self.last_tracking_update = time.monotonic()
                # Real measured boxes only. HOLD never fabricates/replays points.
                detections = [(t.score, t.class_name, t.box) for t in tracks if t.matched]
                self.publish_detections(msg, detections, tracks=tracks)
            else:
                self.publish_detections(msg, detections)
            # Crop the same 0.3 s accumulated cloud used for this prediction.
            # Keep original XYZ/intensity; only the RViz camera is top-down.
            self.publish_person_points(msg.header, points, detections)

            if detections:
                score_text = ", ".join(
                    (
                        f"{class_name}:"
                        f"{score:.2f}"
                    )
                    for (
                        score,
                        class_name,
                        box,
                    ) in detections
                )

                self.get_logger().debug(
                    f"Detected "
                    f"{len(detections)} objects "
                    f"from {len(points)} points"
                    f" | {score_text}"
                )

            else:
                self.get_logger().debug(
                    f"Detected 0 objects "
                    f"from {len(points)} points"
                )

        except Exception as exception:
            self.get_logger().error(
                f"Inference failed: "
                f"{exception!r}"
            )
            self.clear_detection_output(msg)

        finally:
            self.busy = False

    def clear_all_markers(self):
        self.tracker.reset()
        self.last_tracking_update = None
        self.background_points_publisher.publish(make_xyzi_cloud(
            Header(frame_id=self.input_settings.expected_frame_id), np.empty((0, 4), dtype=np.float32),
        ))
        self.last_background_cloud_time = None
        self.person_points_publisher.publish(make_xyzi_cloud(
            Header(frame_id=self.input_settings.expected_frame_id), np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.uint32),
        ))
        self.last_person_cloud_time = None
        marker_array = MarkerArray()
        for marker_id in range(6):
            marker_array.markers.append(make_delete_marker(
                marker_id=marker_id,
                frame_id=self.input_settings.expected_frame_id,
                stamp=Time(),
                namespace="detection_zones",
            ))
        for marker_id in (0, 1):
            marker_array.markers.append(make_delete_marker(
                marker_id=marker_id,
                frame_id=self.input_settings.expected_frame_id,
                stamp=Time(),
                namespace="lidar_origin",
            ))

        for marker_id in sorted(self.previous_marker_ids):
            marker_array.markers.append(
                make_delete_marker(
                    marker_id=marker_id,
                    frame_id=self.input_settings.expected_frame_id,
                    stamp=self.get_clock()
                    .now()
                    .to_msg(),
                    namespace="ai_3d_boxes",
                )
            )

            marker_array.markers.append(
                make_delete_marker(
                    marker_id=marker_id,
                    frame_id=self.input_settings.expected_frame_id,
                    stamp=self.get_clock()
                    .now()
                    .to_msg(),
                    namespace="ai_3d_labels",
                )
            )

        self.publisher.publish(
            marker_array
        )
        self.previous_marker_ids.clear()


# ================================================================
# Main
# ================================================================

def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = WarehouseDetectorNode()

        rclpy.spin(node)

    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    finally:
        if node is not None:
            if rclpy.ok():
                node.clear_all_markers()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
