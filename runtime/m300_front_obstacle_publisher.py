#!/usr/bin/env python3
"""Class-agnostic, latest-frame front-region occupancy; no AI or tracking.

Defaults: +X forward, 4 m forward depth, 60 degree full opening angle,
0.3 m near cutoff, sensor-relative Z [-0.2, 1.5] m, at least 5 points.
The region is a triangular prism with its tip clipped by the near cutoff,
NOT a circular sector. No TF transform, accumulation or ground estimation.

Outputs under /front_obstacle:
  state: CLEAR / OCCUPIED / NO_DATA / INVALID_FRAME / INVALID_CLOUD
  data_valid: whether the most recent input is valid and not timed out
  detected: Bool emitted ONLY for valid input frames; do not use alone
  point_count: valid points in the region (0 for invalid/no-data states)
  zone: RED / YELLOW / GREEN, or UNKNOWN for invalid/no-data
        GREEN includes CLEAR; state still distinguishes occupied from clear
  zone_distance_m: nearest point's forward-axis depth (inf clear, NaN invalid)
  distance_m: nearest horizontal point distance when occupied, inf when
              clear, NaN when invalid/no-data (not box-centre distance)
  points: original XYZ points in the region only when occupied
  markers: region coloured by the nearest forward point's zone, forward arrow,
           state and measurement text; grey when invalid/no-data

Always check state/data_valid; silence or NO_DATA does not mean clear.
This is an informational prototype, NOT a certified protective device.
Parameters are startup-only. Keep the existing LiDAR driver running.
"""
import math
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, Float32, Header, String, UInt32
from visualization_msgs.msg import Marker, MarkerArray


# EDIT THESE ZONE VALUES (metres), then restart the publisher.
# Distances are depths along the selected FRONT axis, not radial XY distances.
# RED: near_m..RED_ZONE_MAX_M; YELLOW: red..yellow; GREEN: yellow..green.
# GREEN_ZONE_MAX_M sets the entire ROI's default forward depth. A green
# occupied frame is NOT clear space. All these are display/information values.
RED_ZONE_MAX_M = 2.0
YELLOW_ZONE_MAX_M = 3.0
GREEN_ZONE_MAX_M = 4.0

ZONE_RGB = {
    'RED': (1.0, 0.15, 0.15),
    'YELLOW': (1.0, 0.8, 0.1),
    'GREEN': (0.2, 1.0, 0.3),
    'UNKNOWN': (0.5, 0.5, 0.5),
}


def zone_for_distance(forward_distance, red_max, yellow_max):
    """Inclusive maxima: exactly red_max is red, yellow_max is yellow."""
    if not math.isfinite(forward_distance) or forward_distance < 0:
        return 'UNKNOWN'
    if forward_distance <= red_max:
        return 'RED'
    if forward_distance <= yellow_max:
        return 'YELLOW'
    return 'GREEN'


@dataclass(frozen=True)
class FrontRegion:
    distance_m: float = GREEN_ZONE_MAX_M
    full_angle_deg: float = 60.0
    forward_yaw_deg: float = 0.0
    near_m: float = 0.3
    min_z_m: float = -0.2
    max_z_m: float = 1.5

    def __post_init__(self):
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError('Region parameters must be finite')
        if not (0 <= self.near_m < self.distance_m and
                0 < self.full_angle_deg < 180 and self.min_z_m < self.max_z_m):
            raise ValueError('Require 0 <= near < distance, 0 < angle < 180, min_z < max_z')
        if not math.isfinite(self.distance_m * self.slope):
            raise ValueError('Region width overflows')

    @property
    def slope(self):
        return math.tan(math.radians(self.full_angle_deg / 2))

    def world_point(self, forward, lateral, z):
        angle = math.radians(self.forward_yaw_deg)
        c, s = math.cos(angle), math.sin(angle)
        return forward * c - lateral * s, forward * s + lateral * c, z

    def forward_distances(self, xyz):
        angle = math.radians(self.forward_yaw_deg)
        xy = xyz[:, :2].astype(np.float64, copy=False)
        return xy[:, 0] * math.cos(angle) + xy[:, 1] * math.sin(angle)

    def select(self, xyz):
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        xy = xyz[:, :2].astype(np.float64, copy=False)
        angle = math.radians(self.forward_yaw_deg)
        c, s = math.cos(angle), math.sin(angle)
        forward = xy[:, 0] * c + xy[:, 1] * s
        lateral = -xy[:, 0] * s + xy[:, 1] * c
        mask = ((forward >= self.near_m) & (forward <= self.distance_m) &
                (np.abs(lateral) <= forward * self.slope) &
                (xyz[:, 2] >= self.min_z_m) & (xyz[:, 2] <= self.max_z_m))
        return xyz[mask]

    def corners(self):
        n, f, slope = self.near_m, self.distance_m, self.slope
        base = [(n, -n * slope), (f, -f * slope),
                (f, f * slope), (n, n * slope)]
        return [self.world_point(u, v, z)
                for z in (self.min_z_m, self.max_z_m) for u, v in base]


def extract_xyz(msg):
    """Decode XYZ respecting endian, field offsets and point/row padding."""
    scalar_types = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
                    5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    names = ['x', 'y', 'z']
    fields = {}
    for field in msg.fields:
        if field.name in names:
            if field.name in fields:
                raise ValueError('Duplicate XYZ field')
            fields[field.name] = field
    if set(fields) != set(names):
        raise ValueError('Missing XYZ fields')
    endian = '>' if msg.is_bigendian else '<'
    formats, offsets = [], []
    for name in names:
        field = fields[name]
        if field.count != 1 or field.datatype not in scalar_types:
            raise ValueError('Unsupported XYZ field')
        fmt = endian + scalar_types[field.datatype]
        if field.offset < 0 or field.offset + np.dtype(fmt).itemsize > msg.point_step:
            raise ValueError('XYZ field outside point_step')
        formats.append(fmt)
        offsets.append(field.offset)
    if not msg.width or not msg.height:
        return np.empty((0, 3), dtype='<f4')
    if msg.row_step < msg.width * msg.point_step or len(msg.data) < msg.row_step * msg.height:
        raise ValueError('Truncated cloud or invalid row_step')
    dtype = np.dtype(dict(names=names, formats=formats, offsets=offsets,
                         itemsize=msg.point_step))
    rows = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.row_step, msg.point_step))
    with np.errstate(over='ignore', invalid='ignore'):
        xyz = np.column_stack([rows[n].reshape(-1) for n in names]).astype('<f4')
    return xyz[np.isfinite(xyz).all(axis=1) & np.any(xyz != 0, axis=1)]


def xyz_cloud(header, xyz):
    return PointCloud2(
        header=header, height=1, width=len(xyz), is_bigendian=False,
        fields=[PointField(name=name, offset=index * 4,
                           datatype=PointField.FLOAT32, count=1)
                for index, name in enumerate(('x', 'y', 'z'))],
        point_step=12, row_step=12 * len(xyz), is_dense=True,
        data=np.asarray(xyz, dtype='<f4').tobytes())


class FrontObstaclePublisher(Node):
    def __init__(self):
        super().__init__('m300_front_obstacle')
        descriptor = ParameterDescriptor(read_only=True)

        def param(name, default):
            return self.declare_parameter(name, default, descriptor).value

        self.region = FrontRegion(**{name: float(param(name, default))
                                     for name, default in vars(FrontRegion()).items()})
        self.red_zone_max = float(param('red_zone_max_m', RED_ZONE_MAX_M))
        self.yellow_zone_max = float(param('yellow_zone_max_m', YELLOW_ZONE_MAX_M))
        if not (math.isfinite(self.red_zone_max) and math.isfinite(self.yellow_zone_max) and
                self.region.near_m < self.red_zone_max < self.yellow_zone_max < self.region.distance_m):
            raise ValueError('Require near_m < red_zone_max_m < yellow_zone_max_m < distance_m (green extent)')
        self.min_points = int(param('min_points', 5))
        self.timeout = float(param('input_timeout_sec', 0.5))
        self.frame_id = str(param('expected_frame_id', 'm300_lidar'))
        prefix = str(param('output_prefix', '/front_obstacle')).rstrip('/')
        if not (1 <= self.min_points <= 4294967295 and
                math.isfinite(self.timeout) and self.timeout >= 0.1):
            raise ValueError('min_points must be positive; input_timeout_sec must be >= 0.1')
        if not prefix or not self.frame_id:
            raise ValueError('output_prefix and expected_frame_id must be nonempty')
        self.valid = False
        self.occupied = False
        self.state = 'NO_DATA'
        self.count = 0
        self.distance = math.nan
        self.zone_distance = math.nan
        self.zone = 'UNKNOWN'
        self.last_arrival = None
        self.points = np.empty((0, 3), dtype='<f4')
        self.header = Header(frame_id=self.frame_id)
        self.publishers_by_name = {}
        for name, message_type in [('state', String), ('data_valid', Bool),
                                   ('detected', Bool), ('point_count', UInt32),
                                   ('zone', String), ('zone_distance_m', Float32),
                                   ('distance_m', Float32), ('points', PointCloud2),
                                   ('markers', MarkerArray)]:
            self.publishers_by_name[name] = self.create_publisher(message_type, f'{prefix}/{name}', 1)
        input_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                               durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(
            PointCloud2, str(param('input_topic', '/pointcloud')), self.on_cloud, input_qos)
        # Do not freeze the timeout when a bag/simulation pauses its ROS clock.
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.watchdog = self.create_timer(0.1, self.on_timer, clock=self.steady_clock)
        self.publish_outputs(emit_detection=False, emit_points=True)
        self.get_logger().info(
            f'Front ROI: yaw={self.region.forward_yaw_deg:g} deg (+X at 0), '
            f'forward={self.region.near_m:g}..{self.region.distance_m:g} m, '
            f'full angle={self.region.full_angle_deg:g} deg, '
            f'sensor Z={self.region.min_z_m:g}..{self.region.max_z_m:g} m, '
            f'zones (forward depth): RED <= {self.red_zone_max:g} m, '
            f'YELLOW <= {self.yellow_zone_max:g} m, GREEN to {self.region.distance_m:g} m; '
            f'min_points={self.min_points}; latest frame only, NO AI. '
            'NO_DATA/invalid does not mean clear. Not a certified safety device.')

    def invalidate(self, state):
        self.valid = self.occupied = False
        self.state = state
        self.count = 0
        self.distance = math.nan
        self.zone_distance = math.nan
        self.zone = 'UNKNOWN'
        self.points = np.empty((0, 3), dtype='<f4')
        self.header = Header(frame_id=self.frame_id)

    def on_cloud(self, msg):
        self.last_arrival = time.monotonic()
        previous_state = self.state
        if msg.header.frame_id != self.frame_id:
            self.invalidate('INVALID_FRAME')
            if previous_state != self.state:
                self.get_logger().warning(f'Expected {self.frame_id!r}; got {msg.header.frame_id!r}; no TF transform')
        else:
            try:
                xyz = extract_xyz(msg)
                if len(xyz) == 0:
                    raise ValueError('No finite, nonzero XYZ returns; cannot establish clear space')
                selected = self.region.select(xyz)
                self.count = len(selected)
                self.valid = True
                self.occupied = self.count >= self.min_points
                self.state = 'OCCUPIED' if self.occupied else 'CLEAR'
                self.header = msg.header
                self.points = selected if self.occupied else np.empty((0, 3), dtype='<f4')
                self.distance = float(np.hypot(selected[:, 0], selected[:, 1]).min()) if self.occupied else math.inf
                self.zone_distance = float(self.region.forward_distances(selected).min()) if self.occupied else math.inf
                self.zone = (zone_for_distance(self.zone_distance, self.red_zone_max, self.yellow_zone_max)
                             if self.occupied else 'GREEN')
            except (ValueError, TypeError, OverflowError) as error:
                self.invalidate('INVALID_CLOUD')
                if previous_state != self.state:
                    self.get_logger().warning(str(error))
        self.publish_outputs(emit_detection=self.valid, emit_points=True)

    def on_timer(self):
        timed_out = (self.last_arrival is None or
                     time.monotonic() - self.last_arrival >= self.timeout)
        became_stale = timed_out and self.state != 'NO_DATA'
        if timed_out:
            self.invalidate('NO_DATA')
        self.publish_outputs(emit_detection=False, emit_points=became_stale)

    def publish_outputs(self, emit_detection, emit_points):
        pub = self.publishers_by_name
        pub['state'].publish(String(data=self.state))
        pub['data_valid'].publish(Bool(data=self.valid))
        pub['point_count'].publish(UInt32(data=self.count))
        pub['distance_m'].publish(Float32(data=self.distance))
        pub['zone'].publish(String(data=self.zone))
        pub['zone_distance_m'].publish(Float32(data=self.zone_distance))
        if emit_detection:
            pub['detected'].publish(Bool(data=self.occupied))
        if emit_points:
            pub['points'].publish(xyz_cloud(self.header, self.points))
        pub['markers'].publish(self.make_markers())

    def make_markers(self):
        def marker(identifier, kind):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.ns = 'front_obstacle_roi'
            m.id, m.type, m.action = identifier, kind, Marker.ADD
            m.pose.orientation.w = 1.0
            m.lifetime = Duration(sec=0, nanosec=400000000)
            m.color.r, m.color.g, m.color.b = ZONE_RGB[self.zone]
            m.color.a = 0.8
            return m

        corners = self.region.corners()
        wire = marker(0, Marker.LINE_LIST)
        wire.scale.x = 0.03
        edges = [(i, (i + 1) % 4) for i in range(4)]
        edges += [(i + 4, (i + 1) % 4 + 4) for i in range(4)]
        edges += [(i, i + 4) for i in range(4)]
        wire.points = [Point(x=float(corners[i][0]), y=float(corners[i][1]), z=float(corners[i][2]))
                       for edge in edges for i in edge]
        arrow = marker(1, Marker.ARROW)
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.035, 0.08, 0.10
        arrow.color.r, arrow.color.g, arrow.color.b = 0.1, 0.8, 1.0
        x, y, z = self.region.world_point(min(0.6, self.region.distance_m), 0, 0)
        arrow.points = [Point(), Point(x=float(x), y=float(y), z=float(z))]
        text = marker(2, Marker.TEXT_VIEW_FACING)
        text.scale.z = 0.16
        x, y, z = self.region.world_point(self.region.distance_m, 0, self.region.max_z_m + 0.15)
        text.pose.position = Point(x=float(x), y=float(y), z=float(z))
        text.text = f'{self.zone} | {self.state} | {self.count} points'
        if self.occupied:
            text.text += f' | nearest {self.distance:.2f} m'
            text.text += f' | forward {self.zone_distance:.2f} m'
        text.text += f'\nFRONT {self.region.forward_yaw_deg:g} deg | {self.region.distance_m:g} m / {self.region.full_angle_deg:g} deg'
        text.text += f' | R<={self.red_zone_max:g} Y<={self.yellow_zone_max:g} G<={self.region.distance_m:g}'
        return MarkerArray(markers=[wire, arrow, text])


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FrontObstaclePublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
