#!/usr/bin/env python3
"""Offline MCAP -> XYZI samples with optional temporal accumulation.

Output: points/*.bin (little-endian float32 Nx4), frames.csv, manifest.json.
Coordinates and intensity retain sensor units. No motion correction or voxelization.
Labels and OpenPCDet dataset metadata must be prepared separately.
"""
import argparse
import csv
import json
import hashlib
import sys
from collections import deque
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.lidar_input import InputSettings, decode_xyzi


def standard(bounds, decay_time=0, frame_id='m300_lidar', intensity_field='intensity'):
    profile = dict(version='m300-xyzi-single-v1', features=['x', 'y', 'z', 'intensity'],
                dtype='<f4', coordinate_frame=frame_id, length_unit='metre',
                coordinates='unchanged sensor coordinates; no transform',
                intensity='raw, unnormalized; zero retained',
                nonfinite_policy='drop point if any XYZI value is nonfinite',
                zero_xyz_policy='retain', bounds=bounds,
                bounds_rule='inclusive XYZ box in sensor frame',
                accumulation_seconds=0, motion_compensation=False,
                voxelization=False, ground_removal=False, robot_removal=False)
    if decay_time > 0:
        profile.update(version='m300-xyzi-temporal-v1',
                       accumulation_seconds=decay_time,
                       window='inclusive [latest_header_stamp - decay, latest_header_stamp]',
                       warmup='emit partial windows',
                       timestamp_basis='message header; not per-point timestamp',
                       motion_policy='uncompensated sensor-frame concatenation')
    # Preserve historical M300 profile/hash for unchanged defaults.
    if frame_id != 'm300_lidar' or intensity_field != 'intensity':
        profile['version'] = ('lidar-xyzi-temporal-v1' if decay_time > 0
                              else 'lidar-xyzi-single-v1')
        profile['source_intensity_field'] = intensity_field
    return profile


class TemporalBuffer:
    """Keep all source frames within an inclusive integer-nanosecond window."""
    def __init__(self, decay_time):
        self.duration_ns = round(decay_time * 1_000_000_000)
        self.frames = deque()

    def append(self, stamp, points, raw, invalid, cropped):
        if self.frames and stamp <= self.frames[-1][0]:
            raise ValueError('Temporal buffer requires increasing timestamps')
        self.frames.append((stamp, points, raw, invalid, cropped))
        while self.frames[0][0] < stamp - self.duration_ns:
            self.frames.popleft()

    def merged(self):
        points = np.concatenate([f[1] for f in self.frames], axis=0)
        counts = [sum(f[i] for f in self.frames) for i in (2, 3, 4)]
        return points, counts


def extract_points(msg, intensity_field='intensity'):
    """Use the same scalar XYZI decoder as live inference."""
    return decode_xyzi(msg, intensity_field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='New directory; existing output is never overwritten')
    parser.add_argument('--topic', default='/pointcloud')
    parser.add_argument('--frame-id', default='m300_lidar',
                        help='Require this exact source frame; no automatic TF conversion')
    parser.add_argument('--intensity-field', default='intensity',
                        help='Scalar intensity field name (e.g. reflectivity); no normalization')
    parser.add_argument('--decay-time', type=float, default=0.0,
                        help='Trailing header-time window in seconds; default 0 = one frame')
    parser.add_argument('--allow-uncompensated', action='store_true',
                        help='Accept accumulation without ego-motion compensation')
    parser.add_argument('--reference-manifest', type=Path,
                        help='Require identical preprocessing standard to this manifest')
    parser.add_argument('--stride', type=int, default=1,
                        help='Export every Nth message (default: all)')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Maximum exported frames, 0 means all')
    parser.add_argument('--bounds', type=float, nargs=6,
                        metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
                        help='Optional crop in original sensor frame, metres')
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 0:
        parser.error('stride must be >=1 and max-frames >=0')
    if not np.isfinite(args.decay_time) or args.decay_time < 0:
        parser.error('decay-time must be finite and >=0')
    if args.decay_time > 0 and not args.allow_uncompensated:
        parser.error('Accumulation requires --allow-uncompensated; no pose correction is implemented')
    if not args.bag.exists():
        parser.error('Bag does not exist')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    if args.bounds and (not np.isfinite(args.bounds).all() or
                        np.any(np.array(args.bounds[:3]) >= args.bounds[3:])):
        parser.error('Bounds must be finite with minimum < maximum')
    try:
        InputSettings(args.topic, args.frame_id, args.intensity_field, args.decay_time)
    except ValueError as error:
        parser.error(str(error))
    profile = standard(args.bounds, args.decay_time, args.frame_id, args.intensity_field)
    signature = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    if args.reference_manifest:
        reference = json.loads(args.reference_manifest.read_text())
        if reference.get('status') != 'complete' or reference.get('standard') != profile:
            parser.error('Reference must be complete and use the same standard/bounds')

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topics.get(args.topic) != 'sensor_msgs/msg/PointCloud2':
        parser.error('Selected topic is missing or is not PointCloud2')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'points').mkdir()
    manifest = dict(bag=str(args.bag.resolve()), topic=args.topic,
                    standard=profile, standard_sha256=signature,
                    max_frames=args.max_frames,
                    features=['x', 'y', 'z', 'intensity'], dtype='<f4',
                    accumulation_seconds=args.decay_time, motion_compensation=False,
                    intensity='raw, unnormalized', stride=args.stride,
                    bounds=args.bounds, status='incomplete')
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    seen = saved = total_points = 0
    previous_stamp = None
    buffer = TemporalBuffer(args.decay_time)
    if args.decay_time > 0:
        print('WARNING: accumulating without motion compensation; moving scenes can smear.', flush=True)
    with (args.output / 'frames.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'source_index', 'bag_timestamp_ns',
                         'header_timestamp_ns', 'frame_id', 'raw_points',
                         'saved_points', 'nonfinite_removed', 'crop_removed',
                         'window_frames', 'window_start_ns', 'window_span_ns'])
        while reader.has_next():
            _, data, bag_ns = reader.read_next()
            index = seen
            seen += 1
            msg = deserialize_message(data, PointCloud2)
            stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
            if msg.header.frame_id != args.frame_id:
                raise ValueError(f'Unexpected frame: {msg.header.frame_id!r}; no automatic transform')
            if stamp <= 0 or (previous_stamp is not None and stamp <= previous_stamp):
                raise ValueError('Header timestamps must be positive and strictly increasing')
            previous_stamp = stamp
            points = extract_points(msg, args.intensity_field)
            raw = len(points)
            points = points[np.isfinite(points).all(axis=1)]
            invalid = raw - len(points)
            before_crop = len(points)
            if args.bounds:
                xyz = points[:, :3]
                points = points[((xyz >= args.bounds[:3]) &
                                 (xyz <= args.bounds[3:])).all(axis=1)]
            buffer.append(stamp, points, raw, invalid, before_crop - len(points))
            # Stride only selects output samples; skipped source frames still contribute.
            if index % args.stride:
                continue
            points, (raw, invalid, cropped) = buffer.merged()
            sample_id = f'{index:06d}'
            with (args.output / 'points' / f'{sample_id}.bin').open('xb') as out:
                points.astype('<f4', copy=False).tofile(out)
            writer.writerow([sample_id, index, bag_ns, stamp, msg.header.frame_id,
                             raw, len(points), invalid, cropped, len(buffer.frames),
                             buffer.frames[0][0], stamp - buffer.frames[0][0]])
            saved += 1
            total_points += len(points)
            if saved % 100 == 0:
                print(f'Exported {saved} frames', flush=True)
            if args.max_frames and saved >= args.max_frames:
                break
    if saved == 0 or total_points == 0:
        raise ValueError('No usable points exported; output remains incomplete')
    manifest.update(status='complete', saved_frames=saved, saved_points=total_points,
                    source_messages_read=seen)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {saved} frames / {total_points} points to {args.output}')


if __name__ == '__main__':
    main()
