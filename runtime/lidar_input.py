"""Sensor-independent PointCloud2 XYZI decoding; no ROS imports or transforms."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class InputSettings:
    input_topic: str = "/pointcloud"
    expected_frame_id: str = "m300_lidar"
    intensity_field: str = "intensity"
    decay_time_sec: float = 0.3
    score_threshold: float = 0.3

    def __post_init__(self):
        for name in ("input_topic", "expected_frame_id", "intensity_field"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or any(c.isspace() for c in value):
                raise ValueError(f"{name} must be a nonempty string without whitespace")
        if self.intensity_field in {"x", "y", "z"}:
            raise ValueError("intensity_field must not alias a coordinate")
        if not math.isfinite(self.decay_time_sec) or self.decay_time_sec < 0:
            raise ValueError("decay_time_sec must be finite and nonnegative")
        if not math.isfinite(self.score_threshold) or not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be finite and within [0, 1]")


def decode_xyzi(msg, intensity_field="intensity"):
    """Return raw float32 Nx4, honoring offsets/endian/row padding.

    No invented intensity, normalization, axis conversion, or finite filtering.
    Caller applies its own nonfinite policy AFTER float32 conversion.
    Additional fields (ring/time/etc.) are ignored.
    """
    InputSettings(intensity_field=intensity_field)
    scalar_types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4",
                    6: "u4", 7: "f4", 8: "f8"}
    names = ["x", "y", "z", intensity_field]
    fields = {}
    for field in msg.fields:
        if field.name in names:
            if field.name in fields:
                raise ValueError(f"Duplicate point field: {field.name}")
            fields[field.name] = field
    missing = set(names) - fields.keys()
    if missing:
        raise ValueError(f"PointCloud missing required fields: {sorted(missing)}")
    endian = ">" if msg.is_bigendian else "<"
    formats, offsets = [], []
    for name in names:
        field = fields[name]
        if field.count != 1 or field.datatype not in scalar_types:
            raise ValueError(f"Unsupported scalar point field: {name}")
        fmt = endian + scalar_types[field.datatype]
        if field.offset < 0 or field.offset + np.dtype(fmt).itemsize > msg.point_step:
            raise ValueError(f"Point field outside point_step: {name}")
        formats.append(fmt)
        offsets.append(field.offset)
    if msg.width == 0 or msg.height == 0:
        return np.empty((0, 4), dtype="<f4")
    if msg.row_step < msg.width * msg.point_step or len(msg.data) < msg.row_step * msg.height:
        raise ValueError("Truncated PointCloud2 or invalid row_step")
    dtype = np.dtype(dict(names=names, formats=formats, offsets=offsets,
                         itemsize=msg.point_step))
    rows = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.row_step, msg.point_step))
    with np.errstate(over="ignore", invalid="ignore"):
        return np.column_stack([rows[name].reshape(-1) for name in names]).astype("<f4")
