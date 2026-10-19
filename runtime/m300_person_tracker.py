"""Short sensor-frame association/box holding; no ROS, motion prediction or ML.

Greedy one-to-one XY centre matching is intentionally lightweight. It is not
ego-motion compensation and can swap IDs when people cross or stand close.
Only detections can start/refresh a track; held boxes never refresh their age.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class PersonTrack:
    track_id: int
    score: float
    class_name: str
    box: np.ndarray
    last_seen_ns: int
    hits: int = 1
    matched: bool = True
    hold_age: float = 0.0


class PersonTracker:
    def __init__(self, start_score=0.6, keep_score=0.6, hold_seconds=0.3,
                 match_distance=0.8, min_hits=2, max_tracks=10):
        if not (math.isfinite(start_score) and math.isfinite(keep_score)
                and 0 <= keep_score <= start_score <= 1):
            raise ValueError("Require 0 <= keep_score <= start_score <= 1")
        if not math.isfinite(hold_seconds) or not 0 < hold_seconds <= 1.0:
            raise ValueError("hold_seconds must be finite, positive and <= 1 second")
        if not math.isfinite(match_distance) or match_distance <= 0:
            raise ValueError("match_distance must be finite and positive")
        if type(min_hits) is not int or min_hits < 1:
            raise ValueError("min_hits must be a positive integer")
        if type(max_tracks) is not int or max_tracks < 1:
            raise ValueError("max_tracks must be a positive integer")
        self.start_score = start_score
        self.keep_score = keep_score
        self.hold_ns = round(hold_seconds * 1e9)
        self.match_distance = match_distance
        self.min_hits = min_hits
        self.max_tracks = max_tracks
        self.tracks = {}
        self.next_id = 0
        self.last_stamp_ns = None

    def reset(self):
        self.tracks.clear()
        self.last_stamp_ns = None
        # Never recycle IDs; allows callers to delete old RViz markers safely.

    def update(self, detections, stamp_ns):
        if stamp_ns <= 0:
            raise ValueError("Tracking requires a positive message timestamp")
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            self.reset()
        self.last_stamp_ns = stamp_ns
        self.tracks = {key: t for key, t in self.tracks.items()
                       if stamp_ns - t.last_seen_ns <= self.hold_ns}
        candidates = []
        for score, name, box in detections:
            box = np.asarray(box, dtype=np.float32)
            if (name.casefold() not in {"person", "pedestrian"}
                    or not math.isfinite(score) or not self.keep_score <= score <= 1
                    or box.shape != (7,) or not np.isfinite(box).all()
                    or (box[3:6] <= 0).any()):
                continue
            candidates.append((float(score), name, box.copy()))

        pairs = []
        for key, track in self.tracks.items():
            track.matched = False
            for index, (score, name, box) in enumerate(candidates):
                # Tentative tracks need consecutive strong detections too.
                if name.casefold() != track.class_name.casefold():
                    continue
                if track.hits < self.min_hits and score < self.start_score:
                    continue
                distance = float(np.linalg.norm(box[:2] - track.box[:2]))
                z_gate = max(0.5, float(box[5] + track.box[5]) / 2)
                if distance <= self.match_distance and abs(float(box[2] - track.box[2])) <= z_gate:
                    pairs.append((distance, key, index))
        used_tracks, used_candidates = set(), set()
        for _, key, index in sorted(pairs):
            if key in used_tracks or index in used_candidates:
                continue
            track = self.tracks[key]
            track.score, track.class_name, track.box = candidates[index]
            track.last_seen_ns = stamp_ns
            track.hits += 1
            track.matched = True
            track.hold_age = 0.0
            used_tracks.add(key)
            used_candidates.add(index)

        # Drop one-frame tentative flashes; confirmed tracks may briefly coast.
        self.tracks = {key: t for key, t in self.tracks.items()
                       if t.matched or t.hits >= self.min_hits}
        for index in sorted(range(len(candidates)), key=lambda i: -candidates[i][0]):
            score, name, box = candidates[index]
            if index in used_candidates or score < self.start_score:
                continue
            if len(self.tracks) >= self.max_tracks:
                break
            key = self.next_id
            self.next_id += 1
            self.tracks[key] = PersonTrack(key, score, name, box, stamp_ns)
        visible = []
        for key, track in sorted(self.tracks.items()):
            track.hold_age = (stamp_ns - track.last_seen_ns) / 1e9
            if track.hits >= self.min_hits:
                visible.append(track)
        return visible
