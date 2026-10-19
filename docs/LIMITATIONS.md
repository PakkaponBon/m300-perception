# Known limitations and safety scope

## Domain shift and lookalikes

A chair headrest, shelving, and other vertical structures can resemble sparse human returns. High model confidence can still be wrong. More thresholds or longer tracking do not reliably solve this.

Collect reviewed hard negatives from the target environment, and positive examples at missed viewpoints/distances, including stationary and seated people. An empty label is a negative only when **no target person is present**. Unlabelled visible people are annotation errors, not negatives. Keep a scene/recording-level held-out test where possible.

## Accumulation and motion

The 0.3-second input is concatenation of whole messages by header timestamp, not per-point deskewing or pose compensation. A moving sensor can smear static structures; people can leave motion trails even with a stationary sensor. Training and deployment must use compatible preprocessing. Raising, lowering, rotating, or inverting the LiDAR changes the coordinate distribution and visibility; a software transform cannot restore unseen returns.

## Tracking and visualization

The tracker is short-lived greedy association, not a motion model or robust multi-object tracker. It may switch IDs, confirm repeated false detections, or briefly hold an incorrect box. A HOLD marker is not a fresh model observation. Sparse background points are only context, not a map or semantic wall detector.

## Geometric occupancy

The front ROI counts returns, not objects. Ground/floor returns, isolated noise, glass, occlusion, body mounting, and insufficient samples may cause false occupied or false clear. The sensor-relative height slice is not ground estimation. CLEAR means only that the valid sample did not meet the configured occupancy criterion.

## Deployment and stale data

Always consume `state` and `data_valid` together with obstacle measurements. The `detected` Bool is only updated on valid clouds; its last value can remain cached across invalid/no-data states. No message, invalid data, or a dead node is never proof of clear space.

This software does not send motor commands, implement a protective stop, or satisfy a functional-safety standard. Keep independent risk controls and safety-rated systems wherever people and machinery interact.
