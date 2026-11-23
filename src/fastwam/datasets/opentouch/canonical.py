"""Canonical-frame kinematics and two-finger pressure conversion.

OpenTouch has no robot/world calibration.  The trajectory produced here is a
dimensionless, palm-width-normalized pseudo TCP trajectory and must not be sent
to a robot controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

ACTION_DIM = 7
PROPRIO_DIM = 8
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17

# Copied from OpenTouch five-finger/assets/fingertip_taxels.json.  Coordinates
# refer directly to right_pressure[row, col].
DEFAULT_TWO_FINGER_TAXELS = {
    "thumb": ((2, 3), (1, 3), (0, 3), (2, 2), (1, 2), (0, 2),
              (2, 1), (1, 1), (0, 1), (1, 0), (0, 0)),
    "index": ((6, 15), (5, 15), (6, 14), (5, 14), (6, 13), (5, 13)),
}


def _normalize(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps), n[..., 0]


def project_so3(matrix: np.ndarray) -> np.ndarray:
    """Project one or more matrices to the closest proper rotation."""
    u, _, vh = np.linalg.svd(matrix)
    out = u @ vh
    bad = np.linalg.det(out) < 0
    if np.any(bad):
        u = u.copy()
        u[bad, :, -1] *= -1
        out = u @ vh
    return out


def hand_rotation(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a right-handed palm frame from wrist and MCP landmarks."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (21, 3):
        raise ValueError(f"landmarks must have shape [T,21,3], got {points.shape}")
    x, nx = _normalize(points[:, INDEX_MCP] - points[:, PINKY_MCP])
    y0 = points[:, MIDDLE_MCP] - points[:, WRIST]
    y0 = y0 - (x * y0).sum(-1, keepdims=True) * x
    y, ny = _normalize(y0)
    z, nz = _normalize(np.cross(x, y))
    y = np.cross(z, x)
    rot = np.stack((x, y, z), axis=-1)
    finite = np.isfinite(points).all(axis=(1, 2)) & np.isfinite(rot).all(axis=(1, 2))
    valid = finite & (nx > 1e-5) & (ny > 1e-5) & (nz > 1e-5)
    rot[valid] = project_so3(rot[valid])
    rot[~valid] = np.eye(3)
    return rot, valid


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Numerically stable SO(3) to quaternion conversion, scalar last."""
    m = np.asarray(matrix, dtype=np.float64)
    q = np.empty(m.shape[:-2] + (4,), dtype=np.float64)
    # Eigen decomposition of Bar-Itzhack's symmetric K matrix avoids branches.
    flat_m, flat_q = m.reshape(-1, 3, 3), q.reshape(-1, 4)
    for i, r in enumerate(flat_m):
        k = np.array([
            [r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0], r[2, 1] - r[1, 2]],
            [r[0, 1] + r[1, 0], r[1, 1] - r[0, 0] - r[2, 2], r[1, 2] + r[2, 1], r[0, 2] - r[2, 0]],
            [r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], r.trace()],
        ]) / 3.0
        _, vec = np.linalg.eigh(k)
        flat_q[i] = vec[:, -1]
        if flat_q[i, 3] < 0:
            flat_q[i] *= -1
    # Enforce temporal sign continuity.
    for i in range(1, len(flat_q)):
        if np.dot(flat_q[i - 1], flat_q[i]) < 0:
            flat_q[i] *= -1
    return q


def so3_log(matrix: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """SO(3) logarithm as an axis-angle vector, including near-pi handling."""
    m = np.asarray(matrix, dtype=np.float64)
    flat = m.reshape(-1, 3, 3)
    out = np.zeros((len(flat), 3), dtype=np.float64)
    for i, r in enumerate(flat):
        cos_theta = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        if theta < eps:
            out[i] = 0.5 * vee
        elif np.pi - theta < 1e-4:
            vals, vecs = np.linalg.eigh((r + np.eye(3)) * 0.5)
            axis = vecs[:, np.argmax(vals)]
            if np.dot(axis, vee) < 0:
                axis = -axis
            out[i] = theta * axis
        else:
            out[i] = theta / (2.0 * np.sin(theta)) * vee
    return out.reshape(m.shape[:-2] + (3,))


@dataclass(frozen=True)
class CanonicalTrajectory:
    proprio: np.ndarray
    action: np.ndarray
    pose_valid: np.ndarray
    action_valid: np.ndarray
    rotation: np.ndarray
    palm_width: float
    anchor_index: int


def canonicalize_hand_trajectory(
    palm_position: np.ndarray,
    landmarks: np.ndarray,
    timestamps_ns: np.ndarray | None = None,
    *,
    nominal_hz: float = 30.0,
    dt_tolerance: float = 0.5,
    gripper_range: tuple[float, float] = (0.05, 1.50),
    max_translation_per_frame: float = 1.0,
    max_rotation_per_frame: float = 1.5,
) -> CanonicalTrajectory:
    """Convert a complete demo to a fixed canonical pseudo robot trajectory."""
    p = np.asarray(palm_position, dtype=np.float64)
    lm = np.asarray(landmarks, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(lm):
        raise ValueError(f"palm_position/landmarks mismatch: {p.shape}, {lm.shape}")
    rot, rotation_valid = hand_rotation(lm)
    palm_finite = np.isfinite(p).all(-1)
    width = np.linalg.norm(lm[:, INDEX_MCP] - lm[:, PINKY_MCP], axis=-1)
    pose_valid = rotation_valid & palm_finite & np.isfinite(width) & (width > 1e-5)
    valid_idx = np.flatnonzero(pose_valid)
    if not len(valid_idx):
        raise ValueError("demo has no geometrically valid hand pose")
    anchor = int(valid_idx[0])
    initial = valid_idx[:5]
    palm_width = float(np.median(width[initial]))
    scale = 1.0 / max(palm_width, 1e-8)
    r0 = rot[anchor]
    p0 = p[anchor]
    pc = np.einsum("ij,tj->ti", r0.T, p - p0) * scale
    rc = np.einsum("ij,tjk->tik", r0.T, rot)
    tip_distance = np.linalg.norm(lm[:, THUMB_TIP] - lm[:, INDEX_TIP], axis=-1) * scale
    low, high = gripper_range
    gripper = np.clip((tip_distance - low) / (high - low), 0.0, 1.0)
    quat = matrix_to_quaternion_xyzw(rc)
    proprio = np.concatenate((pc, quat, gripper[:, None]), axis=-1).astype(np.float32)
    rel_rot = np.einsum("tij,tkj->tik", rc[1:], rc[:-1])
    action = np.concatenate((pc[1:] - pc[:-1], so3_log(rel_rot), gripper[1:, None]), axis=-1)
    action_valid = pose_valid[:-1] & pose_valid[1:]
    if timestamps_ns is not None:
        ts = np.asarray(timestamps_ns, dtype=np.float64)
        if ts.shape != (len(p),):
            raise ValueError(f"timestamps must have shape {(len(p),)}, got {ts.shape}")
        dt = np.diff(ts) * 1e-9
        expected = 1.0 / nominal_hz
        action_valid &= np.isfinite(dt) & (dt >= expected * (1 - dt_tolerance)) & (dt <= expected * (1 + dt_tolerance))
    action_valid &= (
        np.isfinite(action).all(axis=1)
        & (np.linalg.norm(action[:, :3], axis=1) <= max_translation_per_frame)
        & (np.linalg.norm(action[:, 3:6], axis=1) <= max_rotation_per_frame)
    )
    action = action.astype(np.float32)
    proprio[~pose_valid] = 0
    action[~action_valid] = 0
    return CanonicalTrajectory(proprio, action, pose_valid, action_valid, rc.astype(np.float32), palm_width, anchor)


def pressure_to_two_finger(
    pressure: np.ndarray,
    taxels: Mapping[str, Sequence[Sequence[int]]] = DEFAULT_TWO_FINGER_TAXELS,
    *,
    adc_max: float = 3072.0,
    min_valid_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate OpenTouch ADC taxels to thumb/index relative pressure."""
    raw = np.asarray(pressure, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (16, 16):
        raise ValueError(f"pressure must have shape [T,16,16], got {raw.shape}")
    force, valid = [], []
    for finger in ("thumb", "index"):
        coords = np.asarray(taxels[finger], dtype=np.int64)
        values = raw[:, coords[:, 0], coords[:, 1]]
        taxel_valid = np.isfinite(values) & (values > 0) & (values <= adc_max)
        finger_valid = taxel_valid.mean(axis=1) >= min_valid_fraction
        corrected = np.where(taxel_valid, np.maximum(adc_max - values, 0), 0)
        denom = np.maximum(taxel_valid.sum(axis=1), 1)
        finger_force = corrected.sum(axis=1) / denom
        finger_force[~finger_valid] = 0
        force.append(finger_force)
        valid.append(finger_valid)
    return np.stack(force, axis=-1).astype(np.float32), np.stack(valid, axis=-1)


def normalize_force(force: np.ndarray, stats: Mapping[str, np.ndarray | float]) -> np.ndarray:
    """Apply the contract-defined force preprocessing in its fixed order."""
    x = np.asarray(force, dtype=np.float32)
    deadzone = np.asarray(stats["force_deadzone"], dtype=np.float32)
    asinh_scale = np.asarray(stats["force_asinh_scale"], dtype=np.float32)
    median = np.asarray(stats["force_median"], dtype=np.float32)
    scale = np.asarray(stats["force_scale"], dtype=np.float32)
    clip = float(np.asarray(stats.get("force_clip", 5.0)))
    x = np.maximum(x - deadzone, 0)
    x = np.arcsinh(x / np.maximum(asinh_scale, 1e-6))
    x = (x - median) / np.maximum(scale, 1e-6)
    return np.clip(x, -clip, clip).astype(np.float32)
