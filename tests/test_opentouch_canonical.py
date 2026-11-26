import numpy as np

from fastwam.datasets.opentouch.canonical import (
    canonicalize_hand_trajectory,
    pressure_to_two_finger,
    so3_log,
)


def _landmarks(frames=4):
    points = np.zeros((frames, 21, 3), dtype=np.float64)
    points[:, 0] = [0, 0, 0]
    points[:, 5] = [1, 1, 0]
    points[:, 9] = [0, 1, 0]
    points[:, 17] = [-1, 1, 0]
    points[:, 4] = [-0.2, 1.5, 0]
    points[:, 8] = [0.2, 1.5, 0]
    return points


def test_canonical_translation_and_absolute_gripper():
    lm = _landmarks()
    palm = np.stack([np.arange(4), np.zeros(4), np.zeros(4)], axis=-1)
    timestamps = np.arange(4) * (1e9 / 30)
    result = canonicalize_hand_trajectory(palm, lm, timestamps)
    assert np.allclose(result.proprio[0, :3], 0)
    assert np.allclose(result.proprio[0, 3:7], [0, 0, 0, 1])
    assert np.allclose(result.action[:, :3], [0.5, 0, 0])  # palm width is 2
    assert np.allclose(result.action[:, 3:6], 0)
    assert np.allclose(result.action[:, 6], result.proprio[1:, 7])


def test_so3_log_rotation():
    theta = 0.2
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    assert np.allclose(so3_log(rotation), [0, 0, theta], atol=1e-6)


def test_two_finger_invalid_taxels_are_not_averaged():
    raw = np.full((1, 16, 16), 3072, dtype=np.float32)
    raw[0, 2, 3] = 2072
    force, valid = pressure_to_two_finger(raw)
    assert valid.all()
    assert np.isclose(force[0, 0], 1000 / 11)
    broken = raw.copy()
    broken[0, 2, 3] = np.nan
    broken[0, 1, 3] = np.nan
    broken[0, 0, 3] = np.nan
    _, broken_valid = pressure_to_two_finger(broken)
    assert not broken_valid[0, 0]
