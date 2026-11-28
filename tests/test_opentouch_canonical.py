import numpy as np

from fastwam.datasets.opentouch.canonical import (
    DEFAULT_TWO_FINGER_TAXELS,
    canonicalize_hand_trajectory,
    normalize_force,
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


def test_two_finger_direct_adc_deficit_and_invalid_taxels():
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


def test_two_finger_adc_reference_and_above_reference_are_valid_no_contact():
    raw = np.full((3, 16, 16), 3072.0, dtype=np.float32)
    raw[0] = 3000.0
    raw[2] = 3100.0
    force, valid = pressure_to_two_finger(raw)
    assert valid.all()
    assert np.allclose(force[0], 72.0)
    assert np.allclose(force[1:], 0.0)


def test_two_finger_valid_fraction_boundary_and_mask_coordinates():
    assert len(DEFAULT_TWO_FINGER_TAXELS["thumb"]) == 11
    assert len(DEFAULT_TWO_FINGER_TAXELS["index"]) == 6
    assert DEFAULT_TWO_FINGER_TAXELS["thumb"][0] == (2, 3)
    assert DEFAULT_TWO_FINGER_TAXELS["index"][0] == (6, 15)
    raw = np.full((2, 16, 16), 3072.0, dtype=np.float32)
    thumb = DEFAULT_TWO_FINGER_TAXELS["thumb"]
    index = DEFAULT_TWO_FINGER_TAXELS["index"]
    for row, col in thumb[:2]:
        raw[0, row, col] = np.nan  # 9/11 remains valid.
    for row, col in thumb[:3]:
        raw[1, row, col] = np.nan  # 8/11 is below 80%.
    raw[0, index[0][0], index[0][1]] = 0.0  # 5/6 remains valid.
    for row, col in index[:2]:
        raw[1, row, col] = np.inf  # 4/6 is below 80%.
    _, valid = pressure_to_two_finger(raw)
    assert valid.tolist() == [[True, True], [False, False]]


def test_force_v3_normalization_bounds_and_round_trip():
    knee = np.array([10.0, 20.0], dtype=np.float32)
    upper = np.array([100.0, 200.0], dtype=np.float32)
    normalization_max = np.arcsinh(upper / knee)
    stats = {
        "force_knee": knee,
        "force_upper": upper,
        "force_normalization_max": normalization_max,
        "force_clip_min": np.asarray(0.0, dtype=np.float32),
        "force_clip_max": np.asarray(1.0, dtype=np.float32),
    }
    clean = np.array([[0.0, 0.0], [10.0, 20.0], [100.0, 200.0], [200.0, 400.0]])
    normalized = normalize_force(clean, stats)
    assert np.allclose(normalized[0], 0.0)
    assert np.allclose(normalized[2], 1.0)
    assert np.allclose(normalized[3], 1.0)
    restored = np.sinh(normalized[:3] * normalization_max) * knee
    assert np.allclose(restored, clean[:3], atol=1e-5)
