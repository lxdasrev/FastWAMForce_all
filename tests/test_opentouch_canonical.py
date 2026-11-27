import numpy as np

from fastwam.datasets.opentouch.canonical import (
    calibrate_and_clean_force,
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


def test_per_demo_smooth_deadband_preserves_contact_dynamics():
    baseline = np.array([100.0, 200.0], dtype=np.float32)
    force = np.tile(baseline, (20, 1))
    force[:8] += np.array([
        [-2, 1], [2, -1], [0, 0], [1, 2], [-1, -2], [2, 1], [-2, 0], [0, -1],
    ], dtype=np.float32)
    force[8:, 0] += np.linspace(0, 100, 12)
    force[8:, 1] += np.linspace(0, 200, 12)
    valid = np.ones_like(force, dtype=bool)

    clean, contact, estimated_baseline, noise = calibrate_and_clean_force(
        force, valid, onset_idx=8,
    )
    assert np.allclose(estimated_baseline, baseline, atol=1.0)
    assert np.all(noise >= 5.0)
    assert np.allclose(clean[:8], 0.0)
    assert not contact[:8].any()
    assert np.all(np.diff(clean[9:, 0]) >= 0)
    assert clean[-1, 1] > clean[-1, 0] > 0


def test_per_demo_smooth_deadband_removes_pure_noise_with_fallback_calibration():
    rng = np.random.default_rng(7)
    baseline = np.array([120.0, 240.0], dtype=np.float32)
    force = baseline + rng.normal(0.0, 1.0, size=(64, 2)).astype(np.float32)
    valid = np.ones_like(force, dtype=bool)
    clean, contact, estimated_baseline, noise = calibrate_and_clean_force(force, valid)
    assert np.allclose(estimated_baseline, baseline, atol=2.0)
    assert np.all(noise == 5.0)
    assert np.all(clean == 0.0)
    assert not contact.any()


def test_force_v2_normalization_bounds_and_round_trip():
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


def test_force_cleaning_keeps_single_finger_invalid_masked():
    force = np.full((12, 2), 100.0, dtype=np.float32)
    force[8:, 0] = 200.0
    force[:, 1] = 999.0
    valid = np.ones((12, 2), dtype=bool)
    valid[:, 1] = False
    clean, contact, _, _ = calibrate_and_clean_force(force, valid, onset_idx=8)
    assert np.all(clean[:, 1] == 0)
    assert not contact[:, 1].any()
    assert contact[-1, 0]
