from types import SimpleNamespace

import numpy as np

from fastwam.datasets.opentouch.canonical import normalize_force
from fastwam.datasets.opentouch.preprocess import (
    _fit_force_normalization,
    build_force_quality_report,
)


def test_train_only_positive_force_stats_do_not_collapse_on_zeros():
    force = np.zeros((1000, 2), dtype=np.float32)
    force[-20:, 0] = np.linspace(1.0, 100.0, 20)
    force[-10:, 1] = np.linspace(1.0, 200.0, 10)
    valid = np.ones_like(force, dtype=bool)
    stats = _fit_force_normalization(force, valid)
    normalized = normalize_force(force, stats)

    assert np.all(stats["force_knee"] > 0)
    assert np.all(stats["force_upper"] > stats["force_knee"])
    assert np.all(np.diff(normalized[-10:, 0]) > 0)
    assert np.all(np.diff(normalized[-10:, 1]) > 0)
    assert np.all(normalized >= 0) and np.all(normalized <= 1)


def test_force_quality_report_tracks_contact_variation_and_saturation(tmp_path):
    trajectories = tmp_path / "trajectories"
    trajectories.mkdir()
    clean = np.zeros((40, 2), dtype=np.float32)
    clean[8:, 0] = np.linspace(0.0, 100.0, 32)
    clean[8:, 1] = np.linspace(0.0, 200.0, 32)
    valid = np.ones_like(clean, dtype=bool)
    np.savez_compressed(trajectories / "demo.npz", force_clean=clean, force_valid=valid)
    stats = _fit_force_normalization(clean, valid)
    record = SimpleNamespace(
        sequence_id="source::demo_000",
        split="train",
        cache_file="trajectories/demo.npz",
    )
    windows = {
        "train": [{"sequence_id": record.sequence_id, "current_frame": 7}],
        "val": [],
        "test": [],
    }
    report = build_force_quality_report([record], windows, tmp_path, stats)

    assert report["train"]["future_contact_window_count"] == 1
    assert report["train"]["constant_contact_window_count"] == 0
    assert all(report["train"]["upper_saturation_below_5_percent_per_finger"])
    assert all(value > 0 for value in report["train"]["mean_future_amplitude_per_finger"])
