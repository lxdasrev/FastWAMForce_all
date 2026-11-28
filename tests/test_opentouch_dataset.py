import hashlib
import io
import json

import h5py
import numpy as np
import pytest
from PIL import Image

from fastwam.datasets.opentouch.canonical import DATA_VERSION
from fastwam.datasets.opentouch.dataset import OpenTouchForceDataset


def _jpeg():
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), (40, 80, 120)).save(stream, format="JPEG")
    return np.frombuffer(stream.getvalue(), dtype=np.uint8)


@pytest.mark.parametrize("history,current,total", [(0, 0, 33), (16, 16, 49)])
def test_dataset_33_and_49_contract(tmp_path, history, current, total):
    source = tmp_path / "source.hdf5"
    payload = _jpeg()
    with h5py.File(source, "w") as handle:
        group = handle.create_group("data/demo_000")
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        images = group.create_dataset("rgb_images_jpeg", (49,), dtype=dtype)
        for i in range(49):
            images[i] = payload
    trajectories = tmp_path / "trajectories"
    trajectories.mkdir()
    np.savez_compressed(
        trajectories / "demo.npz",
        action=np.zeros((48, 7), np.float32), action_valid=np.ones(48, bool),
        proprio=np.zeros((49, 8), np.float32), pose_valid=np.ones(49, bool),
        force_relative_pressure=np.ones((49, 2), np.float32),
        force_valid=np.ones((49, 2), bool), force_contact=np.ones((49, 2), bool),
    )
    window = {
        "sequence_id": "source::demo_000", "source_file": str(source), "demo_key": "demo_000",
        "cache_file": "trajectories/demo.npz", "current_frame": current, "prompt": "Use the gripper.",
    }
    manifest = {"version": DATA_VERSION, "windows": {"train": [window], "val": [window], "test": []}}
    contract = {"version": DATA_VERSION, "force_history_frames": history}
    (tmp_path / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "data_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    np.savez(
        tmp_path / "normalization.npz",
        action_mean=np.zeros(7), action_scale=np.ones(7),
        proprio_mean=np.zeros(8), proprio_scale=np.ones(8),
        force_knee=np.ones(2), force_upper=np.full(2, 3.0),
        force_normalization_max=np.full(2, np.arcsinh(3.0)),
        force_clip_min=np.asarray(0.0), force_clip_max=np.asarray(1.0),
        pressure_adc_reference=np.asarray(3072.0),
    )
    (tmp_path / "force_quality_report.json").write_text("{}", encoding="utf-8")
    hashes = {}
    for name in (
        "split_manifest.json", "data_contract.json", "normalization.npz",
        "force_quality_report.json",
    ):
        hashes[name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    (tmp_path / "contract_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    dataset = OpenTouchForceDataset(
        tmp_path, split="train", force_history_frames=history, use_text_embed_cache=False,
    )
    sample = dataset[0]
    assert sample["video"].shape == (3, 9, 224, 224)
    assert sample["action_chunk"].shape == (32, 7)
    assert sample["proprio"].shape == (8,)
    assert sample["force"].shape == (total, 2)
    assert sample["force_contact"].shape == (total, 2)
    assert sample["force_contact"].all()
    assert sample["force_observed_frames"] == history + 1
    assert sample["split_group"] == "source::demo_000"
    action = np.ones((2, 7), dtype=np.float32)
    force = np.ones((2, 2), dtype=np.float32)
    assert np.allclose(dataset.denormalize_action(action), action)
    assert np.allclose(dataset.denormalize_force(force), 3.0)
    assert np.allclose(
        dataset.denormalize_force(np.array([[-1.0, 2.0]], dtype=np.float32)),
        [[0.0, 3.0]],
    )
    assert dataset.fixed_eval_indices(1, activity_ranked=True, unique_groups=True) == [0]
    dataset.windows = [
        {"sequence_id": "a", "split_group": "g1", "force_activity_score": 0.1},
        {"sequence_id": "b", "split_group": "g1", "force_activity_score": 0.9},
        {"sequence_id": "c", "split_group": "g2", "force_activity_score": 0.4},
    ]
    assert set(dataset.fixed_eval_indices(2, activity_ranked=True, unique_groups=True)) == {1, 2}

    contract["version"] = "opentouch-fastwam-force-v2"
    (tmp_path / "data_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="version mismatch"):
        OpenTouchForceDataset(
            tmp_path, split="train", force_history_frames=history,
            use_text_embed_cache=False, strict_contract=False,
        )
