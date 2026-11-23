"""Build the lightweight OpenTouch cache consumed by FastWAMForce.

RGB JPEG bytes remain in the source HDF5 files.  This command stores only the
canonical pseudo trajectory, two-finger pressure, validity, split windows and
the data contract that is bound to checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from .canonical import (
    DEFAULT_TWO_FINGER_TAXELS,
    canonicalize_hand_trajectory,
    pressure_to_two_finger,
)

DATA_VERSION = "opentouch-fastwam-force-v1"
FUTURE_FRAMES = 32
VIDEO_STRIDE = 4
VIDEO_FRAMES = 9


def _absolute_path(path: Path) -> Path:
    """Make a path absolute without resolving UNC/WSL symlinks over the network."""
    return path if path.is_absolute() else Path.cwd() / path


@dataclass
class DemoRecord:
    sequence_id: str
    source_file: str
    demo_key: str
    length: int
    ts_start: int
    ts_end: int
    environment: str
    action_label: str
    object_name: str
    onset_idx: int | None
    split_group: str = ""
    split: str = ""
    cache_file: str = ""


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload) -> str:
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(packed).hexdigest()


def _load_annotations(annotation_dir: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    if not annotation_dir.exists():
        return result
    for path in sorted(annotation_dir.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                clip_id = str(row.get("clip_id", "")).strip()
                if clip_id:
                    result[clip_id] = row
    return result


def _annotation_for(annotations: dict[str, dict], stem: str, demo_key: str) -> dict:
    direct = annotations.get(f"{stem}::{demo_key}")
    if direct is not None:
        return direct
    # Some exports use different zero padding for the demo index.
    index = int(demo_key.rsplit("_", 1)[-1])
    suffix = f"::demo_{index}"
    candidates = [value for key, value in annotations.items() if key.startswith(stem) and key.endswith(suffix)]
    return candidates[0] if len(candidates) == 1 else {}


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def scan_records(input_dir: Path, annotation_dir: Path) -> list[DemoRecord]:
    annotations = _load_annotations(annotation_dir)
    records: list[DemoRecord] = []
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    for source in sorted(input_dir.glob("*.hdf5")):
        with h5py.File(source, "r", locking=False) as handle:
            demos = handle["data"]
            for demo_key in sorted(demos):
                group = demos[demo_key]
                length = int(group["timestamps"].shape[0])
                timestamps = group["timestamps"]
                annotation = _annotation_for(annotations, source.stem, demo_key)
                ts_start = _int_or_none(annotation.get("ts_start"))
                ts_end = _int_or_none(annotation.get("ts_end"))
                if ts_start is None:
                    ts_start = int(timestamps[0])
                if ts_end is None:
                    ts_end = int(timestamps[-1])
                records.append(DemoRecord(
                    sequence_id=f"{source.stem}::{demo_key}",
                    source_file=str(_absolute_path(source)),
                    demo_key=demo_key,
                    length=length,
                    ts_start=ts_start,
                    ts_end=ts_end,
                    environment=str(annotation.get("environment", "unknown") or "unknown"),
                    action_label=str(annotation.get("action", "interact with") or "interact with"),
                    object_name=str(annotation.get("object_name", "object") or "object"),
                    onset_idx=_int_or_none(annotation.get("onset_idx")),
                ))
    return records


def assign_connected_groups(records: list[DemoRecord], gap_frames: int = FUTURE_FRAMES, hz: float = 30.0) -> None:
    """Prevent temporally overlapping/nearby clips in a source from leaking."""
    max_gap_ns = int(gap_frames / hz * 1e9)
    by_source: dict[str, list[DemoRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_file].append(record)
    for source, items in by_source.items():
        items.sort(key=lambda item: (item.ts_start, item.ts_end))
        group_index, current_end = -1, -1
        stem = Path(source).stem
        for item in items:
            if group_index < 0 or item.ts_start > current_end + max_gap_ns:
                group_index += 1
                current_end = item.ts_end
            else:
                current_end = max(current_end, item.ts_end)
            item.split_group = f"{stem}::group_{group_index:04d}"


def stratified_group_split(records: list[DemoRecord], seed: int = 42) -> None:
    """Deterministic group split, stratified by environment and action."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[DemoRecord]] = defaultdict(list)
    for record in records:
        groups[record.split_group].append(record)
    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for group_id, items in groups.items():
        representative = items[0]
        if representative.onset_idx is None:
            contact_bin = "unknown_contact"
        else:
            contact_ratio = max(0.0, min(1.0, (representative.length - representative.onset_idx) / representative.length))
            contact_bin = "short_contact" if contact_ratio < 0.25 else "medium_contact" if contact_ratio < 0.50 else "long_contact"
        source = Path(representative.source_file).stem
        strata[(source, representative.environment, contact_bin)].append(group_id)
    assignment: dict[str, str] = {}
    for group_ids in strata.values():
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_test = int(round(n * 0.10)) if n >= 5 else 0
        n_val = int(round(n * 0.10)) if n >= 3 else 0
        for i, group_id in enumerate(group_ids):
            assignment[group_id] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    # Make all three splits non-empty for normal-sized datasets.
    if len(groups) >= 10:
        counts = Counter(assignment.values())
        train_groups = [key for key, value in assignment.items() if value == "train"]
        for split in ("val", "test"):
            if not counts[split] and train_groups:
                selected = train_groups.pop()
                assignment[selected] = split
    for record in records:
        record.split = assignment[record.split_group]


def _safe_stats(values: np.ndarray, valid: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    flattened = values.reshape(-1, dim)
    mask = valid.reshape(-1)
    selected = flattened[mask]
    if not len(selected):
        raise ValueError("cannot fit normalization: no valid training values")
    mean = selected.mean(axis=0, dtype=np.float64)
    std = selected.std(axis=0, dtype=np.float64)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def convert_records(records: list[DemoRecord], output_dir: Path) -> dict[str, np.ndarray]:
    cache_dir = output_dir / "trajectories"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_action, train_action_valid = [], []
    train_proprio, train_pose_valid = [], []
    train_force, train_force_valid, baseline = [], [], []
    by_source: dict[str, list[DemoRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_file].append(record)
    for source, source_records in by_source.items():
        with h5py.File(source, "r", locking=False) as handle:
            for record in source_records:
                group = handle[f"data/{record.demo_key}"]
                timestamps = np.asarray(group["timestamps"])
                trajectory = canonicalize_hand_trajectory(
                    np.asarray(group["right_palm_pos"]),
                    np.asarray(group["right_hand_landmarks"]),
                    timestamps,
                )
                force_raw, force_valid = pressure_to_two_finger(np.asarray(group["right_pressure"]))
                dt = np.diff(timestamps.astype(np.float64)) * 1e-9
                expected_dt = 1.0 / 30.0
                frame_time_valid = np.ones(len(timestamps), dtype=bool)
                frame_time_valid[1:] = (
                    np.isfinite(dt) & (dt >= expected_dt * 0.5) & (dt <= expected_dt * 1.5)
                )
                force_valid &= frame_time_valid[:, None]
                force_raw[~force_valid] = 0
                safe_name = record.sequence_id.replace("::", "__") + ".npz"
                cache_path = cache_dir / safe_name
                np.savez_compressed(
                    cache_path,
                    proprio=trajectory.proprio,
                    action=trajectory.action,
                    pose_valid=trajectory.pose_valid,
                    action_valid=trajectory.action_valid,
                    force_raw=force_raw,
                    force_valid=force_valid,
                    timestamps=timestamps,
                    palm_width=np.asarray(trajectory.palm_width, dtype=np.float32),
                    anchor_index=np.asarray(trajectory.anchor_index, dtype=np.int32),
                )
                record.cache_file = str(cache_path.relative_to(output_dir)).replace("\\", "/")
                if record.split == "train":
                    train_action.append(trajectory.action)
                    train_action_valid.append(trajectory.action_valid)
                    train_proprio.append(trajectory.proprio)
                    train_pose_valid.append(trajectory.pose_valid)
                    train_force.append(force_raw)
                    train_force_valid.append(force_valid)
                    if record.onset_idx is not None and record.onset_idx > 0:
                        base_idx = np.arange(min(record.onset_idx, len(force_raw)))
                    else:
                        totals = np.where(force_valid, force_raw, 0).sum(-1)
                        base_idx = np.argsort(totals)[:max(1, int(np.ceil(len(totals) * 0.20)))]
                    base = force_raw[base_idx]
                    base_valid = force_valid[base_idx]
                    baseline.append(np.where(base_valid, base, np.nan))
    actions = np.concatenate(train_action)
    action_valid = np.concatenate(train_action_valid)
    proprios = np.concatenate(train_proprio)
    pose_valid = np.concatenate(train_pose_valid)
    force = np.concatenate(train_force)
    force_valid = np.concatenate(train_force_valid)
    base = np.concatenate(baseline)
    action_mean, action_scale = _safe_stats(actions, action_valid, 7)
    proprio_mean, proprio_scale = _safe_stats(proprios, pose_valid, 8)
    deadzone = np.nanpercentile(base, 99, axis=0).astype(np.float32)
    deadzone = np.nan_to_num(deadzone, nan=0.0)
    positive = np.where(force_valid, np.maximum(force - deadzone, 0), np.nan)
    asinh_scale = np.nanmedian(np.where(positive > 0, positive, np.nan), axis=0).astype(np.float32)
    asinh_scale = np.nan_to_num(asinh_scale, nan=1.0)
    compressed = np.arcsinh(positive / np.maximum(asinh_scale, 1e-6))
    force_median = np.nanmedian(compressed, axis=0).astype(np.float32)
    q05, q95 = np.nanpercentile(compressed, [5, 95], axis=0)
    force_scale = np.maximum((q95 - q05) * 0.5, 1e-6).astype(np.float32)
    return {
        "action_mean": action_mean,
        "action_scale": action_scale,
        "proprio_mean": proprio_mean,
        "proprio_scale": proprio_scale,
        "force_deadzone": deadzone,
        "force_asinh_scale": asinh_scale,
        "force_median": force_median,
        "force_scale": force_scale,
        "force_clip": np.asarray(5.0, dtype=np.float32),
        "pressure_adc_max": np.asarray(3072.0, dtype=np.float32),
    }


def build_windows(records: list[DemoRecord], history_frames: int, window_stride: int) -> dict[str, list[dict]]:
    windows = {"train": [], "val": [], "test": []}
    for record in records:
        start = history_frames
        stop = record.length - FUTURE_FRAMES
        for current in range(start, stop, window_stride):
            windows[record.split].append({
                "sequence_id": record.sequence_id,
                "current_frame": current,
            })
    return windows


def write_contract(
    records: list[DemoRecord], windows: dict[str, list[dict]], stats: dict[str, np.ndarray],
    output_dir: Path, input_dir: Path, history_frames: int, seed: int, hash_sources: bool,
) -> None:
    np.savez(output_dir / "normalization.npz", **stats)
    taxel_payload = {key: [list(pair) for pair in value] for key, value in DEFAULT_TWO_FINGER_TAXELS.items()}
    source_files = sorted({record.source_file for record in records})
    source_hashes = {path: _sha256(Path(path)) if hash_sources else None for path in source_files}
    split_sequences = {
        split: sorted(record.sequence_id for record in records if record.split == split)
        for split in ("train", "val", "test")
    }
    manifest = {
        "version": DATA_VERSION,
        "seed": seed,
        "splits": split_sequences,
        "windows": windows,
        "records": [asdict(record) for record in records],
    }
    try:
        repo_root = Path(__file__).resolve().parents[4]
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    contract = {
        "version": DATA_VERSION,
        "processing_git_commit": git_commit,
        "source_root": str(_absolute_path(input_dir)),
        "force_semantics": "relative_pressure_from_adc_not_newtons",
        "canonical_frame": "fixed_first_valid_demo_hand_frame",
        "translation_unit": "initial_palm_width",
        "rotation": "so3_log_r_next_r_current_transpose_radians",
        "quaternion_order": "qx_qy_qz_qw",
        "action_semantics": ["delta_x", "delta_y", "delta_z", "rotvec_x", "rotvec_y", "rotvec_z", "next_absolute_gripper"],
        "proprio_semantics": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"],
        "nominal_hz": 30.0,
        "video_indices": list(range(0, FUTURE_FRAMES + 1, VIDEO_STRIDE)),
        "force_history_frames": history_frames,
        "force_observed_frames": history_frames + 1,
        "force_future_frames": FUTURE_FRAMES,
        "taxels": taxel_payload,
        "taxel_mask_hash": _json_hash(taxel_payload),
        "pressure_pipeline": ["3072-raw", "valid_taxel_mean", "per_finger_deadzone", "asinh", "train_robust_normalize", "clip_-5_5"],
        "retarget": {"gripper_distance_palm_width_range": [0.05, 1.50]},
        "source_files_sha256": source_hashes,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "data_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    hashes = {
        name: _sha256(output_dir / name)
        for name in ("normalization.npz", "data_contract.json", "split_manifest.json")
    }
    (output_dir / "contract_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--force-history-frames", type=int, choices=(0, 16), default=0)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-source-hash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_dir = args.annotation_dir or args.input_dir / "final_annotations"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = scan_records(args.input_dir, annotation_dir)
    if not records:
        raise FileNotFoundError(f"no OpenTouch .hdf5 files found under {args.input_dir}")
    assign_connected_groups(records)
    stratified_group_split(records, args.seed)
    stats = convert_records(records, args.output_dir)
    windows = build_windows(records, args.force_history_frames, args.window_stride)
    write_contract(records, windows, stats, args.output_dir, args.input_dir,
                   args.force_history_frames, args.seed, not args.skip_source_hash)
    print(json.dumps({"demos": len(records), "windows": {k: len(v) for k, v in windows.items()}}, indent=2))


if __name__ == "__main__":
    main()
