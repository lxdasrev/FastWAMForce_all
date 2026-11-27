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
    DATA_VERSION,
    DEFAULT_TWO_FINGER_TAXELS,
    calibrate_and_clean_force,
    canonicalize_hand_trajectory,
    normalize_force,
    pressure_to_two_finger,
)

FUTURE_FRAMES = 32
VIDEO_STRIDE = 4
VIDEO_FRAMES = 9
MIN_CALIBRATION_FRAMES = 8
CALIBRATION_FALLBACK_FRACTION = 0.20
FORCE_MAD_SCALE = 1.4826
FORCE_NOISE_FLOOR = 5.0
DEADBAND_LOW_SIGMA = 2.0
DEADBAND_HIGH_SIGMA = 4.0
FORCE_UPPER_PERCENTILE = 99.5


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


def _fit_force_normalization(force: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Fit the v2 positive-pressure transform from training data only."""
    knee = np.empty(2, dtype=np.float32)
    upper = np.empty(2, dtype=np.float32)
    for finger in range(2):
        positive = force[:, finger][valid[:, finger] & (force[:, finger] > 0)]
        if positive.size == 0:
            raise ValueError(f"cannot fit force normalization: no positive training values for finger {finger}")
        knee[finger] = max(float(np.median(positive)), 1e-6)
        upper[finger] = max(
            float(np.percentile(positive, FORCE_UPPER_PERCENTILE)),
            float(knee[finger]) + 1e-6,
        )
    normalization_max = np.arcsinh(upper / knee).astype(np.float32)
    return {
        "force_knee": knee,
        "force_upper": upper,
        "force_normalization_max": normalization_max,
        "force_clip_min": np.asarray(0.0, dtype=np.float32),
        "force_clip_max": np.asarray(1.0, dtype=np.float32),
    }


def convert_records(records: list[DemoRecord], output_dir: Path) -> dict[str, np.ndarray]:
    cache_dir = output_dir / "trajectories"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_action, train_action_valid = [], []
    train_proprio, train_pose_valid = [], []
    train_force, train_force_valid = [], []
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
                force_clean, force_contact, force_baseline, force_noise = calibrate_and_clean_force(
                    force_raw,
                    force_valid,
                    onset_idx=record.onset_idx,
                    min_calibration_frames=MIN_CALIBRATION_FRAMES,
                    fallback_fraction=CALIBRATION_FALLBACK_FRACTION,
                    mad_scale=FORCE_MAD_SCALE,
                    noise_floor=FORCE_NOISE_FLOOR,
                    deadband_low_sigma=DEADBAND_LOW_SIGMA,
                    deadband_high_sigma=DEADBAND_HIGH_SIGMA,
                )
                safe_name = record.sequence_id.replace("::", "__") + ".npz"
                cache_path = cache_dir / safe_name
                np.savez_compressed(
                    cache_path,
                    proprio=trajectory.proprio,
                    action=trajectory.action,
                    pose_valid=trajectory.pose_valid,
                    action_valid=trajectory.action_valid,
                    force_raw=force_raw,
                    force_clean=force_clean,
                    force_valid=force_valid,
                    force_contact=force_contact,
                    force_baseline=force_baseline,
                    force_noise=force_noise,
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
                    train_force.append(force_clean)
                    train_force_valid.append(force_valid)
    actions = np.concatenate(train_action)
    action_valid = np.concatenate(train_action_valid)
    proprios = np.concatenate(train_proprio)
    pose_valid = np.concatenate(train_pose_valid)
    force = np.concatenate(train_force)
    force_valid = np.concatenate(train_force_valid)
    action_mean, action_scale = _safe_stats(actions, action_valid, 7)
    proprio_mean, proprio_scale = _safe_stats(proprios, pose_valid, 8)
    result = {
        "action_mean": action_mean,
        "action_scale": action_scale,
        "proprio_mean": proprio_mean,
        "proprio_scale": proprio_scale,
        "pressure_adc_max": np.asarray(3072.0, dtype=np.float32),
    }
    result.update(_fit_force_normalization(force, force_valid))
    return result


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


def build_force_quality_report(
    records: list[DemoRecord],
    windows: dict[str, list[dict]],
    output_dir: Path,
    stats: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Summarize signal preservation without duplicating frames across windows."""
    record_by_id = {record.sequence_id: record for record in records}
    report: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        valid_count = np.zeros(2, dtype=np.int64)
        zero_count = np.zeros(2, dtype=np.int64)
        saturation_count = np.zeros(2, dtype=np.int64)
        contact_windows = 0
        constant_contact_windows = 0
        future_std_sum = np.zeros(2, dtype=np.float64)
        future_amplitude_sum = np.zeros(2, dtype=np.float64)
        future_stat_count = np.zeros(2, dtype=np.int64)

        split_records = [record for record in records if record.split == split]
        for record in split_records:
            with np.load(output_dir / record.cache_file) as cache:
                clean = np.asarray(cache["force_clean"], dtype=np.float32)
                valid = np.asarray(cache["force_valid"], dtype=bool)
            normalized = normalize_force(clean, stats)
            valid_count += valid.sum(axis=0)
            zero_count += (valid & (clean <= 0)).sum(axis=0)
            saturation_count += (valid & (normalized >= 1.0 - 1e-6)).sum(axis=0)

        windows_by_sequence: dict[str, list[int]] = defaultdict(list)
        for item in windows[split]:
            windows_by_sequence[item["sequence_id"]].append(int(item["current_frame"]))
        for sequence_id, current_frames in windows_by_sequence.items():
            record = record_by_id[sequence_id]
            with np.load(output_dir / record.cache_file) as cache:
                clean = np.asarray(cache["force_clean"], dtype=np.float32)
                valid = np.asarray(cache["force_valid"], dtype=bool)
            for current in current_frames:
                future = clean[current + 1:current + FUTURE_FRAMES + 1]
                future_valid = valid[current + 1:current + FUTURE_FRAMES + 1]
                has_contact = bool(np.any(future_valid & (future > 0)))
                contact_windows += int(has_contact)
                finger_constant = []
                for finger in range(2):
                    values = future[:, finger][future_valid[:, finger]]
                    if values.size:
                        amplitude = float(np.ptp(values))
                        future_std_sum[finger] += float(np.std(values))
                        future_amplitude_sum[finger] += amplitude
                        future_stat_count[finger] += 1
                        finger_constant.append(amplitude <= 1e-6)
                    else:
                        finger_constant.append(True)
                constant_contact_windows += int(has_contact and all(finger_constant))

        window_count = len(windows[split])
        valid_den = np.maximum(valid_count, 1)
        stat_den = np.maximum(future_stat_count, 1)
        saturation_fraction = saturation_count / valid_den
        constant_contact_fraction = constant_contact_windows / max(contact_windows, 1)
        report[split] = {
            "valid_elements_per_finger": valid_count.tolist(),
            "zero_fraction_per_finger": (zero_count / valid_den).tolist(),
            "upper_saturation_fraction_per_finger": saturation_fraction.tolist(),
            "upper_saturation_below_5_percent_per_finger": (saturation_fraction < 0.05).tolist(),
            "window_count": window_count,
            "future_contact_window_count": contact_windows,
            "future_contact_window_fraction": contact_windows / max(window_count, 1),
            "constant_contact_window_count": constant_contact_windows,
            "constant_contact_window_fraction": constant_contact_fraction,
            "constant_contact_window_below_5_percent": constant_contact_fraction < 0.05,
            "mean_future_std_per_finger": (future_std_sum / stat_den).tolist(),
            "mean_future_amplitude_per_finger": (future_amplitude_sum / stat_den).tolist(),
        }
    return report


def write_contract(
    records: list[DemoRecord], windows: dict[str, list[dict]], stats: dict[str, np.ndarray],
    quality_report: dict[str, dict], output_dir: Path, input_dir: Path,
    history_frames: int, seed: int, hash_sources: bool,
) -> None:
    np.savez(output_dir / "normalization.npz", **stats)
    (output_dir / "force_quality_report.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )
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
        "force_semantics": "baseline_corrected_relative_pressure_from_adc_not_newtons",
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
        "pressure_pipeline": [
            "3072-raw",
            "valid_taxel_mean",
            "per_demo_per_finger_baseline_median",
            "per_demo_per_finger_mad_noise",
            "smooth_deadband_2sigma_to_4sigma",
            "train_positive_asinh_knee_median",
            "train_positive_upper_p99_5",
            "clip_0_1",
        ],
        "force_calibration": {
            "annotation_onset_min_frames": MIN_CALIBRATION_FRAMES,
            "fallback": "lowest_20_percent_total_relative_pressure_per_demo",
            "fallback_fraction": CALIBRATION_FALLBACK_FRACTION,
            "mad_scale": FORCE_MAD_SCALE,
            "noise_floor_adc": FORCE_NOISE_FLOOR,
            "deadband_low_sigma": DEADBAND_LOW_SIGMA,
            "deadband_high_sigma": DEADBAND_HIGH_SIGMA,
            "normalization_knee": "train_positive_median_per_finger",
            "normalization_upper_percentile": FORCE_UPPER_PERCENTILE,
            "normalization_range": [0.0, 1.0],
        },
        "retarget": {"gripper_distance_palm_width_range": [0.05, 1.50]},
        "source_files_sha256": source_hashes,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "data_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    hashes = {
        name: _sha256(output_dir / name)
        for name in (
            "normalization.npz", "data_contract.json", "split_manifest.json",
            "force_quality_report.json",
        )
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
    quality_report = build_force_quality_report(records, windows, args.output_dir, stats)
    write_contract(records, windows, stats, quality_report, args.output_dir, args.input_dir,
                   args.force_history_frames, args.seed, not args.skip_source_hash)
    print(json.dumps({
        "demos": len(records),
        "windows": {key: len(value) for key, value in windows.items()},
        "force_quality": quality_report,
    }, indent=2))


if __name__ == "__main__":
    main()
