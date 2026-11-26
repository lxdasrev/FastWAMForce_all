"""PyTorch dataset for the cache produced by :mod:`preprocess`."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageOps

from .canonical import normalize_force


class OpenTouchForceDataset(torch.utils.data.Dataset):
    """Return synchronized 9-frame video, 32 actions and 33/49 force frames."""

    is_opentouch_force_dataset = True

    def __init__(
        self,
        cache_dir,
        split="train",
        video_size=(224, 224),
        force_history_frames=0,
        text_embedding_cache_dir=None,
        use_text_embed_cache=True,
        context_len=128,
        strict_contract=True,
        source_root_override=None,
        pretrained_norm_stats=None,
    ):
        del pretrained_norm_stats  # Normalization is contract-bound, never fit by the val set.
        self.cache_dir = Path(cache_dir)
        self.split = str(split)
        self.video_size = tuple(int(v) for v in video_size)
        self.force_history_frames = int(force_history_frames)
        if self.force_history_frames not in (0, 16):
            raise ValueError("force_history_frames must be 0 or 16")
        self.force_observed_frames = self.force_history_frames + 1
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.use_text_embed_cache = bool(use_text_embed_cache)
        self.context_len = int(context_len)
        self.source_root_override = Path(source_root_override) if source_root_override else None
        self._h5_handles = {}

        manifest_path = self.cache_dir / "split_manifest.json"
        contract_path = self.cache_dir / "data_contract.json"
        stats_path = self.cache_dir / "normalization.npz"
        hashes_path = self.cache_dir / "contract_hashes.json"
        for path in (manifest_path, contract_path, stats_path, hashes_path):
            if not path.exists():
                raise FileNotFoundError(f"missing OpenTouch cache artifact: {path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if self.split not in self.manifest["windows"]:
            raise ValueError(f"unknown split {self.split!r}")
        self.windows = self.manifest["windows"][self.split]
        self.records = {
            record["sequence_id"]: record for record in self.manifest.get("records", [])
        }
        self.normalization_path = stats_path
        self.stats = dict(np.load(stats_path))
        self.contract_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        if strict_contract:
            self._validate_hashes(hashes_path)
            contract_history = int(self.contract["force_history_frames"])
            if contract_history != self.force_history_frames:
                raise ValueError(
                    f"force history mismatch: cache={contract_history}, requested={self.force_history_frames}; "
                    "build a matching cache instead of silently padding"
                )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_hashes(self, hashes_path: Path) -> None:
        expected = json.loads(hashes_path.read_text(encoding="utf-8"))
        for name, digest in expected.items():
            actual = self._sha256(self.cache_dir / name)
            if actual != digest:
                raise ValueError(f"OpenTouch contract hash mismatch for {name}: {actual} != {digest}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_handles"] = {}
        return state

    def __del__(self):
        for handle in getattr(self, "_h5_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def _stat_like(value, reference):
        if isinstance(reference, torch.Tensor):
            return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        return np.asarray(value, dtype=np.asarray(reference).dtype)

    def denormalize_action(self, action):
        """Invert the checkpoint-bound seven-dimensional action normalization."""
        mean = self._stat_like(self.stats["action_mean"], action)
        scale = self._stat_like(self.stats["action_scale"], action)
        return action * scale + mean

    def denormalize_force(self, force):
        """Invert robust/asinh normalization to two-finger relative pressure."""
        median = self._stat_like(self.stats["force_median"], force)
        scale = self._stat_like(self.stats["force_scale"], force)
        asinh_scale = self._stat_like(self.stats["force_asinh_scale"], force)
        deadzone = self._stat_like(self.stats["force_deadzone"], force)
        compressed = force * scale + median
        if isinstance(force, torch.Tensor):
            return torch.clamp(torch.sinh(compressed) * asinh_scale + deadzone, min=0.0)
        return np.maximum(np.sinh(compressed) * asinh_scale + deadzone, 0.0)

    def _window_group(self, item: dict) -> str:
        record = self.records.get(item["sequence_id"], {})
        return str(item.get("split_group", record.get("split_group", item["sequence_id"])))

    def _window_has_contact(self, item: dict) -> bool:
        record = self.records.get(item["sequence_id"], {})
        cache_file = item.get("cache_file", record.get("cache_file"))
        if cache_file is None:
            return False
        current = int(item["current_frame"])
        start = current - self.force_history_frames
        stop = current + 33
        with np.load(self.cache_dir / cache_file) as cache:
            force = np.asarray(cache["force_raw"][start:stop], dtype=np.float32)
            valid = np.asarray(cache["force_valid"][start:stop], dtype=bool)
        deadzone = np.asarray(self.stats["force_deadzone"], dtype=np.float32)
        return bool(np.any(valid & (force > deadzone)))

    def fixed_eval_indices(
        self,
        count: int,
        *,
        seed: int = 42,
        contact_only: bool = False,
        unique_groups: bool = False,
    ) -> list[int]:
        """Select deterministic, sequence-group-stratified validation indices."""
        count = max(0, min(int(count), len(self.windows)))
        if count == 0:
            return []
        grouped: dict[str, list[int]] = {}
        for index, item in enumerate(self.windows):
            grouped.setdefault(self._window_group(item), []).append(index)

        def stable_key(value: str) -> str:
            return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()

        group_names = sorted(grouped, key=stable_key)
        for group in group_names:
            grouped[group].sort(key=lambda index: stable_key(f"{group}:{index}"))
        selected: list[int] = []
        selected_groups: set[str] = set()
        depth = 0
        while len(selected) < count:
            added = False
            for group in group_names:
                if unique_groups and group in selected_groups:
                    continue
                candidates = grouped[group]
                if depth >= len(candidates):
                    continue
                index = candidates[depth]
                if not contact_only or self._window_has_contact(self.windows[index]):
                    selected.append(index)
                    selected_groups.add(group)
                    if len(selected) == count:
                        break
                added = True
            if not added:
                break
            depth += 1
        return selected

    def _source_path(self, item: dict) -> Path:
        source = Path(item["source_file"])
        if self.source_root_override is not None:
            source = self.source_root_override / source.name
        return source

    def _handle(self, path: Path):
        key = str(path)
        handle = self._h5_handles.get(key)
        if handle is None:
            os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
            handle = h5py.File(path, "r", locking=False)
            self._h5_handles[key] = handle
        return handle

    def _decode_video(self, item: dict, indices: np.ndarray) -> torch.Tensor:
        dataset = self._handle(self._source_path(item))[f"data/{item['demo_key']}/rgb_images_jpeg"]
        frames = []
        height, width = self.video_size
        for index in indices:
            payload = dataset[int(index)]
            if isinstance(payload, np.ndarray):
                payload = payload.tobytes()
            elif isinstance(payload, np.void):
                payload = bytes(payload)
            with Image.open(io.BytesIO(payload)) as image:
                image = ImageOps.fit(image.convert("RGB"), (width, height), method=Image.Resampling.BILINEAR)
                frames.append(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
        array = np.stack(frames, axis=0)  # T,H,W,C
        return torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is required when use_text_embed_cache=true")
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = Path(self.text_embedding_cache_dir) / f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing text embedding cache: {path}")
        payload = torch.load(path, map_location="cpu")
        context = payload["context"]
        mask = payload["mask"].bool()
        if tuple(context.shape) != (self.context_len, 4096) or tuple(mask.shape) != (self.context_len,):
            raise ValueError(f"bad cached context shapes in {path}: {context.shape}, {mask.shape}")
        context = context.clone()
        context[~mask] = 0
        return context, torch.ones_like(mask)

    def __getitem__(self, index):
        item = dict(self.windows[index])
        record = self.records.get(item["sequence_id"])
        if record is not None:
            for key in ("source_file", "demo_key", "cache_file", "split_group"):
                item.setdefault(key, record[key])
            item.setdefault(
                "prompt",
                f"Use the gripper to {str(record['action_label']).lower()} the {str(record['object_name']).lower()}.",
            )
        current = int(item["current_frame"])
        with np.load(self.cache_dir / item["cache_file"]) as cache:
            action = np.asarray(cache["action"][current:current + 32], dtype=np.float32)
            action_valid = np.asarray(cache["action_valid"][current:current + 32], dtype=bool)
            proprio = np.asarray(cache["proprio"][current], dtype=np.float32)
            pose_valid = bool(cache["pose_valid"][current])
            force_start = current - self.force_history_frames
            force_stop = current + 33
            force = np.asarray(cache["force_raw"][force_start:force_stop], dtype=np.float32)
            force_valid = np.asarray(cache["force_valid"][force_start:force_stop], dtype=bool)
        if len(action) != 32 or len(force) != self.force_observed_frames + 32:
            raise RuntimeError(f"manifest window is incomplete: {item}")
        action = (action - self.stats["action_mean"]) / self.stats["action_scale"]
        proprio = (proprio - self.stats["proprio_mean"]) / self.stats["proprio_scale"]
        action[~action_valid] = 0
        if not pose_valid:
            proprio.fill(0)
        force = normalize_force(force, self.stats)
        force[~force_valid] = 0
        video_indices = current + np.arange(0, 33, 4)
        video = self._decode_video(item, video_indices)
        prompt = item["prompt"]
        result = {
            "video": video,
            "action": torch.from_numpy(action),
            "action_chunk": torch.from_numpy(action),
            "action_is_pad": torch.from_numpy(~action_valid),
            "proprio": torch.from_numpy(proprio),
            "force": torch.from_numpy(force),
            "force_valid": torch.from_numpy(force_valid),
            "force_is_pad": torch.zeros_like(torch.from_numpy(force_valid)),
            "force_observed_frames": self.force_observed_frames,
            "prompt": prompt,
            "image_is_pad": torch.zeros(9, dtype=torch.bool),
            "sequence_id": item["sequence_id"],
            "current_frame": current,
            "split_group": self._window_group(item),
        }
        if self.use_text_embed_cache:
            result["context"], result["context_mask"] = self._get_cached_text_context(prompt)
        return result
