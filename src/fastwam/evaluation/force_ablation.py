"""Deterministic tactile ablations and sequence-group bootstrap statistics."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def ablate_observed_force(
    force: torch.Tensor,
    valid: torch.Tensor,
    mode: str,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply correct/zero/batch-shuffle/time-shuffle observed-force ablations."""
    mode = str(mode).lower()
    x, mask = force.clone(), valid.clone()
    if mode == "correct":
        return x, mask
    if mode == "zero":
        return torch.zeros_like(x), mask
    if mode == "batch_shuffle":
        order = torch.randperm(x.shape[0], generator=generator, device=x.device)
        return x[order], mask[order]
    if mode == "time_shuffle":
        if x.shape[1] <= 1:
            return x, mask
        order = torch.randperm(x.shape[1] - 1, generator=generator, device=x.device)
        order = torch.cat((order, torch.tensor([x.shape[1] - 1], device=x.device)))
        return x[:, order], mask[:, order]
    raise ValueError(f"unknown force ablation mode: {mode}")


@contextmanager
def future_force_to_action(model, enabled: bool):
    """Temporarily enable/disable the F+ -> Action bridge keys."""
    mot = model.dit
    if not hasattr(mot, "action_reads_future_force"):
        raise TypeError("model does not contain ForceBridgeMoT")
    previous = mot.action_reads_future_force
    mot.action_reads_future_force = bool(enabled)
    try:
        yield
    finally:
        mot.action_reads_future_force = previous


def paired_group_bootstrap(
    correct_mse: np.ndarray,
    ablated_mse: np.ndarray,
    group_ids: np.ndarray,
    *,
    samples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap the relative MSE improvement over non-overlapping groups."""
    correct = np.asarray(correct_mse, dtype=np.float64)
    ablated = np.asarray(ablated_mse, dtype=np.float64)
    groups = np.asarray(group_ids)
    if not (correct.shape == ablated.shape == groups.shape):
        raise ValueError("correct_mse, ablated_mse and group_ids must have identical shapes")
    unique = np.unique(groups)
    if not len(unique):
        raise ValueError("bootstrap needs at least one sequence group")
    group_gain = np.asarray([
        (ablated[groups == group].mean() - correct[groups == group].mean())
        / max(ablated[groups == group].mean(), 1e-12)
        for group in unique
    ])
    rng = np.random.default_rng(seed)
    draws = group_gain[rng.integers(0, len(group_gain), size=(samples, len(group_gain)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "relative_improvement": float(group_gain.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "num_groups": int(len(unique)),
        "passes_2pct": bool(group_gain.mean() >= 0.02 and low > 0),
    }


def summarize_bootstrap_outputs(
    eval_dir: str | Path,
    *,
    samples: int = 10000,
    seed: int = 42,
) -> dict[str, object]:
    """Aggregate rank-local OpenTouch NPZ files and run paired group bootstraps."""
    paths = sorted(Path(eval_dir).glob("opentouch_generation_step_*_rank_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no OpenTouch generation results found in {eval_dir}")
    merged: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path) as payload:
            for name in payload.files:
                merged.setdefault(name, []).append(np.asarray(payload[name]))
    arrays = {name: np.concatenate(values, axis=0) for name, values in merged.items()}
    groups = arrays.get("split_group", np.asarray([]))
    summary: dict[str, object] = {
        "num_examples": int(len(groups)),
        "num_groups": int(len(np.unique(groups))),
        "bootstrap_samples": int(samples),
        "seed": int(seed),
        "ablations": {},
    }
    correct = arrays.get("correct_action_mse")
    if correct is None or not len(groups):
        summary["passes_success_criterion"] = False
        return summary
    for name in (
        "zero_action_mse",
        "batch_shuffle_action_mse",
        "time_shuffle_action_mse",
        "disable_future_force_to_action_action_mse",
    ):
        ablated = arrays.get(name)
        if ablated is None:
            continue
        valid = np.isfinite(correct) & np.isfinite(ablated)
        if not bool(valid.any()):
            continue
        summary["ablations"][name.removesuffix("_action_mse")] = paired_group_bootstrap(
            correct[valid], ablated[valid], groups[valid], samples=samples, seed=seed
        )
    required = [summary["ablations"].get(name) for name in ("zero", "batch_shuffle")]
    summary["passes_success_criterion"] = bool(
        all(item is not None and item["passes_2pct"] for item in required)
    )
    return summary
