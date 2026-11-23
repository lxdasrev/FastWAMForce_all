"""Deterministic tactile ablations and sequence-group bootstrap statistics."""

from __future__ import annotations

from contextlib import contextmanager

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
