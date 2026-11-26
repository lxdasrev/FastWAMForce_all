"""Pure metric helpers for OpenTouch action/force evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import torch


MetricPairs = dict[str, tuple[float, float]]


def add_pair(pairs: MetricPairs, name: str, numerator: float, denominator: float) -> None:
    old_num, old_den = pairs.get(name, (0.0, 0.0))
    pairs[name] = (old_num + float(numerator), old_den + float(denominator))


def add_action_metrics(
    pairs: MetricPairs,
    *,
    prefix: str,
    pred_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    valid_steps: torch.Tensor,
    denormalize,
) -> float | None:
    """Accumulate normalized and canonical action errors; return sample MSE."""
    pred = pred_normalized.detach().float().cpu()
    target = target_normalized.detach().float().cpu()
    valid = valid_steps.detach().bool().cpu()
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[-1] != 7:
        raise ValueError(f"action tensors must be identical [T,7], got {pred.shape} and {target.shape}")
    if valid.shape != pred.shape[:1]:
        raise ValueError(f"action valid mask must be [T], got {valid.shape} for {pred.shape}")
    if not bool(valid.any()):
        return None

    diff = pred - target
    expanded = valid[:, None].expand_as(diff)
    squared = diff.square()
    add_pair(pairs, f"{prefix}action_mse_normalized", squared[expanded].sum().item(), expanded.sum().item())

    pred_canonical = denormalize(pred)
    target_canonical = denormalize(target)
    canonical_diff = pred_canonical - target_canonical
    translation_mask = valid[:, None].expand(-1, 3)
    rotation_mask = valid[:, None].expand(-1, 3)
    add_pair(
        pairs,
        f"{prefix}translation_mse_canonical",
        canonical_diff[:, :3].square()[translation_mask].sum().item(),
        translation_mask.sum().item(),
    )
    add_pair(
        pairs,
        f"{prefix}rotation_mse_canonical",
        canonical_diff[:, 3:6].square()[rotation_mask].sum().item(),
        rotation_mask.sum().item(),
    )
    add_pair(
        pairs,
        f"{prefix}gripper_mae_canonical",
        canonical_diff[:, 6].abs()[valid].sum().item(),
        valid.sum().item(),
    )
    return float(squared[expanded].mean().item())


def add_force_metrics(
    pairs: MetricPairs,
    *,
    prefix: str,
    pred_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    effective_valid: torch.Tensor,
    denormalize,
) -> None:
    pred = pred_normalized.detach().float().cpu()
    target = target_normalized.detach().float().cpu()
    valid = effective_valid.detach().bool().cpu()
    if pred.shape != target.shape or pred.shape != valid.shape or pred.ndim != 2 or pred.shape[-1] != 2:
        raise ValueError("force prediction, target and validity must be identical [T,2] tensors")
    if not bool(valid.any()):
        return
    add_pair(
        pairs,
        f"{prefix}force_mse_normalized",
        (pred - target).square()[valid].sum().item(),
        valid.sum().item(),
    )
    pred_pressure = denormalize(pred)
    target_pressure = denormalize(target)
    add_pair(
        pairs,
        f"{prefix}force_mae_relative_pressure",
        (pred_pressure - target_pressure).abs()[valid].sum().item(),
        valid.sum().item(),
    )


def finalize_pairs(pairs: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    return {
        name: (float(numerator) / float(denominator) if denominator > 0 else 0.0)
        for name, (numerator, denominator) in pairs.items()
    }
