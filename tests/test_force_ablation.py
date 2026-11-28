import numpy as np
import torch

from fastwam.evaluation.force_ablation import (
    ablate_observed_force,
    paired_group_bootstrap,
    summarize_bootstrap_outputs,
)


def test_force_ablation_shapes_and_current_frame_preservation():
    force = torch.arange(2 * 17 * 2).reshape(2, 17, 2).float()
    valid = torch.ones_like(force, dtype=torch.bool)
    shuffled, shuffled_valid = ablate_observed_force(
        force, valid, "time_shuffle", generator=torch.Generator().manual_seed(2)
    )
    assert shuffled.shape == force.shape
    assert torch.equal(shuffled[:, -1], force[:, -1])
    assert torch.equal(shuffled_valid, valid)


def test_group_bootstrap_success_rule():
    result = paired_group_bootstrap(
        np.array([0.9, 0.8, 0.7]), np.array([1.0, 0.9, 0.8]), np.array(["a", "b", "c"]),
        samples=1000,
    )
    assert result["relative_improvement"] > 0.02
    assert result["ci95_low"] > 0
    assert result["passes_2pct"]


def test_bootstrap_output_summary(tmp_path):
    np.savez_compressed(
        tmp_path / "opentouch_generation_step_000001_rank_000.npz",
        split_group=np.asarray(["a", "b", "c"]),
        correct_action_mse=np.asarray([0.8, 0.7, 0.6]),
        zero_action_mse=np.asarray([1.0, 0.9, 0.8]),
        batch_shuffle_action_mse=np.asarray([1.1, 1.0, 0.9]),
        future_force_activity_score=np.asarray([0.1, 0.5, 0.9]),
    )
    result = summarize_bootstrap_outputs(tmp_path, samples=1000, seed=42)
    assert result["num_groups"] == 3
    assert result["passes_success_criterion"]
    assert result["high_activity"]["num_examples"] == 2
    assert result["high_activity"]["passes_success_criterion"]
