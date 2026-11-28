from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fastwam.evaluation.opentouch_metrics import add_action_metrics, add_force_metrics, finalize_pairs
from fastwam.trainer import Wan22Trainer


class _FakeAccelerator:
    device = torch.device("cpu")

    def __init__(self, *, is_main_process=True, process_index=0, num_processes=1):
        self.is_main_process = is_main_process
        self.process_index = process_index
        self.num_processes = num_processes

    def gather(self, tensor):
        return tensor

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()

    def wait_for_everyone(self):
        return None


class _FakeDit:
    training = False
    action_reads_future_force = True


class _FakeModel:
    def __init__(self, force_enabled):
        self.force_enabled = force_enabled
        self.dit = _FakeDit()
        self.device = torch.device("cpu")
        self.loss_lambda_video = 1.0
        self.loss_lambda_action = 1.0
        self.loss_lambda_force = 1.0
        self.action_calls = 0
        self.force_calls = 0

    def eval(self):
        self.dit.training = False
        return self

    def training_loss(self, sample):
        return torch.tensor(1.0), {"loss_action": 0.5, "loss_video": 0.25, "force_loss": 0.25}

    def infer_action(self, **kwargs):
        self.action_calls += 1
        return {"action": torch.zeros(32, 7)}

    def infer_action_force(self, **kwargs):
        self.force_calls += 1
        return {
            "action_normalized": torch.zeros(32, 7),
            "future_force_normalized": torch.zeros(32, 2),
        }


class _FakeDataset:
    is_opentouch_force_dataset = True
    force_history_frames = 0
    force_observed_frames = 1
    split = "val"
    contract_hashes = {"normalization.npz": "fake"}

    def __init__(self, normalization_path, history=0, invalid=False):
        self.normalization_path = normalization_path
        self.force_history_frames = history
        self.force_observed_frames = history + 1
        self.invalid = invalid

    def __len__(self):
        return 1

    def fixed_eval_indices(self, count, **kwargs):
        return [0] if count else []

    def denormalize_action(self, value):
        return value + 1.0

    def denormalize_force(self, value):
        return value + 2.0

    def __getitem__(self, index):
        force_frames = self.force_observed_frames + 32
        action = torch.zeros(32, 7)
        action_is_pad = torch.zeros(32, dtype=torch.bool)
        force = torch.zeros(force_frames, 2)
        force_valid = torch.ones(force_frames, 2, dtype=torch.bool)
        if self.invalid:
            action[1] = float("nan")
            action_is_pad[1] = True
            force[self.force_observed_frames + 1, 1] = float("nan")
            force_valid[self.force_observed_frames + 1, 1] = False
        return {
            "video": torch.zeros(3, 9, 16, 16),
            "prompt": "test",
            "action": action,
            "action_is_pad": action_is_pad,
            "proprio": torch.zeros(8),
            "force": force,
            "force_valid": force_valid,
            "force_is_pad": torch.zeros(force_frames, 2, dtype=torch.bool),
            "force_observed_frames": self.force_observed_frames,
            "image_is_pad": torch.zeros(9, dtype=torch.bool),
            "sequence_id": "sequence",
            "current_frame": 0,
            "split_group": "group",
        }


def _cfg(stage, *, print_predictions=True, print_steps=32, selection="activity_random"):
    return OmegaConf.create({
        "seed": 42,
        "eval_num_inference_steps": 2,
        "model": {"init_mode": "scratch"},
        "training": {"stage": stage, "finetune_video": False},
        "evaluation": {
            "offline_loss_samples": 1,
            "offline_generation_samples": 1,
            "print_predictions": print_predictions,
            "print_prediction_samples": 1,
            "print_prediction_steps": print_steps,
            "print_prediction_precision": 5,
            "print_prediction_selection": selection,
            "print_prediction_activity_top_fraction": 0.5,
        },
    })


def _prediction_log(info_mock):
    blocks = []
    for call in info_mock.call_args_list:
        if len(call.args) >= 2 and call.args[0] == "%s":
            blocks.append(str(call.args[1]))
    return "\n".join(blocks)


def test_masked_metrics_return_finite_zero_for_all_invalid():
    pairs = {}
    action_mse = add_action_metrics(
        pairs,
        prefix="",
        pred_normalized=torch.ones(2, 7),
        target_normalized=torch.zeros(2, 7),
        valid_steps=torch.zeros(2, dtype=torch.bool),
        denormalize=lambda value: value,
    )
    add_force_metrics(
        pairs,
        prefix="",
        pred_normalized=torch.ones(2, 2),
        target_normalized=torch.zeros(2, 2),
        effective_valid=torch.zeros(2, 2, dtype=torch.bool),
        denormalize=lambda value: value,
    )
    assert action_mse is None
    assert finalize_pairs(pairs) == {}


def test_action_only_opentouch_evaluation_never_generates_video(tmp_path):
    model = _FakeModel(force_enabled=False)
    runner = Wan22Trainer.evaluation_only(
        model, _FakeDataset(tmp_path / "normalization.npz"), cfg=_cfg("joint"),
        accelerator=_FakeAccelerator(), output_dir=str(tmp_path),
    )
    with patch("fastwam.trainer.logger.info") as info:
        metrics = runner.evaluate()
    log = _prediction_log(info)
    assert model.action_calls == 1
    assert model.force_calls == 0
    assert metrics["generation_action_mse_normalized"] == 0.0
    assert "ACTION canonical" in log
    assert "selection=activity_random" in log
    assert "future_force_activity_score=0.0000" in log
    assert "candidate_count=1 eligible_count=1" in log
    assert "pred=[1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000, 1.00000]" in log
    assert "FORCE relative_pressure_3072_minus_adc" not in log


def test_force_warmup_reports_force_without_action_metric(tmp_path):
    model = _FakeModel(force_enabled=True)
    runner = Wan22Trainer.evaluation_only(
        model, _FakeDataset(tmp_path / "normalization.npz"), cfg=_cfg("force_warmup"),
        accelerator=_FakeAccelerator(), output_dir=str(tmp_path),
    )
    with patch("fastwam.trainer.logger.info") as info:
        metrics = runner.evaluate()
    log = _prediction_log(info)
    assert model.force_calls == 1
    assert model.loss_lambda_video == 0.0
    assert model.loss_lambda_action == 0.0
    assert model.loss_lambda_force == 1.0
    assert metrics["val_loss"] == 1.0
    assert metrics["generation_force_mse_normalized"] == 0.0
    assert "generation_action_mse_normalized" not in metrics
    assert "ACTION canonical" not in log
    assert "FORCE relative_pressure_3072_minus_adc" in log
    assert "pred=[2.00000, 2.00000]" in log


def test_force_joint_h0_and_h16_use_action_force_path(tmp_path):
    for history, expected_calls in ((0, 3), (16, 4)):
        model = _FakeModel(force_enabled=True)
        runner = Wan22Trainer.evaluation_only(
            model, _FakeDataset(tmp_path / "normalization.npz", history=history), cfg=_cfg("joint"),
            accelerator=_FakeAccelerator(), output_dir=str(tmp_path / f"h{history}"),
        )
        with patch("fastwam.trainer.logger.info") as info:
            metrics = runner.evaluate()
        log = _prediction_log(info)
        assert model.loss_lambda_video == 1.0
        assert model.loss_lambda_action == 1.0
        assert model.action_calls == 0
        assert model.force_calls == expected_calls
        assert metrics["generation_action_mse_normalized"] == 0.0
        assert metrics["generation_force_mse_normalized"] == 0.0
        assert "ACTION canonical" in log
        assert "FORCE relative_pressure_3072_minus_adc" in log
        assert f"history_frames={history}" in log
        assert f"observed_force_frames={history + 1}" in log


def test_prediction_log_masks_invalid_values_and_limits_steps(tmp_path):
    model = _FakeModel(force_enabled=True)
    runner = Wan22Trainer.evaluation_only(
        model, _FakeDataset(tmp_path / "normalization.npz", invalid=True),
        cfg=_cfg("joint", print_steps=2), accelerator=_FakeAccelerator(),
        output_dir=str(tmp_path),
    )
    with patch("fastwam.trainer.logger.info") as info:
        runner.evaluate()
    log = _prediction_log(info)
    action_line = next(line for line in log.splitlines() if line.startswith("action step=01"))
    force_line = next(line for line in log.splitlines() if line.startswith("force step=01"))
    assert "valid=0" in action_line and "pred=[NA, NA, NA, NA, NA, NA, NA]" in action_line
    assert "valid=[1,0]" in force_line and "gt=[2.00000, NA]" in force_line
    assert "action step=02" not in log
    assert "force step=02" not in log


def test_prediction_log_can_be_disabled_and_only_main_process_prints(tmp_path):
    for enabled, accelerator in (
        (False, _FakeAccelerator()),
        (True, _FakeAccelerator(is_main_process=False)),
    ):
        model = _FakeModel(force_enabled=True)
        runner = Wan22Trainer.evaluation_only(
            model, _FakeDataset(tmp_path / "normalization.npz"),
            cfg=_cfg("joint", print_predictions=enabled), accelerator=accelerator,
            output_dir=str(tmp_path / f"{enabled}_{accelerator.is_main_process}"),
        )
        with patch("fastwam.trainer.logger.info") as info:
            runner.evaluate()
        assert "[opentouch-prediction]" not in _prediction_log(info)


def test_obsolete_contact_preview_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="obsolete"):
        Wan22Trainer.evaluation_only(
            _FakeModel(force_enabled=True),
            _FakeDataset(tmp_path / "normalization.npz"),
            cfg=_cfg("joint", selection="contact_random"),
            accelerator=_FakeAccelerator(),
            output_dir=str(tmp_path),
        )


def test_activity_random_preview_is_reproducible_and_uses_top_fraction():
    runner = Wan22Trainer.__new__(Wan22Trainer)
    runner.seed = 42
    runner.global_step = 2000
    runner.print_prediction_samples = 2
    runner.print_prediction_selection = "activity_random"
    runner.print_prediction_activity_top_fraction = 0.5
    candidates = [
        {"name": "mostly_constant", "activity_score": 0.05},
        {"name": "activity_a", "activity_score": 0.50},
        {"name": "activity_b", "activity_score": 0.75},
        {"name": "activity_c", "activity_score": 0.30},
    ]
    selected_a, eligible_a = runner._select_prediction_previews(candidates)
    selected_b, eligible_b = runner._select_prediction_previews(candidates)
    assert eligible_a == eligible_b == 2
    assert [item["name"] for item in selected_a] == [item["name"] for item in selected_b]
    assert all(item["activity_score"] >= 0.50 for item in selected_a)


def test_activity_random_all_constant_falls_back_to_stable_highest():
    runner = Wan22Trainer.__new__(Wan22Trainer)
    runner.seed = 42
    runner.global_step = 2000
    runner.print_prediction_samples = 1
    runner.print_prediction_selection = "activity_random"
    runner.print_prediction_activity_top_fraction = 0.5
    selected, eligible = runner._select_prediction_previews([
        {"name": "first", "activity_score": 0.0},
        {"name": "second", "activity_score": 0.0},
        {"name": "third", "activity_score": 0.0},
    ])
    assert eligible == 1
    assert selected[0]["name"] == "first"


def test_fixed_max_activity_and_random_preview_modes():
    runner = Wan22Trainer.__new__(Wan22Trainer)
    runner.seed = 9
    runner.global_step = 2000
    runner.print_prediction_samples = 1
    runner.print_prediction_activity_top_fraction = 0.5
    candidates = [
        {"name": "first", "activity_score": 0.10},
        {"name": "best", "activity_score": 0.80},
        {"name": "other", "activity_score": 0.40},
    ]
    runner.print_prediction_selection = "fixed"
    assert runner._select_prediction_previews(candidates)[0][0]["name"] == "first"
    runner.print_prediction_selection = "max_activity"
    assert runner._select_prediction_previews(candidates)[0][0]["name"] == "best"
    runner.print_prediction_selection = "random"
    first = runner._select_prediction_previews(candidates)[0][0]["name"]
    second = runner._select_prediction_previews(candidates)[0][0]["name"]
    assert first == second


def test_future_force_activity_score_uses_validity_and_padding():
    force = torch.zeros(1, 5, 2)
    valid = torch.ones(1, 5, 2, dtype=torch.bool)
    is_pad = torch.zeros(1, 5, 2, dtype=torch.bool)
    force[0, 1:, 0] = torch.tensor([0.1, 0.9, 0.5, 0.7])
    force[0, 1:, 1] = torch.tensor([0.2, 0.4, 0.8, 1.0])
    valid[0, 2, 0] = False
    is_pad[0, 4, 1] = True
    sample = {
        "force": force,
        "force_valid": valid,
        "force_is_pad": is_pad,
        "force_observed_frames": 1,
    }
    # thumb valid range: 0.7-0.1=0.6; index valid range: 0.8-0.2=0.6.
    assert np.isclose(Wan22Trainer._future_force_activity_score(sample), 0.6)


def test_bridge_diagnostics_are_aggregated_and_printed():
    gate = torch.nn.Parameter(torch.tensor(0.2))
    gate.grad = torch.tensor(0.03)

    class _DiagnosticDit:
        force_to_backbone = {1: 8}
        bridge_gates = torch.nn.ParameterDict({"1": gate})

        def bridge_diagnostics(self):
            return {
                1: {
                    "raw": gate.detach(),
                    "tanh": torch.tanh(gate.detach()),
                    "grad": gate.grad.detach(),
                    "residual_rms": torch.tensor(2.0),
                }
            }

    runner = Wan22Trainer.__new__(Wan22Trainer)
    runner.accelerator = _FakeAccelerator()
    runner.print_bridge_diagnostics = True
    runner.global_step = 100
    runner.training_stage = "joint"
    snapshot = runner._collect_bridge_diagnostics(SimpleNamespace(dit=_DiagnosticDit()))
    assert np.isclose(snapshot[1]["effective_rms"], abs(np.tanh(0.2)) * 2.0)
    assert snapshot[1]["trainable"]
    with patch("fastwam.trainer.logger.info") as info:
        runner._log_bridge_diagnostic_snapshot(snapshot)
    log = _prediction_log(info)
    assert "[bridge-gate] step=100 stage=joint F1->VA8 trainable=1" in log
    assert "grad_after_clip=" in log
    assert "residual_rms=" in log
    assert "effective_rms=" in log

    runner.print_bridge_diagnostics = False
    with patch("fastwam.trainer.logger.info") as info:
        runner._log_bridge_diagnostic_snapshot(snapshot)
    info.assert_not_called()

    gate.grad = None
    gate.data.zero_()
    gate.requires_grad_(False)
    runner.training_stage = "force_warmup"
    warmup = runner._collect_bridge_diagnostics(SimpleNamespace(dit=_DiagnosticDit()))
    assert not warmup[1]["trainable"]
    assert warmup[1]["raw"] == 0.0
    assert warmup[1]["grad"] == 0.0
