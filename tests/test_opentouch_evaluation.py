from contextlib import nullcontext

import numpy as np
import torch
from omegaconf import OmegaConf

from fastwam.evaluation.opentouch_metrics import add_action_metrics, add_force_metrics, finalize_pairs
from fastwam.trainer import Wan22Trainer


class _FakeAccelerator:
    process_index = 0
    num_processes = 1
    device = torch.device("cpu")
    is_main_process = True

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

    def __init__(self, normalization_path, history=0):
        self.normalization_path = normalization_path
        self.force_history_frames = history
        self.force_observed_frames = history + 1

    def __len__(self):
        return 1

    def fixed_eval_indices(self, count, **kwargs):
        return [0] if count else []

    def denormalize_action(self, value):
        return value

    def denormalize_force(self, value):
        return value

    def __getitem__(self, index):
        force_frames = self.force_observed_frames + 32
        return {
            "video": torch.zeros(3, 9, 16, 16),
            "prompt": "test",
            "action": torch.zeros(32, 7),
            "action_is_pad": torch.zeros(32, dtype=torch.bool),
            "proprio": torch.zeros(8),
            "force": torch.zeros(force_frames, 2),
            "force_valid": torch.ones(force_frames, 2, dtype=torch.bool),
            "force_is_pad": torch.zeros(force_frames, 2, dtype=torch.bool),
            "force_observed_frames": self.force_observed_frames,
            "image_is_pad": torch.zeros(9, dtype=torch.bool),
            "sequence_id": "sequence",
            "current_frame": 0,
            "split_group": "group",
        }


def _cfg(stage):
    return OmegaConf.create({
        "seed": 42,
        "eval_num_inference_steps": 2,
        "model": {"init_mode": "scratch"},
        "training": {"stage": stage, "finetune_video": False},
        "evaluation": {
            "offline_loss_samples": 1,
            "offline_generation_samples": 1,
        },
    })


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
    metrics = runner.evaluate()
    assert model.action_calls == 1
    assert model.force_calls == 0
    assert metrics["generation_action_mse_normalized"] == 0.0


def test_force_warmup_reports_force_without_action_metric(tmp_path):
    model = _FakeModel(force_enabled=True)
    runner = Wan22Trainer.evaluation_only(
        model, _FakeDataset(tmp_path / "normalization.npz"), cfg=_cfg("force_warmup"),
        accelerator=_FakeAccelerator(), output_dir=str(tmp_path),
    )
    metrics = runner.evaluate()
    assert model.force_calls == 1
    assert model.loss_lambda_video == 0.0
    assert model.loss_lambda_action == 0.0
    assert model.loss_lambda_force == 1.0
    assert metrics["val_loss"] == 1.0
    assert metrics["generation_force_mse_normalized"] == 0.0
    assert "generation_action_mse_normalized" not in metrics


def test_force_joint_h0_and_h16_use_action_force_path(tmp_path):
    for history, expected_calls in ((0, 3), (16, 4)):
        model = _FakeModel(force_enabled=True)
        runner = Wan22Trainer.evaluation_only(
            model, _FakeDataset(tmp_path / "normalization.npz", history=history), cfg=_cfg("joint"),
            accelerator=_FakeAccelerator(), output_dir=str(tmp_path / f"h{history}"),
        )
        metrics = runner.evaluate()
        assert model.loss_lambda_video == 1.0
        assert model.loss_lambda_action == 1.0
        assert model.action_calls == 0
        assert model.force_calls == expected_calls
        assert metrics["generation_action_mse_normalized"] == 0.0
        assert metrics["generation_force_mse_normalized"] == 0.0
