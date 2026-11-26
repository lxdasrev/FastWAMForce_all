import logging
import json
import hashlib
import inspect
import os
import re
import shutil
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim
from .evaluation.force_ablation import future_force_to_action
from .evaluation.opentouch_metrics import add_action_metrics, add_force_metrics, add_pair

logger = get_logger(__name__)


class Wan22Trainer:
    @classmethod
    def evaluation_only(cls, model, val_dataset, *, cfg: DictConfig, accelerator: Accelerator, output_dir: str):
        """Build a lightweight evaluator without optimizer or training dataloaders."""
        instance = cls.__new__(cls)
        instance.model = model
        instance.val_dataset = val_dataset
        instance.cfg = cfg
        instance.accelerator = accelerator
        instance.output_dir = str(output_dir)
        instance.eval_dir = os.path.join(instance.output_dir, "eval")
        ensure_dir(instance.output_dir)
        ensure_dir(instance.eval_dir)
        instance.seed = int(cfg.seed)
        instance.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        training_cfg = cfg.get("training", {})
        evaluation_cfg = cfg.get("evaluation", {})
        instance.training_stage = str(training_cfg.get("stage", "joint")).strip().lower()
        instance.finetune_video = bool(training_cfg.get("finetune_video", True))
        instance.joint_eval_every = 1
        instance.tactile_ablation_every = 1
        configured_loss = int(evaluation_cfg.get("offline_loss_samples", 0))
        instance.eval_loss_samples = configured_loss or len(val_dataset)
        configured_generation = int(evaluation_cfg.get("offline_generation_samples", 0))
        instance.eval_generation_samples = configured_generation or len(val_dataset)
        instance.global_step = 1
        # Match the train-time stage semantics before running loss validation.
        # In particular, force_warmup must not report video/action losses as
        # part of its total merely because this lightweight runner skips the
        # regular trainer constructor.
        instance._configure_stage_loss_lambdas(model)
        model.eval()
        return instance

    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        training_cfg = cfg.get("training", {})
        self.training_stage = str(training_cfg.get("stage", "joint")).strip().lower()
        self.finetune_video = bool(training_cfg.get("finetune_video", True))
        self.joint_eval_every = int(training_cfg.get("joint_eval_every", 2000))
        self.tactile_ablation_every = int(training_cfg.get("tactile_ablation_every", 2000))
        evaluation_cfg = cfg.get("evaluation", {})
        self.eval_loss_samples = int(evaluation_cfg.get("loss_samples", 64))
        self.eval_generation_samples = int(evaluation_cfg.get("generation_samples", 16))

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        # Keep a stable optimizer parameter-group layout across force_warmup ->
        # joint resumes. Frozen parameters receive no gradients/state during
        # warmup, but must already be present so Accelerate can restore exactly.
        trainable_params = list(self.model.dit.parameters())
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(proprio_encoder.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        if bool(getattr(self.model, "compile_training_denoise", False)):
            logger.info("Compiling training denoise forward/backward before DeepSpeed initialization.")
            compile_start = time.perf_counter()
            warmup_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=self.train_sampler,
                num_workers=0,
            )
            warmup_sample = next(iter(warmup_loader))
            with self.accelerator.autocast():
                warmup_loss, _ = self.model.training_loss(warmup_sample)
            warmup_loss.backward()
            self.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.accelerator.device)
            logger.info(
                "Finished training denoise compile warmup in %.2f seconds.",
                time.perf_counter() - compile_start,
            )
            set_global_seed(self.seed)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            if (resume_path / "trainer_state.json").exists():
                logger.info("Resuming full training state from directory: %s", resume)
                self.load_training_state(str(resume_path))
            elif (resume_path / "model.pt").exists():
                logger.info("Loading strict weight bundle from directory: %s", resume)
                self._validate_resume_contract(resume_path)
                self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
            else:
                raise FileNotFoundError(f"resume directory has neither trainer_state.json nor model.pt: {resume}")
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    def _apply_dit_only_train_mode(self, model):
        model.eval()
        model.requires_grad_(False)
        if not getattr(model, "force_enabled", False):
            model.dit.train()
            model.dit.requires_grad_(True)
        else:
            if self.training_stage not in {"force_warmup", "joint"}:
                raise ValueError("training.stage must be force_warmup or joint for a force model")
            self._configure_stage_loss_lambdas(model)
            force_expert = model.dit.force_expert
            force_expert.train()
            force_expert.requires_grad_(True)
            if self.training_stage != "force_warmup":
                model.action_expert.train()
                model.action_expert.requires_grad_(True)
                model.dit.bridge_gates.requires_grad_(True)
                if self.finetune_video:
                    model.video_expert.train()
                    model.video_expert.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None and (not getattr(model, "force_enabled", False) or self.training_stage == "joint"):
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        logger.info(
            "Configured training stage=%s finetune_video=%s trainable_params=%d",
            self.training_stage,
            self.finetune_video,
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        )

    def _configure_stage_loss_lambdas(self, model) -> None:
        """Apply warmup/joint loss semantics without changing trainable state."""
        if not getattr(model, "force_enabled", False):
            return
        if self.training_stage not in {"force_warmup", "joint"}:
            raise ValueError("training.stage must be force_warmup or joint for a force model")
        if not hasattr(model, "_configured_loss_lambdas"):
            model._configured_loss_lambdas = (
                model.loss_lambda_video, model.loss_lambda_action, model.loss_lambda_force
            )
        configured_video, configured_action, configured_force = model._configured_loss_lambdas
        if self.training_stage == "force_warmup":
            model.loss_lambda_video = 0.0
            model.loss_lambda_action = 0.0
        else:
            model.loss_lambda_video = configured_video
            model.loss_lambda_action = configured_action
        model.loss_lambda_force = configured_force

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        action_is_pad = sample.get("action_is_pad", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        force = sample.get("force", None)
        force_valid = sample.get("force_valid", None)
        force_is_pad = sample.get("force_is_pad", None)
        force_observed_frames = sample.get("force_observed_frames", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])
            if action_is_pad is not None and action_is_pad.ndim == 1:
                action_is_pad = action_is_pad.unsqueeze(0)

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim not in (2, 3):
                raise ValueError(f"sample proprio must batch to [B,D] or [B,T,D], got {tuple(proprio.shape)}")

        if force is not None:
            if force_valid is None or force_is_pad is None:
                raise ValueError("force_valid and force_is_pad are required whenever force is present")
            if force.ndim == 2:
                force = force.unsqueeze(0)
            if force_valid.ndim == 2:
                force_valid = force_valid.unsqueeze(0)
            if force_is_pad.ndim == 2:
                force_is_pad = force_is_pad.unsqueeze(0)
            if force.ndim != 3 or force_valid.shape != force.shape or force_is_pad.shape != force.shape:
                raise ValueError("force/force_valid/force_is_pad must batch to identical [B,T,2] shapes")
            if isinstance(force_observed_frames, torch.Tensor):
                force_observed_frames = int(force_observed_frames.item())

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "action_is_pad": action_is_pad,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
            "force": force,
            "force_valid": force_valid,
            "force_is_pad": force_is_pad,
            "force_observed_frames": force_observed_frames,
            "sequence_id": sample.get("sequence_id"),
            "current_frame": sample.get("current_frame"),
            "split_group": sample.get("split_group", sample.get("sequence_id")),
        }

    def _distributed_finalize_pairs(self, pairs: dict[str, tuple[float, float]]) -> dict[str, float]:
        names = sorted(pairs)
        if not names:
            return {}
        local = torch.tensor(
            [[pairs[name][0], pairs[name][1]] for name in names],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        gathered = self.accelerator.gather(local).reshape(-1, len(names), 2).sum(dim=0).cpu()
        result = {}
        for index, name in enumerate(names):
            numerator, denominator = gathered[index].tolist()
            result[name] = float(numerator / denominator) if denominator > 0 else 0.0
            result[f"{name}_count"] = float(denominator)
        return result

    def _rank_eval_indices(self, count: int, *, contact_only: bool) -> tuple[list[int], list[int]]:
        indices = self.val_dataset.fixed_eval_indices(
            count,
            seed=self.seed,
            contact_only=contact_only,
            unique_groups=contact_only,
        )
        if contact_only and not indices:
            logger.warning("No contact validation windows were found; falling back to all validation windows.")
            indices = self.val_dataset.fixed_eval_indices(
                count, seed=self.seed, contact_only=False, unique_groups=contact_only
            )
        rank = int(self.accelerator.process_index)
        world = int(self.accelerator.num_processes)
        return indices, indices[rank::world]

    @staticmethod
    def _sample_context_kwargs(sample: dict) -> dict:
        if sample["context"] is not None:
            return {
                "prompt": None,
                "context": sample["context"][0],
                "context_mask": sample["context_mask"][0],
            }
        return {"prompt": sample["prompt"][0], "context": None, "context_mask": None}

    def _evaluate_opentouch_losses(self, model) -> dict[str, float]:
        _, local_indices = self._rank_eval_indices(self.eval_loss_samples, contact_only=False)
        pairs: dict[str, tuple[float, float]] = {}
        for index in local_indices:
            sample = self._to_batched_eval_sample(self.val_dataset[index])
            cuda_devices = []
            if torch.cuda.is_available():
                device = torch.device(model.device)
                cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=cuda_devices):
                sample_seed = self.seed * 1_000_003 + int(index)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                with self.accelerator.autocast():
                    loss, loss_dict = model.training_loss(sample)
            add_pair(pairs, "val_loss", float(loss.float().item()), 1.0)
            for name, value in loss_dict.items():
                add_pair(pairs, f"val_{name}", float(value), 1.0)

            if sample.get("action_is_pad") is not None:
                valid = ~sample["action_is_pad"].bool()
                add_pair(pairs, "action_valid_ratio", valid.sum().item(), valid.numel())
            if sample.get("force") is not None:
                effective = sample["force_valid"].bool() & ~sample["force_is_pad"].bool()
                add_pair(pairs, "force_valid_ratio", effective.sum().item(), effective.numel())
        add_pair(
            pairs,
            "loss_eval_samples",
            len(local_indices),
            1.0 / max(int(self.accelerator.num_processes), 1),
        )
        return self._distributed_finalize_pairs(pairs)

    def _run_force_action_inference(self, model, sample: dict, *, observed_force=None, observed_valid=None,
                                    reads_future: bool = True) -> dict:
        observed = int(sample["force_observed_frames"])
        if observed not in (1, 17):
            raise ValueError(f"OpenTouch observed force length must be 1 or 17, got {observed}")
        if observed != int(self.val_dataset.force_observed_frames):
            raise ValueError(
                f"force history contract mismatch: sample={observed}, dataset={self.val_dataset.force_observed_frames}"
            )
        if observed_force is None:
            observed_force = sample["force"][0, :observed]
        if observed_valid is None:
            observed_valid = (
                sample["force_valid"][0, :observed].bool()
                & ~sample["force_is_pad"][0, :observed].bool()
            )
        video0 = sample["video"][0]
        proprio_batch = sample.get("proprio")
        proprio = None if proprio_batch is None else (
            proprio_batch[0] if proprio_batch.ndim == 2 else proprio_batch[0, 0]
        )
        kwargs = {
            **self._sample_context_kwargs(sample),
            "input_image": video0[:, 0].unsqueeze(0),
            "observed_force": observed_force,
            "observed_force_valid": observed_valid,
            "proprio": proprio,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "normalization": str(self.val_dataset.normalization_path),
        }
        with future_force_to_action(model, reads_future):
            return model.infer_action_force(**kwargs)

    def _run_action_only_inference(self, model, sample: dict) -> torch.Tensor:
        video0 = sample["video"][0]
        proprio_batch = sample.get("proprio")
        proprio = None if proprio_batch is None else (
            proprio_batch[0] if proprio_batch.ndim == 2 else proprio_batch[0, 0]
        )
        output = model.infer_action(
            **self._sample_context_kwargs(sample),
            input_image=video0[:, 0].unsqueeze(0),
            action_horizon=int(sample["action_horizon"]),
            proprio=proprio,
            num_inference_steps=self.eval_num_inference_steps,
            seed=42,
        )
        return output["action"]

    def _evaluate_opentouch_generation(
        self,
        model,
        *,
        run_ablation: bool,
        report_action: bool = True,
    ) -> dict[str, float]:
        all_indices, local_indices = self._rank_eval_indices(
            self.eval_generation_samples, contact_only=True
        )
        pairs: dict[str, tuple[float, float]] = {}
        saved: dict[str, list] = {
            "sequence_id": [], "current_frame": [], "split_group": [],
            "action_pred_normalized": [], "action_gt_normalized": [], "action_valid": [],
            "action_pred_canonical": [], "action_gt_canonical": [],
        }
        sample_mse: dict[str, list[float]] = {}
        force_enabled = bool(getattr(model, "force_enabled", False))
        metric_prefixes = ["generation_"] if report_action else []
        if run_ablation and force_enabled:
            metric_prefixes.extend(
                [
                    "ablation_zero_",
                    "ablation_disable_future_force_to_action_",
                    "ablation_batch_shuffle_",
                ]
            )
            if int(getattr(self.val_dataset, "force_observed_frames", 1)) == 17:
                metric_prefixes.append("ablation_time_shuffle_")
        for prefix in metric_prefixes:
            for name in (
                "action_mse_normalized",
                "translation_mse_canonical",
                "rotation_mse_canonical",
                "gripper_mae_canonical",
            ):
                pairs[f"{prefix}{name}"] = (0.0, 0.0)
        if force_enabled:
            pairs["generation_force_mse_normalized"] = (0.0, 0.0)
            pairs["generation_force_mae_relative_pressure"] = (0.0, 0.0)
            pairs["generation_force_valid_ratio"] = (0.0, 0.0)

        for local_position, index in enumerate(local_indices):
            sample = self._to_batched_eval_sample(self.val_dataset[index])
            action_gt = sample["action"][0].float().cpu()
            action_valid = (
                ~sample["action_is_pad"][0].bool().cpu()
                if sample.get("action_is_pad") is not None
                else torch.ones(action_gt.shape[0], dtype=torch.bool)
            )
            if force_enabled:
                generated = self._run_force_action_inference(model, sample)
                action_pred = generated["action_normalized"]
            else:
                generated = None
                action_pred = self._run_action_only_inference(model, sample)

            if report_action:
                correct_mse = add_action_metrics(
                    pairs,
                    prefix="generation_",
                    pred_normalized=action_pred,
                    target_normalized=action_gt,
                    valid_steps=action_valid,
                    denormalize=self.val_dataset.denormalize_action,
                )
                sample_mse.setdefault("correct_action_mse", []).append(
                    np.nan if correct_mse is None else correct_mse
                )

            saved["sequence_id"].append(str(sample["sequence_id"]))
            saved["current_frame"].append(int(sample["current_frame"]))
            saved["split_group"].append(str(sample["split_group"]))
            saved["action_pred_normalized"].append(action_pred.numpy())
            saved["action_gt_normalized"].append(action_gt.numpy())
            saved["action_valid"].append(action_valid.numpy())
            saved["action_pred_canonical"].append(
                self.val_dataset.denormalize_action(action_pred).numpy()
            )
            saved["action_gt_canonical"].append(
                self.val_dataset.denormalize_action(action_gt).numpy()
            )

            if not force_enabled:
                continue
            observed = int(sample["force_observed_frames"])
            future_gt = sample["force"][0, observed:].float().cpu()
            effective_future = (
                sample["force_valid"][0, observed:].bool().cpu()
                & ~sample["force_is_pad"][0, observed:].bool().cpu()
            )
            future_pred = generated["future_force_normalized"]
            add_force_metrics(
                pairs,
                prefix="generation_",
                pred_normalized=future_pred,
                target_normalized=future_gt,
                effective_valid=effective_future,
                denormalize=self.val_dataset.denormalize_force,
            )
            add_pair(pairs, "generation_force_valid_ratio", effective_future.sum().item(), effective_future.numel())
            for key in (
                "force_pred_normalized", "force_gt_normalized", "effective_force_valid",
                "force_pred_relative_pressure", "force_gt_relative_pressure",
                "observed_force_normalized", "observed_force_valid",
            ):
                saved.setdefault(key, [])
            saved["force_pred_normalized"].append(future_pred.numpy())
            saved["force_gt_normalized"].append(future_gt.numpy())
            saved["effective_force_valid"].append(effective_future.numpy())
            saved["force_pred_relative_pressure"].append(
                self.val_dataset.denormalize_force(future_pred).numpy()
            )
            saved["force_gt_relative_pressure"].append(
                self.val_dataset.denormalize_force(future_gt).numpy()
            )
            saved["observed_force_normalized"].append(sample["force"][0, :observed].numpy())
            saved["observed_force_valid"].append(
                (
                    sample["force_valid"][0, :observed].bool()
                    & ~sample["force_is_pad"][0, :observed].bool()
                ).numpy()
            )

            if not run_ablation:
                continue
            observed_force = sample["force"][0, :observed]
            observed_valid = (
                sample["force_valid"][0, :observed].bool()
                & ~sample["force_is_pad"][0, :observed].bool()
            )
            variants = {
                "zero": (torch.zeros_like(observed_force), observed_valid, True),
                "disable_future_force_to_action": (observed_force, observed_valid, False),
            }
            if len(all_indices) > 1:
                global_position = all_indices.index(index)
                shuffled_index = all_indices[(global_position + 1) % len(all_indices)]
                shuffled = self._to_batched_eval_sample(self.val_dataset[shuffled_index])
                variants["batch_shuffle"] = (
                    shuffled["force"][0, :observed],
                    shuffled["force_valid"][0, :observed].bool()
                    & ~shuffled["force_is_pad"][0, :observed].bool(),
                    True,
                )
            if observed == 17:
                order = torch.randperm(16, generator=torch.Generator().manual_seed(45))
                order = torch.cat((order, torch.tensor([16])))
                variants["time_shuffle"] = (observed_force[order], observed_valid[order], True)

            for name, (variant_force, variant_valid, reads_future) in variants.items():
                output = self._run_force_action_inference(
                    model,
                    sample,
                    observed_force=variant_force,
                    observed_valid=variant_valid,
                    reads_future=reads_future,
                )
                variant_action = output["action_normalized"]
                mse = add_action_metrics(
                    pairs,
                    prefix=f"ablation_{name}_",
                    pred_normalized=variant_action,
                    target_normalized=action_gt,
                    valid_steps=action_valid,
                    denormalize=self.val_dataset.denormalize_action,
                )
                sample_mse.setdefault(f"{name}_action_mse", []).append(
                    np.nan if mse is None else mse
                )
                saved.setdefault(f"action_pred_{name}_normalized", []).append(variant_action.numpy())
                saved.setdefault(f"action_pred_{name}_canonical", []).append(
                    self.val_dataset.denormalize_action(variant_action).numpy()
                )

        add_pair(
            pairs,
            "generation_samples",
            len(local_indices),
            1.0 / max(int(self.accelerator.num_processes), 1),
        )
        if saved["sequence_id"]:
            output_path = Path(self.eval_dir) / (
                f"opentouch_generation_step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.npz"
            )
            payload = {}
            for name, values in saved.items():
                if name in {"sequence_id", "split_group"}:
                    payload[name] = np.asarray(values, dtype=str)
                else:
                    payload[name] = np.asarray(values)
            payload.update({name: np.asarray(values, dtype=np.float64) for name, values in sample_mse.items()})
            np.savez_compressed(output_path, **payload)
            metadata = {
                "step": int(self.global_step),
                "split": str(getattr(self.val_dataset, "split", "val")),
                "training_stage": self.training_stage,
                "finetune_video": self.finetune_video,
                "force_enabled": force_enabled,
                "force_history_frames": int(getattr(self.val_dataset, "force_history_frames", 0)),
                "init_mode": str(self.cfg.get("model", {}).get("init_mode", "unknown")),
                "model_id": str(self.cfg.get("model", {}).get("model_id", "unknown")),
                "action_dit_pretrained_path": self.cfg.get("model", {}).get(
                    "action_dit_pretrained_path"
                ),
                "contract_hashes": getattr(self.val_dataset, "contract_hashes", {}),
            }
            output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return self._distributed_finalize_pairs(pairs)

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None
        if not bool(getattr(self.val_dataset, "is_opentouch_force_dataset", False)):
            return self._evaluate_lerobot()

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()
        try:
            metrics = self._evaluate_opentouch_losses(model)
            joint_due = self.joint_eval_every > 0 and self.global_step % self.joint_eval_every == 0
            ablation_due = (
                self.tactile_ablation_every > 0
                and self.global_step % self.tactile_ablation_every == 0
            )
            force_enabled = bool(getattr(model, "force_enabled", False))
            warmup_force_generation = self.training_stage == "force_warmup" and force_enabled and joint_due
            regular_generation = self.training_stage != "force_warmup" and (joint_due or ablation_due)
            if warmup_force_generation or regular_generation:
                metrics.update(
                    self._evaluate_opentouch_generation(
                        model,
                        run_ablation=bool(
                            self.training_stage != "force_warmup" and ablation_due and force_enabled
                        ),
                        report_action=self.training_stage != "force_warmup",
                    )
                )
            metrics["video_expert_trainable"] = float(self.finetune_video)
            metrics["force_history_frames"] = float(self.val_dataset.force_history_frames)
            if self.accelerator.is_main_process:
                metadata = {
                    "step": int(self.global_step),
                    "training_stage": self.training_stage,
                    "finetune_video": self.finetune_video,
                    "force_enabled": bool(getattr(model, "force_enabled", False)),
                    "force_history_frames": int(self.val_dataset.force_history_frames),
                    "init_mode": str(self.cfg.get("model", {}).get("init_mode", "unknown")),
                    "model_id": str(self.cfg.get("model", {}).get("model_id", "unknown")),
                    "action_dit_pretrained_path": self.cfg.get("model", {}).get(
                        "action_dit_pretrained_path"
                    ),
                    "resume": None if self.cfg.get("resume") is None else str(self.cfg.get("resume")),
                    "use_text_embed_cache": bool(getattr(self.val_dataset, "use_text_embed_cache", False)),
                    "contract_hashes": getattr(self.val_dataset, "contract_hashes", {}),
                    "metrics": metrics,
                }
                metadata_path = Path(self.eval_dir) / f"opentouch_eval_step_{self.global_step:06d}.json"
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return metrics
        finally:
            if was_dit_training:
                self._set_dit_only_train_mode()

    @torch.no_grad()
    def _evaluate_lerobot(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # Keep a rank-specific but step-invariant validation item so tactile
        # ablations remain paired across checkpoints.
        eval_index = (self.seed + self.accelerator.process_index) % len(self.val_dataset)
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = None
        if sample.get("proprio") is not None:
            proprio_batch = sample["proprio"]
            proprio = proprio_batch[0] if proprio_batch.ndim == 2 else proprio_batch[0, 0]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        force_action_mse = force_future_mse = None
        force_ablation_metrics = None
        joint_eval_every = int(self.cfg.get("training", {}).get("joint_eval_every", 2000))
        if (
            getattr(model, "force_enabled", False)
            and sample.get("force") is not None
            and joint_eval_every > 0
            and self.global_step % joint_eval_every == 0
        ):
            observed = int(sample["force_observed_frames"])
            force_kwargs = {
                "prompt": None if sample["context"] is not None else prompt,
                "context": sample["context"][0] if sample["context"] is not None else None,
                "context_mask": sample["context_mask"][0] if sample["context_mask"] is not None else None,
                "input_image": input_image,
                "observed_force": sample["force"][0, :observed],
                "observed_force_valid": sample["force_valid"][0, :observed],
                "proprio": proprio,
                "num_inference_steps": self.eval_num_inference_steps,
                "seed": 42,
            }
            generated = model.infer_action_force(**force_kwargs)
            pred_action_force = generated["action_normalized"]
            pred_future_force = generated["future_force_normalized"]
            gt_action_force = sample["action"][0].float().cpu()
            gt_future_force = sample["force"][0, observed:].float().cpu()
            action_valid = (
                ~sample["action_is_pad"][0].bool().cpu()
                if sample.get("action_is_pad") is not None else torch.ones(32, dtype=torch.bool)
            )
            force_valid = sample["force_valid"][0, observed:].bool().cpu()
            force_action_mse = (pred_action_force[action_valid] - gt_action_force[action_valid]).pow(2).mean().item()
            force_future_mse = (
                (pred_future_force[force_valid] - gt_future_force[force_valid]).pow(2).mean().item()
                if bool(force_valid.any()) else 0.0
            )
            force_ablation_metrics = {"correct": force_action_mse}

            def run_ablation(observed_force, observed_valid, *, reads_future=True, ablation_seed=42):
                kwargs = dict(force_kwargs)
                kwargs["observed_force"] = observed_force
                kwargs["observed_force_valid"] = observed_valid
                kwargs["seed"] = ablation_seed
                with future_force_to_action(model, reads_future):
                    output = model.infer_action_force(**kwargs)["action_normalized"]
                return (output[action_valid] - gt_action_force[action_valid]).pow(2).mean().item()

            observed_force = sample["force"][0, :observed]
            observed_valid = sample["force_valid"][0, :observed]
            force_ablation_metrics["zero"] = run_ablation(
                torch.zeros_like(observed_force), observed_valid
            )
            shuffled_index = (eval_index + max(1, len(self.val_dataset) // 2)) % len(self.val_dataset)
            shuffled_sample = self.val_dataset[shuffled_index]
            shuffled_force = shuffled_sample["force"][:observed]
            shuffled_valid = shuffled_sample["force_valid"][:observed]
            force_ablation_metrics["batch_shuffle"] = run_ablation(
                shuffled_force, shuffled_valid
            )
            if observed == 17:
                history_order = torch.randperm(16, generator=torch.Generator().manual_seed(45))
                order = torch.cat((history_order, torch.tensor([16])))
                force_ablation_metrics["time_shuffle"] = run_ablation(
                    observed_force[order], observed_valid[order]
                )
            force_ablation_metrics["disable_future_force_to_action"] = run_ablation(
                observed_force, observed_valid, reads_future=False
            )
            np.savez_compressed(
                os.path.join(self.eval_dir, f"joint_force_step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.npz"),
                action_pred=pred_action_force.numpy(), action_gt=gt_action_force.numpy(),
                force_pred=pred_future_force.numpy(), force_gt=gt_future_force.numpy(),
                force_valid=force_valid.numpy(),
            )

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        if force_action_mse is not None:
            force_metrics = torch.tensor(
                [force_action_mse, force_future_mse], device=self.accelerator.device, dtype=torch.float32
            ).unsqueeze(0)
            force_metrics = self.accelerator.gather_for_metrics(force_metrics).mean(dim=0)
            result["joint_force_action_mse"] = float(force_metrics[0].item())
            result["joint_force_future_mse"] = float(force_metrics[1].item())
            for name, value in force_ablation_metrics.items():
                metric = torch.tensor(value, device=self.accelerator.device, dtype=torch.float32).reshape(1)
                result[f"force_ablation_{name}_action_mse"] = float(
                    self.accelerator.gather_for_metrics(metric).mean().item()
                )
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        bundle_dir = Path(self.weights_dir) / step_tag
        bundle_dir.mkdir(parents=True, exist_ok=True)
        contract_hashes = self._copy_data_contract(bundle_dir)
        ckpt_path = bundle_dir / "model.pt"
        model.save_checkpoint(
            str(ckpt_path), optimizer=None, step=self.global_step, contract_hashes=contract_hashes
        )
        return str(bundle_dir)

    def _copy_data_contract(self, destination: Path) -> dict[str, str] | None:
        cache_dir = getattr(self.train_dataset, "cache_dir", None)
        if cache_dir is None:
            return None
        source_dir = Path(cache_dir)
        names = ("normalization.npz", "data_contract.json", "split_manifest.json", "contract_hashes.json")
        hashes = {}
        for name in names:
            source = source_dir / name
            if not source.exists():
                raise FileNotFoundError(f"training dataset contract artifact missing: {source}")
            target = destination / name
            shutil.copy2(source, target)
            hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        return hashes

    def _validate_resume_contract(self, checkpoint_dir: Path) -> None:
        cache_dir = getattr(self.train_dataset, "cache_dir", None)
        if cache_dir is None:
            return
        for name in ("normalization.npz", "data_contract.json", "split_manifest.json"):
            saved = checkpoint_dir / name
            current = Path(cache_dir) / name
            if not saved.exists():
                raise FileNotFoundError(f"strict Force resume requires {saved}")
            saved_hash = hashlib.sha256(saved.read_bytes()).hexdigest()
            current_hash = hashlib.sha256(current.read_bytes()).hexdigest()
            if saved_hash != current_hash:
                raise ValueError(
                    f"strict Force resume contract mismatch for {name}: {saved_hash} != {current_hash}"
                )

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
            self._copy_data_contract(Path(state_path))
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self._validate_resume_contract(Path(state_dir))
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    bridge_diag_snapshot = None
                    diagnostics_fn = getattr(self.accelerator.unwrap_model(self.model).dit, "bridge_diagnostics", None)
                    if diagnostics_fn is not None:
                        bridge_diag_snapshot = {}
                        for bridge, values in diagnostics_fn().items():
                            bridge_diag_snapshot[bridge] = {
                                "raw": float(values["raw"].float().item()),
                                "tanh": float(values["tanh"].float().item()),
                                "grad": 0.0 if values["grad"] is None else float(values["grad"].float().item()),
                                "residual_rms": 0.0 if values["residual_rms"] is None else float(values["residual_rms"].float().item()),
                            }
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        if bridge_diag_snapshot is not None:
                            for bridge, values in bridge_diag_snapshot.items():
                                wandb_payload[f"bridge/{bridge}/raw_gate"] = values["raw"]
                                wandb_payload[f"bridge/{bridge}/tanh_gate"] = values["tanh"]
                                wandb_payload[f"bridge/{bridge}/grad"] = values["grad"]
                                wandb_payload[f"bridge/{bridge}/residual_rms"] = values["residual_rms"]
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            numeric_metrics = {
                                key: float(value)
                                for key, value in metrics.items()
                                if isinstance(value, (int, float, np.integer, np.floating))
                                and not key.endswith("_count")
                            }
                            description = f"[eval] step={self.global_step} " + " ".join(
                                f"{key}={value:.4f}" for key, value in sorted(numeric_metrics.items())
                            )
                            logger.info(description)
                            eval_payload = {
                                f"eval/{key}": float(value)
                                for key, value in metrics.items()
                                if isinstance(value, (int, float, np.integer, np.floating))
                            }
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
