"""Offline OpenTouch action/force evaluation and paired group bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.evaluation.force_ablation import summarize_bootstrap_outputs
from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision, _resolve_train_device
from fastwam.trainer import Wan22Trainer
from fastwam.utils.config_resolvers import register_default_resolvers


register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    checkpoint = cfg.get("evaluation", {}).get("checkpoint")
    if not checkpoint:
        raise ValueError("evaluation.checkpoint must point to a weight bundle or model.pt")
    precision = _normalize_mixed_precision(cfg.mixed_precision)
    accelerator = Accelerator(mixed_precision=precision)
    device = _resolve_train_device()
    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(precision),
        device=device,
    )
    dataset = instantiate(cfg.data.val)
    if not bool(getattr(dataset, "is_opentouch_force_dataset", False)):
        raise TypeError("scripts/evaluate_opentouch.py requires an OpenTouchForceDataset")
    model.load_checkpoint(str(checkpoint), strict_contract=True)
    checkpoint_hashes = getattr(model, "loaded_contract_hashes", None)
    if not checkpoint_hashes:
        raise ValueError(
            "OpenTouch offline evaluation requires a checkpoint bound to data-contract artifacts"
        )
    for name, current_hash in dataset.contract_hashes.items():
        checkpoint_hash = checkpoint_hashes.get(name)
        if checkpoint_hash != current_hash:
            raise ValueError(
                f"OpenTouch checkpoint/cache contract mismatch for {name}: "
                f"{checkpoint_hash} != {current_hash}"
            )
    model.eval()

    configured_output = cfg.evaluation.get("output_dir")
    output_dir = Path(str(configured_output or (Path(str(cfg.output_dir)) / "offline_evaluation")))
    runner = Wan22Trainer.evaluation_only(
        model,
        dataset,
        cfg=cfg,
        accelerator=accelerator,
        output_dir=str(output_dir),
    )
    metrics = runner.evaluate()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary = summarize_bootstrap_outputs(
            output_dir / "eval",
            samples=int(cfg.evaluation.get("bootstrap_samples", 10000)),
            seed=int(cfg.seed),
            activity_top_fraction=float(
                cfg.evaluation.get("print_prediction_activity_top_fraction", 0.5)
            ),
        )
        (output_dir / "bootstrap_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps({"metrics": metrics, "bootstrap": summary}, indent=2))


if __name__ == "__main__":
    main()
