# FastWAMForce OpenTouch pipeline

This implementation is an offline research path. OpenTouch pressure is a
relative ADC signal, not force in newtons, and its human-hand-derived action is
a palm-width-normalized pseudo TCP trajectory. It must not be sent directly to
a robot. Robot extrinsics, IK, collision/workspace constraints, a live 17-frame
buffer, receding-horizon control and hardware safety remain outside this stage.

## Build the cache

The converter keeps JPEG frames in the original HDF5 and writes only compact
canonical trajectories, thumb/index pressure, masks, split manifests and
train-only statistics.

```bash
python scripts/preprocess_opentouch.py \
  --input-dir /home/lxd/code/embodyaicode/opentouch/opentouch-main/data \
  --output-dir ./data/opentouch_force_cache \
  --force-history-frames 0
```

Use `--force-history-frames 16` to build the 49-frame variant. A cache is bound
to one force mode; the loader intentionally rejects a mismatched override.
Source SHA-256 is enabled by default. `--skip-source-hash` is useful only for a
quick local smoke cache and should not be used for reported experiments.

The output contract contains:

- `normalization.npz`: action/proprio mean-scale and force robust statistics;
- `data_contract.json`: coordinate, unit, timing, taxel and dimension semantics;
- `split_manifest.json`: sequence groups and every legal training window;
- `contract_hashes.json`: strict hashes used by resume and inference.

Generate the prompt embeddings after the cache exists:

```bash
OPENTOUCH_FORCE_CACHE=./data/opentouch_force_cache \
python scripts/precompute_text_embeds.py task=opentouch_force_224_1e-4
```

## Training

The default task is a 10,000-step joint pilot with per-GPU micro batch 1 and
gradient accumulation 4. The process count is checked against an explicitly
provided `CUDA_VISIBLE_DEVICES`; the launcher never selects hardware or changes
offload/freeze policy silently.

```bash
CUDA_VISIBLE_DEVICES=2 \
bash scripts/train_zero2.sh 1 \
task=opentouch_force_224_1e-4 max_steps=200
```

```bash
CUDA_VISIBLE_DEVICES=1,3,5,7 \
bash scripts/train_zero2.sh 4 \
task=opentouch_force_224_1e-4 \
batch_size=1 gradient_accumulation_steps=4 max_steps=10000
```

For the two explicit phases, first run
`training.stage=force_warmup max_steps=1000`, then load the warmup weight bundle
with `training.stage=joint max_steps=10000`. A weight bundle intentionally starts
a new joint optimizer/scheduler; a state directory is only for interruption
recovery within the same stage. Warmup trains only the
ForceExpert (including force-side bridge projections) and zeroes video/action
loss weights. Joint training enables ActionExpert, ForceExpert, bridge gates,
and proprio encoder; `training.finetune_video=false` freezes VideoExpert.

`model.init_mode=pretrained` loads Wan2.2 and interpolated ActionDiT while the
force branch starts randomly. `model.init_mode=scratch` randomizes the three
trainable DiT experts but still loads and freezes the pretrained VAE/text
encoder. A legacy FastWAM checkpoint is an initialization source, never a
strict Force resume.

Each saved weight bundle is a directory containing `model.pt` and all four data
contract files. Contract mismatch is an error by default. Training logs include
raw, scheduler-weighted and final force losses, valid ratios/sample counts, and
flow `x0_hat` prediction-versus-truth summaries. Every configured joint eval
writes full normalized action/force predictions and truths as `.npz`.

## Inference

`FastWAM.infer_action_force` accepts one current force frame or 16 history plus
current, caches only the current video's 30-layer K/V, and jointly denoises 32
actions and 32 future force frames. It never generates future video. Supplying
`normalization.npz` returns both normalized and contract-denormalized tensors.
The original `infer_action` remains the no-force baseline.

## Evaluation

OpenTouch validation never calls the legacy future-video rollout. Every
`eval_every` steps it evaluates deterministic flow losses on 64 fixed,
sequence-group-stratified windows. Every `training.joint_eval_every` steps it
jointly generates action/force on 16 fixed contact groups; tactile ablations are
controlled independently by `training.tactile_ablation_every`. Warmup reports
force loss and, when the generation cadence is reached, future-force metrics
without treating its untrained action as an effect metric. A model configured without ForceExpert uses
`infer_action` as the action-only baseline. Both 1-frame and 17-frame observed
force modes are inferred from the cache contract.

Run a strict checkpoint-bound offline evaluation with one non-overlapping
contact window per validation sequence group:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/evaluate_opentouch.py \
  task=opentouch_force_224_1e-4 \
  evaluation.checkpoint=/path/to/checkpoints/weights/step_010000 \
  evaluation.output_dir=/path/to/offline_eval
```

Use `data.val.split=test` for the test split. The command writes normalized and
canonical action predictions, normalized and relative-pressure force
predictions, effective masks, identifiers, aggregate metrics and a 10,000-draw
paired sequence-group bootstrap summary. The primary success rule requires the
correct-force action MSE to improve by at least 2% over both zero and shuffled
force, with a positive 95% confidence-interval lower bound.
