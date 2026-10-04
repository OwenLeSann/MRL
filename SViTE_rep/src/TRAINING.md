# SViTE+-Tiny: architecture and training

This implements **SViTE+**: SViTE's unstructured weight exploration plus its
learned input-token selector. The backbone has the dimensions of DeiT-Tiny.
The extra distillation token/head and RegNetY-16GF supervision preserve your
earlier hard-distillation request. This combination is an adaptation: the
SViTE paper reports its token-selection experiments on DeiT-Small, and those
results do not establish an accuracy target for this distilled Tiny model.

## Follow one image through the model

1. A normalized `3 × 224 × 224` image becomes 196 patches of size `16 × 16`.
2. A dense projection maps each patch to 192 features. Learned positional
   embeddings preserve each patch's original location.
3. An MLP scores the patch tokens. At the example retention rate of 0.9,
   the selector keeps `floor(196 × 0.9) = 176` of them. The class and distillation
   tokens always survive, giving a sequence of 178 tokens.
4. Twelve transformer blocks process the sequence. Each has 3 attention heads
   (64 features per head), pre-LayerNorm, residual connections, and an MLP
   with hidden width 768. Input embedding width remains **192**, not 176.
5. Two dense classification heads produce 1,000 logits each. The class head
   learns from the image labels; the distillation head learns from the frozen
   teacher's argmax. Evaluation averages the two softmax probability vectors,
   as specified in the supplied DeiT excerpt.

During training, the selector adds Gumbel noise and forms a hard top-k mask.
The straight-through expression `hard - soft.detach() + soft` makes the
forward pass discrete while providing a softmax gradient to the scorer.
Gathering the selected tokens actually reduces the sequence length. Evaluation
uses deterministic top-k scores without noise. The scorer is `192 → 96 → 1`
with GELU: its hidden width is our explicit choice because the paper does not
specify it. Temperature declines linearly from 10 to 0.1, following the released
training engine. Both protected prefix tokens are an extension for distillation.

## Erdős–Rényi initialization and sparse updates

For a matrix with input and output sizes `n_in` and `n_out`, the relative
connection density is

```text
r = (n_in + n_out) / (n_in * n_out)
p = min(1, epsilon * r)
sparsity = 1 - p
```

This is the SET equation you supplied. The `1 - ...` expression refers to
sparsity (removed connections); `p` is density (retained connections).
This ER epsilon is unrelated to the epsilon used for label smoothing.
There are two supported ways to specify the budget:

- **Target sparsity**, the default: `weight_sparsity=0.5`. Solve for a common
  scaling factor so the ER layer densities sum to that budget, saturating small
  layers at density 1. Then randomly select an exact integer number of connections
  per layer. This conditions the ER allocation on exact counts instead of allowing
  the small count fluctuations of independent Bernoulli draws.
- **Literal SET ε**: `--er-epsilon VALUE`. Independently sample each connection
  with probability `p` above. This overrides the target-sparsity allocation;
  the realized sparsity is printed. `--er-epsilon 1` uses the unscaled formula
  literally and is much more sparse than the 50% example. No numerical ε was
  supplied in your excerpts, so the default budget remains an explicit example.

Only transformer-block linear weight matrices are sparsified: fused QKV,
attention output projection, and the two MLP matrices. Patch projection,
positional embeddings, special tokens, norms, biases, scorer, and both classifier
heads remain dense. The 50% setting applies to **5,308,416 prunable weights**,
not to every parameter in the model. Fused QKV is treated as one matrix, as in
the implementation's representation.

At each scheduled update:

1. Collect gradients including the derivatives at inactive connections, on the
   current minibatch. With gradient accumulation, use the sum over microbatches.
2. Choose growth candidates only among connections inactive in the old topology.
3. Mask inactive gradients before AdamW updates the weights.
4. Remove the smallest-magnitude active weights and activate the same number of
   candidates with the largest gradient magnitudes. New weights start at zero;
   reset their AdamW moments too. The number of active mask entries stays fixed.

The fraction replaced is `0.5 * (1 + cos(pi * step / end_step)) / 2`.
Each layer's replacement count is rounded down and capped by its available
inactive connections. Topology freezes at 80% of training. Ordinary weight
training continues for the final 20%. This follows **SViTE's gradient growth**;
it does not use SET's random regrowth or per-epoch signed pruning.

The masks are stored as model buffers and included in checkpoints. EMA copies
the current masks and zeros newly inactive weights rather than retaining an old
dense topology. The reference implementation here uses dense PyTorch kernels:
it provides the sparse algorithm, but does **not** claim sparse-kernel runtime
or memory savings. Dense gradients are retained for growth only on update steps;
this does not implement the paper's optimized sparse gradient-computation kernels.

## Training configuration

`configs/svite_tiny.json` uses the **SViTE paper's DeiT-Tiny training row**.
Its 600 epochs differ from original DeiT's usual 300-epoch run.

| Setting | Value |
|---|---|
| Training initialization | Random sparse student; pretrained frozen teacher |
| Epochs / effective batch | 600 / 512 |
| AdamW learning rate | `5e-4 × effective_batch_size / 512` |
| AdamW betas / epsilon / weight decay | `(0.9, 0.999)` / `1e-8` / `0.05` |
| Learning rate schedule | 5-epoch warmup from `1e-6`, then cosine; minimum `1e-5` |
| Topology update interval | 20,000 optimizer steps |
| Topology end | 80% of calculated total steps (about 1.2 million on ImageNet) |
| Initial replacement fraction | 0.5, cosine decay to zero |
| Label smoothing | 0.1 on true/Mixup/CutMix targets; no smoothing of teacher argmax |
| Mixup / CutMix alpha | 0.8 / 1.0; probability 1; switch probability 0.5 |
| Image augmentation | timm RandAugment, random erasing, crop/flip; repeated augmentation ×3 |
| Ordinary dropout / stochastic depth | 0 / 0.1 |
| EMA decay | 0.99996 |
| Example weight sparsity / token retention | **0.5 / 0.9**, configurable choices |

Biases, norm parameters, and positional/special-token embeddings receive no
weight decay. Mixup/CutMix provides already smoothed probability targets, so the
loss does not smooth them again. The teacher receives exactly the same mixed,
augmented images. Student selection happens after those full images have been
passed into the model; the teacher still sees the full image.

The runner uses gradient accumulation to preserve effective batch 512:
with one GPU and microbatch 64 it accumulates 8 passes; with two GPUs it
accumulates 4. Sparse schedules count optimizer updates, not microbatches.
Incomplete effective batches at an epoch boundary are dropped. CUDA uses
automatic mixed precision; CPU/MPS use float32. DDP is available for CUDA.

## ImageNet on Kaggle or your school's cluster

Use **ILSVRC2012 / ImageNet-1k**, with 1,000 classes. Access options are the
[official ImageNet download page](https://www.image-net.org/download.php) and
[Kaggle's ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
Obtain access through your own account and stage the data on the remote machine.
The project does not contain images, account credentials, or automatic downloads.

On a cluster, first check whether your school already provides ImageNet on shared
storage. On Kaggle, attach the competition data to your notebook after completing
its access requirements. Inspect the mounted files: paths and archive layout can
vary. Avoid assuming that adding data has already extracted all images.

The basic loader expects:

```text
/remote/imagenet/
  train/n01440764/*.JPEG
  train/... (all 1,000 synset directories)
  val/n01440764/*.JPEG
  val/... (the same synset directories)
```

It checks the canonical ImageNet class ordering against timm's metadata so the
teacher and dataset use the same labels. If Kaggle's validation images are flat,
use the original `LOC_val_solution.csv` instead of moving the images:

```sh
python train.py \
  --data-path /remote/ILSVRC/Data/CLS-LOC \
  --val-labels /remote/LOC_val_solution.csv \
  --device cuda --output-dir /remote/runs/svite-tiny
```

The CSV loader translates its synset strings through the training mapping.
Do not use unlabeled test images as validation. Official train/validation tar
archives need extraction and label preparation before using this directory loader.

For a remote Python environment, install the project's requirements or use a
compatible preinstalled PyTorch/torchvision pair with `timm==1.0.30`.
The pinned versions were tested locally on Python 3.13, macOS CPU; choose the
PyTorch CUDA installation appropriate to your remote machine. GPU execution
has not been verified on this Mac. Teacher weights download on first real run;
pre-cache them while the remote environment has internet access:

```sh
python -c 'from teacher import build_teacher; build_teacher()'
```

## Commands

From this directory, first exercise the complete loop with a synthetic dataset
and small synthetic teacher (no data or weight download):

```sh
.venv/bin/python train.py --smoke-test --device cpu --output-dir runs/smoke
```

Smoke mode keeps all 12 Tiny blocks but uses 32-pixel images, two epochs,
effective batch 4, and frequent sparse updates. Its accuracy has no scientific
meaning, and its checkpoints cannot resume an ImageNet experiment.

Single GPU, complete ImageNet run:

```sh
python train.py --data-path /remote/imagenet --device cuda --output-dir runs/svite-tiny
```

Two GPUs (one node, including an allocated cluster node):

```sh
torchrun --standalone --nproc-per-node=2 train.py \
  --data-path /remote/imagenet --device cuda --output-dir runs/svite-tiny
```

For a first real-data check, add `--max-epochs-this-run 1`. This runs one epoch
while retaining the full 600-epoch schedule. A smaller `--batch-size 16` preserves
effective batch 512 through additional accumulation.

Resume at the next epoch, preserving optimizer, scaler, topology, EMA, schedule
position, and random states:

```sh
python train.py --data-path /remote/imagenet --device cuda \
  --resume runs/svite-tiny/last.pt --output-dir runs/svite-tiny
```

Keep the training GPU count and dataset length fixed when resuming. Use the same
torchrun command for a distributed resume. Checkpoints are saved at **epoch
boundaries**; interruption during an epoch restarts that epoch. On hosted notebooks,
persist the output directory between sessions. Full ImageNet training is not
started by installing the project or running its unit tests.

Evaluate a checkpoint; the teacher is not loaded:

```sh
python train.py --data-path /remote/imagenet --device cuda \
  --resume runs/svite-tiny/best.pt --eval
```

Outputs include `config.json`, `metrics.jsonl`, `last.pt`, and `best.pt`.
Metrics report loss, top-1, and top-5 for both the current student and EMA.
`best.pt` is the epoch with the highest top-1 across either version and contains
both versions; evaluation prints both. Validation sharding does not duplicate
examples across GPUs. Accuracy is a fraction of the actual examples evaluated.

## Code map and sources

| File | Purpose |
|---|---|
| `model.py` | Backbone, token selection, probability fusion |
| `sparsity.py` | ER allocation, masks, prune/regrow lifecycle |
| `training.py` | AdamW, schedule, accumulation, EMA, evaluation |
| `train.py` | Configuration, data/device setup, checkpoints, CLI |
| `distillation.py` | Equal-weight class/teacher loss |
| `data.py` | Augmentation and ImageNet layouts |

The attachment supplies SViTE §3.1. Further details were checked against the
[SViTE paper, §3.2 and Table 1](https://arxiv.org/pdf/2106.04533),
[released SViTE code](https://github.com/VITA-Group/SViTE/tree/7dc89fd8fa5f86e797620f00cce6f11e2b73765d/SViTE),
and [DeiT](https://github.com/facebookresearch/deit).
The released `vision_gumbel.py` at that revision truncates tokens in its forward
path; our learned selection follows the paper's straight-through algorithm
instead of treating that truncation as the full method.

Tests cover selection/scorer gradients, a full-resolution forward pass, dense
backbone equivalence, ER budgeting, inactive-gradient regrowth, accumulated
gradients, mask counts, growth initialization, topology freezing, EMA masks,
Mixup smoothing, and label mapping. Local synthetic runs additionally exercise
the complete optimizer/checkpoint/evaluation path. All 20 unit tests passed.
A two-epoch CPU smoke run and an epoch-boundary resumed run produced exactly
equal model, EMA, optimizer, scaler, and global-step states; checkpoint-only
evaluation also passed. ImageNet convergence, CUDA
mixed precision, multi-GPU execution, and paper accuracy remain unverified.
