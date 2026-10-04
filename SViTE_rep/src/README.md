# SViTE+-Tiny with hard distillation

The project now includes a DeiT-Tiny backbone, learned patch-token selection,
dynamic unstructured weight sparsity, and a complete training/evaluation runner.
See [TRAINING.md](TRAINING.md) for the architecture walkthrough, configuration
choices, ImageNet setup, and remote GPU commands. No ImageNet training has been run.

## What each component does

- `teacher.py`: loads **RegNetY-16GF**, specifically timm's
  `regnety_160.deit_in1k` checkpoint trained by the DeiT authors. Parameters are
  frozen; evaluation mode also prevents BatchNorm statistics from changing.
- `distillation.py`: computes `0.5 * CE(student, true_label) +
  0.5 * CE(student, teacher_argmax)`. The teacher receives the same augmented
  batch as the student, with gradients disabled. A tuple of two logit tensors
  directs the two losses to separate heads; a single tensor implements the
  shared-output formulation in equation (3). A distillation token itself is
  an architectural choice, not something the loss creates.
- `data.py`: reads ImageNet-1k and applies timm random resized crop, horizontal
  flip, RandAugment (`rand-m9-mstd0.5-inc1`), ImageNet normalization, and random
  erasing (probability 0.25, pixel mode). Validation uses a deterministic resize
  and center crop. Default input size is 224 × 224.

The loss accepts raw logits of shape `[batch_size, 1000]` and ground-truth
integer labels of shape `[batch_size]`, or already smoothed Mixup/CutMix targets
of shape `[batch_size, 1000]`. Soft targets are not smoothed twice.
Do not apply softmax before the loss.
For example, if the true label is cat and the teacher predicts dog on an
augmented image, half the objective supervises cat and half supervises dog.
Only the student learns from these losses. The teacher is unnecessary at inference.

Ground-truth label smoothing is 0.1; teacher labels remain unsmoothed, matching
the released hard-distillation code. PyTorch/timm smoothing spreads epsilon
over *all* classes: the target class gets `1 - epsilon + epsilon / K`.
The paper's prose describes spreading epsilon over the other `K - 1` classes;
we follow the released code's convention here.

The `color_jitter=0.3` argument follows the DeiT configuration; timm disables
separate color jitter when this RandAugment policy is active. Ordinary student
dropout is zero. Stochastic depth is separately configured at 0.1.

The training runner includes Mixup/CutMix, repeated augmentation, AdamW, warmup,
cosine learning-rate decay, and EMA. Modern timm is used, so this is not a
bit-for-bit recreation of the original software environment.

## Environment and eventual use

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

Verified locally with Python 3.13 and the versions in `requirements.txt`:
20 tests pass, covering the loss and teacher behavior, patch selection, backbone
dimensions, sparse connectivity, gradient accumulation, EMA masks, and inputs.
The installed timm registry points
to the original `regnety_160-a5fe301d.pth` checkpoint. The factory test substitutes
a small model; pretrained weights have not been downloaded or run in these tests.

ImageNet must be provided locally in this layout, using standard synset names:

```text
imagenet/
  train/n01440764/*.JPEG
  train/... (all 1,000 classes)
  val/n01440764/*.JPEG
  val/... (the same classes)
```

The essential training operation (the complete runner also handles sparse updates):

```python
from teacher import build_teacher
from distillation import HardDistillationLoss

teacher = build_teacher(device)  # downloads pretrained weights on first use
criterion = HardDistillationLoss(teacher)

# images are already augmented, normalized, and on the student's device.
# labels are on that device too. optimizer contains only student parameters.
outputs = student(images)
loss = criterion(images, outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

DeiT trains with ImageNet-1k and does not require JFT-300M. Augmentation provides
different views of existing images; it does not supply independent new examples
or guarantee convergence for an arbitrary student architecture.

## Sources

- [DeiT paper](https://arxiv.org/abs/2012.12877)
- [Authors' hard-distillation implementation](https://github.com/facebookresearch/deit/blob/main/losses.py)
- [Authors' training configuration](https://github.com/facebookresearch/deit/blob/main/main.py)
- [Authors' data transforms](https://github.com/facebookresearch/deit/blob/main/datasets.py)
- [timm's DeiT teacher checkpoint](https://huggingface.co/timm/regnety_160.deit_in1k)
