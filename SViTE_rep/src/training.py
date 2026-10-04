"""Training primitives shared by the real runner and the synthetic smoke test."""

import copy
import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn

from model import prediction_probabilities


def make_optimizer(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim <= 1 or name in {"pos_embed", "cls_token", "dist_token"}:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config["learning_rate"] * config["effective_batch_size"] / 512,
        betas=(0.9, 0.999), eps=1e-8,
    )


def learning_rate(epoch, config):
    """Epoch-based warmup followed by cosine decay (DeiT/timm convention)."""
    base = config["learning_rate"] * config["effective_batch_size"] / 512
    warmup = config["warmup_epochs"]
    if epoch < warmup:
        return config["warmup_learning_rate"] + (base - config["warmup_learning_rate"]) * epoch / warmup
    return config["min_learning_rate"] + (base - config["min_learning_rate"]) * (
        1 + math.cos(math.pi * epoch / config["epochs"])
    ) / 2


class ModelEMA:
    """EMA weights with the current sparse topology, never stale connections."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for name, target in self.module.state_dict().items():
            if target.is_floating_point():
                target.lerp_(source[name], 1 - self.decay)
            else:
                target.copy_(source[name])
        for module in self.module.modules():
            if hasattr(module, "mask"):
                module.weight.mul_(module.mask)


def train_epoch(model, criterion, loader, optimizer, topology, scaler, device,
                config, epoch, global_step, total_steps, mixup=None, ema=None,
                log_every=50):
    model.train()
    raw_model = model.module if hasattr(model, "module") else model
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    accumulation = config["effective_batch_size"] // (config["batch_size"] * world_size)
    usable_batches = len(loader) // accumulation * accumulation
    if not usable_batches:
        raise ValueError("Dataset is too small for one effective batch.")
    rate = learning_rate(epoch, config)
    for group in optimizer.param_groups:
        group["lr"] = rate
    loss_sum = 0.0
    examples = 0
    for batch_index, (images, labels) in enumerate(loader):
        if batch_index >= usable_batches:
            break  # keep a constant effective batch size
        if batch_index % accumulation == 0:
            optimizer.zero_grad(set_to_none=True)
            topology.begin_step(global_step + 1)
        images, labels = images.to(device), labels.to(device)
        if mixup is not None:
            images, labels = mixup(images, labels)
        progress = min(1.0, global_step / max(1, total_steps - 1))
        temperature = config["temperature_start"] + progress * (
            config["temperature_end"] - config["temperature_start"]
        )
        last_microbatch = (batch_index + 1) % accumulation == 0
        sync = model.no_sync() if hasattr(model, "no_sync") and not last_microbatch else nullcontext()
        with sync:
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = model(images, temperature=temperature)
                loss = criterion(images, outputs, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {global_step}.")
            scaler.scale(loss / accumulation).backward()
        loss_sum += loss.detach().item() * images.shape[0]
        examples += images.shape[0]
        if last_microbatch:
            scaler.unscale_(optimizer)
            topology.prepare_update()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            topology.finish_step(optimizer, skipped=skipped)
            if not skipped:
                if ema is not None:
                    ema.update(raw_model)
                global_step += 1
            if rank == 0 and (global_step % log_every == 0 or batch_index + 1 == usable_batches):
                print(f"epoch={epoch + 1} step={global_step}/{total_steps} "
                      f"loss={loss_sum / examples:.4f} lr={rate:.3g} tau={temperature:.3f}", flush=True)
    metrics = torch.tensor([loss_sum, examples], dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics)
    return {"loss": (metrics[0] / metrics[1]).item(), "lr": rate}, global_step


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    # Model probabilities, as specified in the pasted DeiT section. No teacher.
    totals = torch.zeros(4, dtype=torch.float32, device=device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        probabilities = prediction_probabilities(model(images))
        top = probabilities.topk(min(5, probabilities.shape[-1]), dim=-1).indices
        totals[0] += nn.functional.nll_loss(probabilities.clamp_min(1e-12).log(), labels, reduction="sum")
        totals[1] += (top[:, 0] == labels).sum()
        totals[2] += (top == labels[:, None]).any(dim=-1).sum()
        totals[3] += labels.numel()
    if dist.is_initialized():
        dist.all_reduce(totals)
    if totals[3] == 0:
        raise ValueError("Validation dataset is empty.")
    return {"loss": (totals[0] / totals[3]).item(),
            "top1": (100 * totals[1] / totals[3]).item(),
            "top5": (100 * totals[2] / totals[3]).item()}
