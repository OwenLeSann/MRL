"""Run SViTE+-Tiny with hard distillation; use --smoke-test before ImageNet."""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset, TensorDataset
from timm.data import Mixup
from timm.data.distributed_sampler import RepeatAugSampler

from data import build_imagenet_datasets
from distillation import HardDistillationLoss
from model import SViTETiny
from sparsity import SparseTopology
from teacher import build_teacher
from training import ModelEMA, evaluate, make_optimizer, train_epoch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/svite_tiny.json")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--val-labels", type=Path, help="Kaggle LOC_val_solution.csv for a flat val directory")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/svite-tiny"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval", action="store_true", help="Evaluate --resume without loading a teacher")
    parser.add_argument("--smoke-test", action="store_true", help="Small synthetic run; no downloads")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Images per GPU per forward pass")
    parser.add_argument("--effective-batch-size", type=int, help="Total images per optimizer step")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--weight-sparsity", type=float)
    parser.add_argument("--er-epsilon", type=float, help="Literal SET probability; overrides target sparsity")
    parser.add_argument("--token-keep-rate", type=float)
    parser.add_argument("--max-epochs-this-run", type=int, help="Stop at an epoch boundary without changing the schedule")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def validate_config(config, world_size):
    micro = config["batch_size"] * world_size
    if config["batch_size"] < 1 or config["effective_batch_size"] < micro or config["effective_batch_size"] % micro:
        raise ValueError("effective_batch_size must be a positive multiple of batch_size * GPU count.")
    if (config["mixup_alpha"] > 0 or config["cutmix_alpha"] > 0) and config["batch_size"] % 2:
        raise ValueError("timm Mixup/CutMix requires an even per-device batch size.")
    if config["epochs"] < 1 or config["workers"] < 0:
        raise ValueError("epochs must be positive and workers nonnegative.")
    if not 0 <= config["warmup_epochs"] < config["epochs"]:
        raise ValueError("warmup_epochs must be less than epochs; edit the config for short experiments.")
    if not 0 <= config["weight_sparsity"] < 1 or not 0 < config["sparse_end_fraction"] <= 1:
        raise ValueError("Invalid sparsity or end fraction.")


def main():
    args = arguments()
    config = json.loads(args.config.read_text())
    if not args.resume and not args.eval and (args.output_dir / "last.pt").exists():
        raise FileExistsError("Output directory already has a checkpoint; use --resume or a new --output-dir.")
    checkpoint = None
    if args.resume:
        # Only load trusted checkpoints produced by this training script.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = checkpoint["config"].copy()
        if args.smoke_test != checkpoint["smoke_test"]:
            raise ValueError("Synthetic checkpoints require --smoke-test; they cannot resume ImageNet training.")
    for key in ("epochs", "batch_size", "effective_batch_size", "workers", "weight_sparsity", "er_epsilon", "token_keep_rate"):
        value = getattr(args, key)
        if value is not None:
            if checkpoint and key != "workers" and value != config[key]:
                raise ValueError(f"Cannot change {key} while resuming the same experiment.")
            config[key] = value
    if args.no_amp:
        config["amp"] = False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name, local_rank) if device_name == "cuda" else torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1:
        if device.type != "cuda":
            raise ValueError("Distributed runs currently require CUDA; single-device CPU/MPS are supported.")
        dist.init_process_group("nccl")
    if args.smoke_test and not checkpoint:
        config.update(image_size=32, epochs=2, warmup_epochs=0, batch_size=2,
                      effective_batch_size=4 * world_size, workers=0,
                      sparse_update_interval=1, repeated_augmentation=False)
    validate_config(config, world_size)
    if args.eval and checkpoint is None:
        raise ValueError("--eval requires --resume CHECKPOINT.")
    if args.max_epochs_this_run is not None and args.max_epochs_this_run < 1:
        raise ValueError("--max-epochs-this-run must be positive.")
    seed_all(config["seed"])
    if args.smoke_test:
        # Isolated generator keeps synthetic examples identical after a resume.
        generator = torch.Generator().manual_seed(123)
        train_set = TensorDataset(torch.randn(8 * world_size, 3, 32, 32, generator=generator),
                                  torch.randint(1000, (8 * world_size,), generator=generator))
        val_set = train_set
        class_mapping = None
    else:
        if args.data_path is None:
            raise ValueError("Supply --data-path to ImageNet, or use --smoke-test.")
        train_set, val_set = build_imagenet_datasets(args.data_path, config["image_size"], args.val_labels)
        class_mapping = train_set.class_to_idx
    if config["repeated_augmentation"]:
        sampler = RepeatAugSampler(train_set, num_replicas=world_size, rank=rank)
    else:
        sampler = torch.utils.data.DistributedSampler(train_set, num_replicas=world_size, rank=rank, seed=config["seed"])
    loader = DataLoader(train_set, sampler=sampler, batch_size=config["batch_size"],
                        num_workers=config["workers"], pin_memory=device.type == "cuda", drop_last=True)
    # Strided validation partitions avoid counting padding images twice under DDP.
    val_loader = DataLoader(Subset(val_set, list(range(rank, len(val_set), world_size))),
                            batch_size=config["batch_size"], num_workers=config["workers"],
                            pin_memory=device.type == "cuda")
    accumulation = config["effective_batch_size"] // (config["batch_size"] * world_size)
    steps_per_epoch = len(loader) // accumulation
    if steps_per_epoch < 1:
        raise ValueError("No complete effective batches available.")
    total_steps = config["epochs"] * steps_per_epoch
    if checkpoint and not args.eval and (checkpoint["total_steps"] != total_steps or checkpoint["world_size"] != world_size):
        raise ValueError("Dataset length or GPU count changed; resume would change the sparse schedule.")
    if checkpoint and checkpoint["class_to_idx"] != class_mapping:
        raise ValueError("ImageNet class mapping changed since the checkpoint.")
    model = SViTETiny(config["image_size"], config["num_classes"],
                      config["token_keep_rate"], config["drop_path_rate"])
    topology = SparseTopology(model, density=1 - config["weight_sparsity"],
                              update_interval=config["sparse_update_interval"],
                              end_step=max(1, int(config["sparse_end_fraction"] * total_steps)),
                              initial_drop_fraction=config["initial_drop_fraction"],
                              initialize=checkpoint is None, er_epsilon=config["er_epsilon"])
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    model.to(device)
    if args.eval:
        if rank == 0:
            print("student", evaluate(model, val_loader, device))
        else:
            evaluate(model, val_loader, device)
        if checkpoint["ema"] is not None:
            model.load_state_dict(checkpoint["ema"])
            metrics = evaluate(model, val_loader, device)
            if rank == 0:
                print("ema", metrics)
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and device.type == "cuda")
    ema = ModelEMA(model, config["ema_decay"]) if config["ema_decay"] else None
    # Serialize pretrained cache population across ranks on the same machine.
    if args.smoke_test:
        torch.set_num_threads(min(4, torch.get_num_threads()))
        seed_all(config["seed"] + 790)
        teacher = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1000)).to(device)
    else:
        if rank != 0 and dist.is_initialized():
            dist.barrier()
        teacher = build_teacher(device)
        if rank == 0 and dist.is_initialized():
            dist.barrier()
    criterion = HardDistillationLoss(teacher, config["label_smoothing"])
    mixup = None
    if config["mixup_alpha"] > 0 or config["cutmix_alpha"] > 0:
        mixup = Mixup(mixup_alpha=config["mixup_alpha"], cutmix_alpha=config["cutmix_alpha"],
                      prob=1.0, switch_prob=0.5, mode="batch", num_classes=config["num_classes"],
                      label_smoothing=config["label_smoothing"])
    start_epoch, global_step, best = 0, 0, -1.0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if ema:
            ema.module.load_state_dict(checkpoint["ema"])
        start_epoch, global_step, best = checkpoint["epoch"] + 1, checkpoint["global_step"], checkpoint["best_top1"]
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world_size > 1 else model
    seed_all(config["seed"] + rank)
    if checkpoint:
        restore_rng(checkpoint["rng_states"][rank], device)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps({**topology.statistics(), "kept_patch_tokens": model.kept_patch_tokens,
                          "effective_batch_size": config["effective_batch_size"],
                          "steps_per_epoch": steps_per_epoch, "sparse_end_step": topology.end_step,
                          "synthetic": args.smoke_test}), flush=True)
    stop_epoch = config["epochs"] if args.max_epochs_this_run is None else min(config["epochs"], start_epoch + args.max_epochs_this_run)
    for epoch in range(start_epoch, stop_epoch):
        sampler.set_epoch(epoch)
        metrics, global_step = train_epoch(wrapped, criterion, loader, optimizer, topology, scaler,
                                           device, config, epoch, global_step, total_steps, mixup, ema)
        validation = evaluate(model, val_loader, device)
        ema_validation = evaluate(ema.module, val_loader, device) if ema else None
        score = max(validation["top1"], ema_validation["top1"] if ema_validation else -1)
        improved = score > best
        best = max(best, score)
        states = [None] * world_size
        local_state = rng_state(device)
        if dist.is_initialized():
            dist.all_gather_object(states, local_state)
        else:
            states[0] = local_state
        if rank == 0:
            record = {"epoch": epoch + 1, "global_step": global_step, "train": metrics,
                      "validation": validation, "ema_validation": ema_validation}
            print(json.dumps(record), flush=True)
            with (args.output_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "ema": ema.module.state_dict() if ema else None,
                     "config": config, "epoch": epoch, "global_step": global_step,
                     "total_steps": total_steps, "best_top1": best, "rng_states": states,
                     "world_size": world_size, "class_to_idx": class_mapping, "smoke_test": args.smoke_test}
            temporary = args.output_dir / "checkpoint.tmp"
            torch.save(state, temporary)
            temporary.replace(args.output_dir / "last.pt")
            if improved:
                torch.save(state, args.output_dir / "best.pt")
        if dist.is_initialized():
            dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
