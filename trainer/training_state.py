"""Data order, accumulation, and replay for the training entry points."""

import hashlib
import math
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from trainer.trainer_utils import Logger, SkipBatchSampler


def resume_batch(checkpoint, examples, batch_size, accumulation_steps, world_size):
    """Map the last completed optimizer window to the current loader's batch index."""
    if not checkpoint:
        return 0
    progress = checkpoint.get("progress", {})
    old_world = checkpoint.get("world_size", 1)
    old_batch = progress.get("batch_size", batch_size)
    total = math.ceil(examples / (old_world * old_batch))
    observed = checkpoint.get("original_step", checkpoint.get("step", 0))
    if "completed_step" in progress:
        completed = progress["completed_step"]
        total = progress["epoch_batches"]
    else:
        # Older checkpoints contain weights and optimizer slots, but no partial gradients.
        completed = observed if observed == total else observed - observed % accumulation_steps
    if completed < observed:
        Logger(f"Replaying unfinished accumulation window after batch {completed} (saved at {observed}).")
    new_total = math.ceil(examples / (world_size * batch_size))
    if completed >= total:
        return new_total
    return min(new_total, completed * old_world * old_batch // (world_size * batch_size))


def epoch_loader(dataset, sampler, args, epoch, checkpoint=None, *, collate_fn=None, rollout_complete=False):
    if args.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    seed = args.seed + epoch
    generator = torch.Generator().manual_seed(seed)
    if hasattr(dataset, "epoch_seed"):
        dataset.epoch_seed = seed
    if sampler is not None:
        sampler.set_epoch(epoch)
        indices = sampler
    else:
        indices = torch.randperm(len(dataset), generator=generator).tolist()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    window = 1 if rollout_complete else args.accumulation_steps
    skip = resume_batch(checkpoint, len(dataset), args.batch_size, window, world_size)
    batches = SkipBatchSampler(indices, args.batch_size, skip)
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=args.num_workers, generator=generator,
                        collate_fn=collate_fn, pin_memory=str(args.device).startswith("cuda"))
    if skip:
        Logger(f"Epoch {epoch + 1}: resuming after {skip} completed batches.")
    return loader, skip


def training_batches(loader, epoch, start_step, seed=42):
    """Replay sampling and dropout on the same local rank without serializing worker RNGs."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    for step, batch in enumerate(loader, start_step + 1):
        key = f"{seed}:{epoch}:{step}:{rank}".encode("ascii")
        step_seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        random.seed(step_seed)
        np.random.seed(step_seed % 2**32)
        torch.manual_seed(step_seed)
        yield step, batch


def window_batches(step, total, accumulation_steps, start_step=0):
    """Number of microbatches contributing to the optimizer update containing this step."""
    start = (step - 1) // accumulation_steps * accumulation_steps
    return min(total, start + accumulation_steps) - max(start, start_step)


class TokenAccumulator:
    """Average gradients over the actual supervised tokens across microbatches and ranks."""

    def __init__(self, optimizer, scaler, args):
        self.optimizer = optimizer
        self.scaler = scaler
        self.grad_clip = args.grad_clip
        self.capacity = args.batch_size * args.max_seq_len * args.accumulation_steps
        self.tokens = torch.zeros((), dtype=torch.long, device=args.device)

    def backward(self, mean_loss, valid_tokens):
        self.tokens.add_(valid_tokens)
        # Bound the magnitude during AMP accumulation, then normalize once per update.
        self.scaler.scale(mean_loss * (valid_tokens / self.capacity)).backward()

    def step(self):
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(self.tokens)
        count = self.tokens.item()
        if count:
            params = [p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
            # DDP has already averaged gradients over ranks on each backward pass.
            factor = self.capacity * world_size / count
            for p in params:
                p.grad.mul_(factor)
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            Logger("Skipped optimizer update: no supervised tokens in this window.")
        self.optimizer.zero_grad(set_to_none=True)
        self.tokens.zero_()
