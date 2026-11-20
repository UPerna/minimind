"""Compare two CPU DDP ranks against one optimizer over all supervised positions."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.training_state import TokenAccumulator


def check():
    if dist.get_world_size() != 2:
        raise ValueError("Run with torchrun --standalone --nproc-per-node=2")
    torch.manual_seed(37)
    control = torch.nn.Linear(3, 1, bias=False)
    model = DistributedDataParallel(copy.deepcopy(control))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.2)
    reference = torch.optim.AdamW(control.parameters(), lr=0.02, weight_decay=0.2)
    args = SimpleNamespace(batch_size=2, max_seq_len=2, accumulation_steps=2, device="cpu", grad_clip=0.0)
    window = TokenAccumulator(optimizer, torch.amp.GradScaler("cpu", enabled=False), args)
    features = torch.randn(5, 3)
    targets = torch.randn(5)
    # Each window contains two microbatches on each rank. Include uneven and all-empty windows.
    for ranks in [([[0], [1, 2]], [[], []]), ([[2], [4]], [[1, 3], [0]]), ([[], []], [[], []])]:
        for indices in ranks[dist.get_rank()]:
            squared_errors = (model(features).flatten() - targets).square()
            loss = squared_errors[indices].sum() / max(len(indices), 1)
            window.backward(loss, torch.tensor(len(indices)))
        window.step()
        all_indices = [i for batches in ranks for batch in batches for i in batch]
        if all_indices:
            (control(features[all_indices]).flatten() - targets[all_indices]).square().mean().backward()
            reference.step()
            reference.zero_grad(set_to_none=True)
        torch.testing.assert_close(model.module.weight, control.weight)
        for key, value in reference.state[control.weight].items():
            torch.testing.assert_close(optimizer.state[model.module.weight][key], value)
    if dist.get_rank() == 0:
        print("Two-rank CPU gradients, weights and AdamW state matched the single-process reference.")


if __name__ == "__main__":
    dist.init_process_group("gloo")
    try:
        check()
    finally:
        dist.destroy_process_group()
