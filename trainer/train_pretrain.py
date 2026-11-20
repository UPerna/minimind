import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model
from trainer.training_state import epoch_loader, training_batches, TokenAccumulator

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    window = TokenAccumulator(optimizer, scaler, args)
    for step, (input_ids, labels) in training_batches(loader, epoch, start_step, args.seed):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss

        window.backward(loss, labels[..., 1:].ne(-100).sum())

        if step % args.accumulation_steps == 0 or step == iters:
            window.step()

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item()
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         epoch_batches=iters, batch_size=args.batch_size, accumulation_steps=args.accumulation_steps)
            model.train()
            del state_dict

        del input_ids, labels, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="Directory for saved models")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="Prefix for saved weight filenames")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Mixed-precision data type")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between log messages")
    parser.add_argument("--save_interval", type=int, default=1000, help="Steps between model saves")
    parser.add_argument('--hidden_size', default=768, type=int, help="Hidden size")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers")
    parser.add_argument('--max_seq_len', default=340, type=int, help="Maximum training sequence length (roughly 1.5 to 1.7 Chinese characters per token)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Use the MoE architecture (0 = no, 1 = yes)")
    parser.add_argument('--seed', default=42, type=int, help="Seed for initialization, data order and batch replay")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="Path to pretraining data")
    parser.add_argument('--from_weight', default='none', type=str, help="Starting weights; use none to train from scratch")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="Automatically detect and resume a checkpoint (0 = no, 1 = yes)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable SwanLab logging")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="SwanLab project name")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="Use torch.compile acceleration (0 = no, 1 = yes)")
    args = parser.parse_args()

    # ========== 1. Initialize the environment and random seed. ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. Set directories and model parameters, and look for a checkpoint. ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. Configure mixed precision. ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. Configure experiment logging. ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. Define the model, dataset, and optimizer. ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds, seed=args.seed) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. Restore checkpoint state. ==========
    start_epoch = 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
    
    # ========== 7. Compile and wrap for distributed training. ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. Start training. ==========
    for epoch in range(start_epoch, args.epochs):
        loader, skip = epoch_loader(train_ds, train_sampler, args, epoch, ckp_data)
        ckp_data = None
        train_epoch(epoch, loader, len(loader) + skip, skip, wandb)
    
    # ========== 9. Clean up distributed processes. ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()