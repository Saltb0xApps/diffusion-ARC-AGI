# train_arc.py
from __future__ import annotations

import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from arc_model import ArcFlux
from arc_dataset import ArcHFDataset


def parse_args():
    p = argparse.ArgumentParser("Train ArcFlux on ARC-AGI-1 with W&B logging")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="checkpoints_arc")

    # wandb
    p.add_argument("--wandb_project", type=str, default="diffusion-arc-agi")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging")

    return p.parse_args()


def build_model(device: torch.device) -> ArcFlux:
    model_config = {
        "num_layers": 6,
        "num_single_layers": 18,
        "in_channels": 64,
        "attention_head_dim": 128,
        "num_attention_heads": 8,
        "joint_attention_dim": 512,
        "pooled_projection_dim": 512,
        "max_seq_len": 30 * 30 * 2,
        "num_colors": 10,
        "pad_token_id": 10,
    }
    model = ArcFlux(model_config).to(device)
    return model


def evaluate_loss(model: ArcFlux, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            seq_ids, prefix_lens, target_mask, out_shapes, sample_ids = batch
            seq_ids = seq_ids.to(device)
            prefix_lens = prefix_lens.to(device)
            target_mask = target_mask.to(device)

            loss = model(seq_ids, prefix_lens, target_mask)
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- datasets ---
    train_dataset = ArcHFDataset(
        split="training",
        use_test_targets=True,
        pad_token_id=10,
    )
    eval_dataset = ArcHFDataset(
        split="evaluation",
        use_test_targets=True,
        pad_token_id=10,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=eval_dataset.collate_fn,
    )

    # --- model + optimizer ---
    model = build_model(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- wandb init ---
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
        )
        wandb.watch(model, log="gradients", log_freq=100)

    # --- training loop ---
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            seq_ids, prefix_lens, target_mask, out_shapes, sample_ids = batch

            seq_ids = seq_ids.to(device)
            prefix_lens = prefix_lens.to(device)
            target_mask = target_mask.to(device)

            optimizer.zero_grad()
            loss = model(seq_ids, prefix_lens, target_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            avg_loss = running_loss / global_step
            pbar.set_postfix(loss=avg_loss)

            if use_wandb and global_step % 10 == 0:
                wandb.log({"train/loss_step": loss.item(), "step": global_step})

        # --- end-of-epoch metrics ---
        avg_train_loss = running_loss / max(global_step, 1)
        eval_loss = evaluate_loss(model, eval_loader, device)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={avg_train_loss:.4f}, eval_loss={eval_loss:.4f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss_epoch": avg_train_loss,
                    "eval/loss_epoch": eval_loss,
                    "step": global_step,
                }
            )

        # --- checkpoint ---
        ckpt_path = os.path.join(args.save_dir, f"arcflux_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
