# train_arc.py

from __future__ import annotations

import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_model import ArcFlux
from arc_dataset import ArcHFDataset


def parse_args():
    p = argparse.ArgumentParser("Train ArcFlux on ARC-AGI-1")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="checkpoints_arc")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- dataset: training split, using train + test pairs ---
    dataset = ArcHFDataset(split="training", use_test_targets=True, pad_token_id=10)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # --- model config (feel free to tweak) ---
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

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # --- training loop ---
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

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
            pbar.set_postfix(loss=running_loss / global_step)

        avg_loss = running_loss / max(global_step, 1)
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

        ckpt_path = os.path.join(args.save_dir, f"arcflux_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
