from __future__ import annotations

import argparse
import os
import warnings
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from datasets import load_dataset

from arc_model import ArcFlux
from arc_dataset import ArcHFDataset
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity_error()

# Silence most torch-related warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch.*")


def parse_args():
    p = argparse.ArgumentParser("Train ArcFlux on ARC-AGI-1 with W&B logging")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="checkpoints_arc")

    # wandb
    p.add_argument("--wandb_project", type=str, default="diffusion-arc-agi")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging")

    # eval settings
    p.add_argument("--eval_steps", type=int, default=64,
                   help="Diffusion steps to use for full eval")
    p.add_argument("--full_eval_interval", type=int, default=5,
                   help="Run full eval every N epochs (default 5)")

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

    # gradient checkpointing saves memory
    if hasattr(model.transformer, "enable_gradient_checkpointing"):
        model.transformer.enable_gradient_checkpointing()

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


def grids_equal(a, b):
    if len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return False
        for x, y in zip(row_a, row_b):
            if x != y:
                return False
    return True


def grid_to_ascii(grid):
    """
    Represent a 2D integer grid as a string of numbers, one row per line.
    """
    arr = np.asarray(grid)
    lines = []
    for row in arr:
        # use fixed width so columns align a bit
        lines.append(" ".join(f"{int(v):2d}" for v in row))
    return "\n".join(lines)

def evaluate_accuracy_full(
    model: ArcFlux,
    device: torch.device,
    num_steps: int = 32,
) -> float:
    """
    Full accuracy eval on ARC-AGI-1 evaluation split, like eval_arc.py,
    but without writing JSON to disk.

    Modified to:
      - evaluate on 40 random evaluation tasks
      - print ASCII art of input, prediction and ground truth
    """
    model.eval()
    ds_eval = load_dataset("dataartist/arc-agi")["evaluation"]

    MAX_TASKS = 40  # evaluate on 40 random tasks from the 400-task evaluation split

    num_tasks = len(ds_eval)
    k = min(MAX_TASKS, num_tasks)

    rng = random.Random(42)  # fixed seed for reproducibility; change if desired
    selected_indices = rng.sample(range(num_tasks), k=k)

    total = 0
    correct = 0

    with torch.no_grad():
        for idx in selected_indices:
            row = ds_eval[idx]
            task_id = row["id"]

            for i, test_pair in enumerate(row["test"]):
                inp = torch.tensor(test_pair["input"], dtype=torch.long)   # (H_in, W_in)
                gt_out = test_pair["output"]

                H_out = len(gt_out)
                W_out = len(gt_out[0]) if H_out > 0 else 0

                prefix_ids = inp.view(1, -1).to(device)  # (1, L_prefix)

                pred_ids = model.sample_with_prefix(
                    prefix_ids,
                    out_height=H_out,
                    out_width=W_out,
                    num_inference_steps=num_steps,
                )
                pred_grid = pred_ids.cpu().numpy()[0].tolist()

                is_correct = grids_equal(pred_grid, gt_out)
                if is_correct:
                    correct += 1
                total += 1

                # ASCII art printout
                print(f"\n[Full eval] Task {task_id} (idx {idx}) - test {i} - {'CORRECT' if is_correct else 'WRONG'}")
                print("INPUT:")
                print(grid_to_ascii(test_pair["input"]))
                print("\nPREDICTED:")
                print(grid_to_ascii(pred_grid))
                print("\nGROUND TRUTH:")
                print(grid_to_ascii(gt_out))
                print("-" * 40)

    acc = correct / max(total, 1)
    print(
        f"\n[Full eval] Exact grid accuracy on random subset of evaluation split "
        f"({k} tasks out of {num_tasks}): "
        f"{acc*100:.2f}% ({correct}/{total})"
    )

    model.train()
    return acc


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- datasets ---
    train_dataset = ArcHFDataset(
        split="train",
        use_test_targets=True,
        pad_token_id=10,
        dataset_id="Asap7772/arc-agi-mixed-barc",
    )
    eval_dataset = ArcHFDataset(
        split="evaluation",
        use_test_targets=True,
        pad_token_id=10,
        dataset_id="dataartist/arc-agi",
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

    # --- wandb ---
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
                "eval_steps": args.eval_steps,
                "full_eval_interval": args.full_eval_interval,
            },
        )
        wandb.watch(model, log="gradients", log_freq=100)

    # --- training loop ---
    global_step = 0
    running_loss = 0.0
    for epoch in range(args.epochs):
        model.train()
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

        # --- end-of-epoch eval loss (always) ---
        avg_train_loss = running_loss / max(global_step, 1)
        eval_loss = evaluate_loss(model, eval_loader, device)

        epoch_num = epoch + 1
        run_full_eval = (
            (epoch_num % args.full_eval_interval == 0)
            or (epoch_num == args.epochs)
        )

        if run_full_eval:
            eval_acc = evaluate_accuracy_full(
                model, device, num_steps=args.eval_steps
            )
            print(
                f"Epoch {epoch_num}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"eval_loss={eval_loss:.4f}, "
                f"FULL eval_acc={eval_acc*100:.2f}%"
            )
        else:
            eval_acc = None
            print(
                f"Epoch {epoch_num}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"eval_loss={eval_loss:.4f} "
                f"(full eval skipped this epoch)"
            )

        if use_wandb:
            log_dict = {
                "epoch": epoch_num,
                "train/loss_epoch": avg_train_loss,
                "eval/loss_epoch": eval_loss,
                "step": global_step,
            }
            if eval_acc is not None:
                log_dict["eval/accuracy_full_epoch"] = eval_acc
            wandb.log(log_dict)

        # --- checkpoint ---
        ckpt_path = os.path.join(args.save_dir, f"arcflux_epoch_{epoch_num}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
