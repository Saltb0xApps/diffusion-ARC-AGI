# train_arc.py
from __future__ import annotations

import argparse
import os
import random

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from datasets import load_dataset

from arc_model import ArcFlux
from arc_dataset import (
    ArcHFDataset,
    MAX_SEQ_LEN,
    MAX_CONTEXT_PAIRS,
    PAD_TOKEN_ID,
    SEP_TOKEN_ID,
)


def parse_args():
    p = argparse.ArgumentParser("Train ArcFlux on ARC-AGI-1 with W&B + full eval")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="checkpoints_arc")

    # wandb
    p.add_argument("--wandb_project", type=str, default="diffusion-arc-agi")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging")

    # eval settings
    p.add_argument("--eval_steps", type=int, default=100,
                   help="Diffusion steps to use for full eval")
    p.add_argument("--eval_samples", type=int, default=4,
                   help="Number of samples per test grid for full eval")
    p.add_argument("--full_eval_interval", type=int, default=2,
                   help="Run full eval every N epochs")
    p.add_argument("--eval_max_tasks", type=int, default=40,
                   help="Max number of random eval tasks to use in full eval")
    p.add_argument("--eval_seed", type=int, default=0,
                   help="Base random seed for sampling eval tasks")

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
        "max_seq_len": MAX_SEQ_LEN,
        "num_colors": 10,
        "pad_token_id": PAD_TOKEN_ID,
    }
    model = ArcFlux(model_config).to(device)

    # gradient checkpointing saves memory (if available)
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


def print_grid(name: str, grid):
    """
    Print a 2D grid (list of lists of ints) as ASCII.
    """
    print(f"{name}:")
    for row in grid:
        print(" " + " ".join(str(int(x)) for x in row))
    print()


def build_eval_prefix_tokens(
    row: dict,
    test_pair: dict,
    max_context_pairs: int = MAX_CONTEXT_PAIRS,  # unused, kept for signature
) -> list[int]:
    """
    Eval-time prefix (matches dataset now):
      [SEP, test_input_flattened]
    """
    prefix_tokens: list[int] = []
    test_inp = torch.tensor(test_pair["input"], dtype=torch.long).flatten().tolist()
    prefix_tokens.append(SEP_TOKEN_ID)
    prefix_tokens.extend(test_inp)
    return prefix_tokens


def evaluate_accuracy_full(
    model: ArcFlux,
    device: torch.device,
    num_steps: int = 100,
    num_samples: int = 4,
    max_tasks: int = 40,
    seed: int = 0,
) -> float:
    """
    Full accuracy eval on a RANDOM subset of ARC-AGI-1 evaluation tasks, using:
      - num_steps diffusion steps,
      - num_samples samples per grid,
      - up to max_tasks tasks sampled with given seed,
    and printing ASCII grids for a subset of cases.
    """
    model.eval()
    raw = load_dataset("dataartist/arc-agi")["evaluation"]

    # sample tasks
    indices = list(range(len(raw)))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    indices = indices[: max_tasks]
    ds_eval = [raw[i] for i in indices]

    total = 0
    correct = 0

    # limit the number of grids we print to avoid spam
    max_print = 20
    printed = 0

    outer = tqdm(
        ds_eval,
        desc=f"Full eval ({num_steps} steps, {num_samples} samples, {len(ds_eval)} tasks)",
        total=len(ds_eval),
    )

    with torch.no_grad():
        for row in outer:
            task_id = row["id"]
            for j, test_pair in enumerate(row["test"]):
                gt_out = test_pair["output"]
                H_out = len(gt_out)
                W_out = len(gt_out[0]) if H_out > 0 else 0
                target_len = H_out * W_out

                # ---- build prefix: just this test input ----
                prefix_tokens = build_eval_prefix_tokens(row, test_pair)
                prefix_len = len(prefix_tokens)
                if prefix_len + target_len > MAX_SEQ_LEN:
                    max_prefix_len = MAX_SEQ_LEN - target_len
                    if max_prefix_len <= 0:
                        prefix_tokens = []
                    else:
                        prefix_tokens = prefix_tokens[-max_prefix_len:]

                prefix_ids = torch.tensor(
                    prefix_tokens, dtype=torch.long, device=device
                ).unsqueeze(0)  # (1, L_prefix)

                success = False
                first_pred = None
                final_pred = None
                hit_sample = None

                for s in range(num_samples):
                    pred_ids = model.sample_with_prefix(
                        prefix_ids,
                        out_height=H_out,
                        out_width=W_out,
                        num_inference_steps=num_steps,
                    )
                    pred_grid = pred_ids.cpu().numpy()[0].tolist()
                    if first_pred is None:
                        first_pred = pred_grid

                    if grids_equal(pred_grid, gt_out):
                        success = True
                        final_pred = pred_grid
                        hit_sample = s + 1
                        break

                if success:
                    print(
                        f"[PASS] task {task_id} test {j} "
                        f"(hit at sample {hit_sample}/{num_samples})"
                    )
                else:
                    print(
                        f"[FAIL] task {task_id} test {j} "
                        f"(no match in {num_samples} samples)"
                    )

                # Print grids for first few cases
                if printed < max_print:
                    pred_to_show = final_pred if (success and final_pred is not None) else first_pred
                    print_grid("INPUT", test_pair["input"])
                    if pred_to_show is not None:
                        print_grid("PRED", pred_to_show)
                    print_grid("GT", gt_out)
                    print("-" * 40)
                    printed += 1

                correct += int(success)
                total += 1
                outer.set_postfix(acc=f"{correct / max(total, 1):.3f}")

    model.train()
    return correct / max(total, 1)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- datasets ---
    train_dataset = ArcHFDataset(
        split="training",
        use_test_targets=True,
        pad_token_id=PAD_TOKEN_ID,
        max_context_pairs=MAX_CONTEXT_PAIRS,
        max_seq_len=MAX_SEQ_LEN,
    )
    eval_dataset = ArcHFDataset(
        split="evaluation",
        use_test_targets=True,
        pad_token_id=PAD_TOKEN_ID,
        max_context_pairs=MAX_CONTEXT_PAIRS,
        max_seq_len=MAX_SEQ_LEN,
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
                "eval_samples": args.eval_samples,
                "full_eval_interval": args.full_eval_interval,
                "eval_max_tasks": args.eval_max_tasks,
                "max_seq_len": MAX_SEQ_LEN,
                "max_context_pairs": MAX_CONTEXT_PAIRS,
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

        # --- end-of-epoch eval loss (always) ---
        avg_train_loss = running_loss / max(global_step, 1)
        eval_loss = evaluate_loss(model, eval_loader, device)

        epoch_num = epoch + 1
        run_full_eval = (
            (epoch_num % args.full_eval_interval == 0)
            or (epoch_num == args.epochs)
        )

        if run_full_eval:
            # vary seed by epoch so we see different task subsets
            eval_acc = evaluate_accuracy_full(
                model,
                device,
                num_steps=args.eval_steps,
                num_samples=args.eval_samples,
                max_tasks=args.eval_max_tasks,
                seed=args.eval_seed + epoch_num,
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
