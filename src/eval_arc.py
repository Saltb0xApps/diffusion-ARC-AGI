# eval_arc.py
from __future__ import annotations

import argparse
import random

import torch
from datasets import load_dataset
from tqdm import tqdm

from arc_model import ArcFlux
from arc_dataset import (
    MAX_SEQ_LEN,
    MAX_CONTEXT_PAIRS,  # still imported but effectively 0
    PAD_TOKEN_ID,
    SEP_TOKEN_ID,
)


def parse_args():
    p = argparse.ArgumentParser("Standalone evaluation of ArcFlux on ARC-AGI-1")
    p.add_argument("--ckpt", type=str, required=True,
                  help="Path to model checkpoint (.pt) from train_arc.py")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_steps", type=int, default=100,
                  help="Diffusion inference steps")
    p.add_argument("--num_samples", type=int, default=8,
                  help="Number of samples per test grid")
    p.add_argument("--max_tasks", type=int, default=40,
                  help="Max number of random eval tasks")
    p.add_argument("--seed", type=int, default=0,
                  help="Random seed for task sampling")
    return p.parse_args()


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
    print(f"{name}:")
    for row in grid:
        print(" " + " ".join(str(int(x)) for x in row))
    print()


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
    if hasattr(model.transformer, "enable_gradient_checkpointing"):
        model.transformer.enable_gradient_checkpointing()
    return model


def build_eval_prefix_tokens(
    row: dict,
    test_pair: dict,
    max_context_pairs: int = MAX_CONTEXT_PAIRS,  # unused, kept for signature
) -> list[int]:
    """
    Eval-time prefix:
      [SEP, test_input_flattened]
    """
    prefix_tokens: list[int] = []
    test_inp = torch.tensor(test_pair["input"], dtype=torch.long).flatten().tolist()
    prefix_tokens.append(SEP_TOKEN_ID)
    prefix_tokens.extend(test_inp)
    return prefix_tokens


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    raw = load_dataset("dataartist/arc-agi")["evaluation"]

    # sample tasks
    indices = list(range(len(raw)))
    rnd = random.Random(args.seed)
    rnd.shuffle(indices)
    indices = indices[: args.max_tasks]
    ds_eval = [raw[i] for i in indices]

    total = 0
    correct = 0

    outer = tqdm(
        ds_eval,
        desc=f"Eval ({args.num_steps} steps, {args.num_samples} samples, {len(ds_eval)} tasks)",
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
                ).unsqueeze(0)

                success = False
                first_pred = None
                final_pred = None
                hit_sample = None

                for s in range(args.num_samples):
                    pred_ids = model.sample_with_prefix(
                        prefix_ids,
                        out_height=H_out,
                        out_width=W_out,
                        num_inference_steps=args.num_steps,
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
                        f"(hit at sample {hit_sample}/{args.num_samples})"
                    )
                else:
                    print(
                        f"[FAIL] task {task_id} test {j} "
                        f"(no match in {args.num_samples} samples)"
                    )

                pred_to_show = final_pred if (success and final_pred is not None) else first_pred
                print_grid("INPUT", test_pair["input"])
                if pred_to_show is not None:
                    print_grid("PRED", pred_to_show)
                print_grid("GT", gt_out)
                print("-" * 40)

                correct += int(success)
                total += 1
                outer.set_postfix(acc=f"{correct / max(total, 1):.3f}")

    accuracy = correct / max(total, 1)
    print(f"Exact grid accuracy on sampled tasks: {accuracy*100:.2f}% "
          f"({correct}/{total})")


if __name__ == "__main__":
    main()
