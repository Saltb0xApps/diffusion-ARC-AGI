# eval_arc.py
from __future__ import annotations

import argparse
import json

import torch
from datasets import load_dataset
from tqdm import tqdm

from arc_model import ArcFlux


def parse_args():
    p = argparse.ArgumentParser("Evaluate ArcFlux on ARC-AGI-1 evaluation split")
    p.add_argument("--ckpt", type=str, required=True,
                  help="Path to model checkpoint (.pt) from train_arc.py")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_steps", type=int, default=64,
                  help="Diffusion inference steps")
    p.add_argument("--out", type=str, default="arc_eval_predictions.json")
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


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = build_model(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds_eval = load_dataset("dataartist/arc-agi")["evaluation"]

    all_results = {}
    total = 0
    correct = 0

    for row in tqdm(ds_eval, desc="Evaluating tasks"):
        task_id = row["id"]
        task_results = []

        # we do NOT feed the train examples as prefix here; we only use the test input itself
        for i, test_pair in enumerate(row["test"]):
            inp = torch.tensor(test_pair["input"], dtype=torch.long)   # (H_in, W_in)
            gt_out = test_pair["output"]

            H_out = len(gt_out)
            W_out = len(gt_out[0]) if H_out > 0 else 0

            prefix_ids = inp.view(1, -1)  # (1, L_prefix)

            with torch.no_grad():
                pred_grid_ids = model.sample_with_prefix(
                    prefix_ids.to(device),
                    out_height=H_out,
                    out_width=W_out,
                    num_inference_steps=args.num_steps,
                )

            pred_grid = pred_grid_ids.cpu().numpy()[0].tolist()

            is_correct = grids_equal(pred_grid, gt_out)
            total += 1
            correct += int(is_correct)

            task_results.append(
                {
                    "test_index": i,
                    "input": test_pair["input"],
                    "predicted": pred_grid,
                    "ground_truth": gt_out,
                    "correct": bool(is_correct),
                }
            )

        all_results[task_id] = task_results

    accuracy = correct / max(total, 1)
    print(f"Exact grid accuracy on evaluation split: {accuracy*100:.2f}% "
          f"({correct}/{total})")

    with open(args.out, "w") as f:
        json.dump(
            {
                "checkpoint": args.ckpt,
                "accuracy": accuracy,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"Wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
