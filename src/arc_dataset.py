# arc_dataset.py

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

import random
import json


class ArcHFDataset(Dataset):
    """
    Wraps ARC-AGI-1 from Hugging Face.

    Each item is one (input, output) pair from a task's train or test list,
    turned into a sequence [input_flat ; output_flat].

    Args:
        split: "train" or "evaluation"
        use_test_targets:
            If True, also use the 'test' pairs from that split as supervised
            training examples (useful for "train on train+test" on the training
            split, as you mentioned).
        pad_token_id: which token index is used as PAD (must match model).
    """

    def __init__(
        self,
        split: str = "train",
        use_test_targets: bool = True,
        pad_token_id: int = 10,
        dataset_id: str = "Asap7772/arc-agi-mixed-barc",
    ):
        super().__init__()
        self.split = split
        self.pad_token_id = pad_token_id

        hf_ds = load_dataset(dataset_id)[split]  # 9.6k rows

        self.samples: List[Dict[str, Any]] = []

        for row in hf_ds:
            # check if row train is string or list
            if isinstance(row["train"], str):
                train_pairs = json.loads(row["train"])
            else:
                train_pairs = row["train"]

            if isinstance(row["test"], str):
                test_pairs = json.loads(row["test"])
            else:
                test_pairs = row["test"]

            # you can make a synthetic task_id from row index or hash
            task_id = row.get("task_id", None)
            if task_id is None:
                # fall back to a deterministic synthetic id if needed
                task_id = row.get("id", None) or f"mixed_{len(self.samples)}"

            # training pairs for the task
            for k, pair in enumerate(train_pairs):
                self.samples.append(
                    {
                        "input": np.asarray(pair["input"], dtype=np.int64),
                        "output": np.asarray(pair["output"], dtype=np.int64),
                        "task_id": task_id,
                        "kind": "train",
                        "idx": k,
                    }
                )
            # optionally also use test pairs as supervised examples
            if use_test_targets:
                for k, pair in enumerate(test_pairs):
                    self.samples.append(
                        {
                            "input": np.asarray(pair["input"], dtype=np.int64),
                            "output": np.asarray(pair["output"], dtype=np.int64),
                            "task_id": task_id,
                            "kind": "test",
                            "idx": k,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def collate_fn(self, batch):
        """
        For each sample:
          - flatten input and output grids
          - concatenate into seq = [input_flat ; output_flat]
          - record prefix_len = len(input_flat) and where the output lives
        Returns:
          seq_ids:    (B, L_max) long
          prefix_lens:(B,) long
          target_mask:(B, L_max) bool
          out_shapes: (B, 2) long  (H_out, W_out)
          sample_ids: list[str]
        """
        pad_id = self.pad_token_id
        B = len(batch)

        seqs: List[torch.LongTensor] = []
        prefix_lens_list: List[int] = []
        target_lens_list: List[int] = []
        out_shapes = torch.zeros(B, 2, dtype=torch.long)
        sample_ids: List[str] = []

        for i, sample in enumerate(batch):
            inp = torch.from_numpy(sample["input"]).long()
            out = torch.from_numpy(sample["output"]).long()

            H_out, W_out = out.shape
            out_shapes[i, 0] = H_out
            out_shapes[i, 1] = W_out

            inp_flat = inp.view(-1)   # (L_in,)
            out_flat = out.view(-1)   # (L_out,)
            seq = torch.cat([inp_flat, out_flat], dim=0)  # (L_in+L_out,)

            seqs.append(seq)
            prefix_lens_list.append(inp_flat.numel())
            target_lens_list.append(out_flat.numel())
            sample_ids.append(f"{sample['task_id']}:{sample['kind']}:{sample['idx']}")

        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.long)
        target_lens = torch.tensor(target_lens_list, dtype=torch.long)

        L_max = max(s.size(0) for s in seqs)
        seq_ids = torch.full((B, L_max), pad_id, dtype=torch.long)
        target_mask = torch.zeros((B, L_max), dtype=torch.bool)

        for i, seq in enumerate(seqs):
            L = seq.size(0)
            seq_ids[i, :L] = seq
            start = prefix_lens[i].item()
            end = start + target_lens[i].item()
            target_mask[i, start:end] = True

        return seq_ids, prefix_lens, target_mask, out_shapes, sample_ids
