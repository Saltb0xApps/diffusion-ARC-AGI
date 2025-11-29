# arc_dataset.py
from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# --- shared token + length constants ---
NUM_COLORS = 10           # 0..9
PAD_TOKEN_ID = 10         # padding
SEP_TOKEN_ID = 11         # separator for grids / IO blocks
MAX_CONTEXT_PAIRS = 0     # DISABLED: no extra IO context, just the current input
MAX_SEQ_LEN = 2048        # total sequence length (prefix + target + PAD)


class ArcHFDataset(Dataset):
    """
    ARC-AGI dataset from dataartist/arc-agi.

    For each sample:
      - Choose a target pair (input, output) from this task (train+test if use_test_targets=True).
      - Build prefix tokens:

          [SEP, target_input_flattened]

      - Suffix tokens (to be inpainted) = target_output_flattened.

      - If prefix + suffix > max_seq_len, crop prefix from the left so that
        prefix_len + target_len <= max_seq_len.

    Returns:
      seq_ids:     (MAX_SEQ_LEN,) long
      prefix_len:  () long scalar (how many tokens belong to prefix)
      target_mask: (MAX_SEQ_LEN,) bool (True where suffix tokens live)
      out_shape:   (2,) long [H_out, W_out]
      sample_id:   string (task_id:local_idx)
    """

    def __init__(
        self,
        split: str = "training",
        use_test_targets: bool = True,
        pad_token_id: int = PAD_TOKEN_ID,
        max_context_pairs: int = MAX_CONTEXT_PAIRS,  # kept for API, but unused now
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.split = split
        self.pad_token_id = pad_token_id
        self.sep_token_id = SEP_TOKEN_ID
        self.max_seq_len = max_seq_len
        self.use_test_targets = use_test_targets

        self.hf_ds = load_dataset("dataartist/arc-agi")[split]

        # index: list of (task_idx, target_idx, n_train, n_all)
        self.index: List[tuple[int, int, int, int]] = []
        for task_idx, row in enumerate(self.hf_ds):
            train_pairs = row["train"]
            test_pairs = row["test"]
            all_pairs = train_pairs + (test_pairs if use_test_targets else [])
            n_train = len(train_pairs)
            n_all = len(all_pairs)
            for local_idx in range(n_all):
                self.index.append((task_idx, local_idx, n_train, n_all))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        task_idx, target_idx, n_train, n_all = self.index[idx]
        row = self.hf_ds[task_idx]
        train_pairs = row["train"]
        test_pairs = row["test"]
        all_pairs = train_pairs + (test_pairs if self.use_test_targets else [])

        target_pair = all_pairs[target_idx]

        # --- target input/output ---
        t_inp_arr = np.array(target_pair["input"], dtype=np.int64)
        t_out_arr = np.array(target_pair["output"], dtype=np.int64)

        H_out, W_out = t_out_arr.shape
        t_inp_flat = t_inp_arr.flatten()
        t_out_flat = t_out_arr.flatten()

        # --- prefix = [SEP, input_flat] ---
        prefix_tokens = [self.sep_token_id]
        prefix_tokens.extend(t_inp_flat.tolist())

        prefix_len = len(prefix_tokens)
        target_len = t_out_flat.size
        max_total = self.max_seq_len

        # --- crop prefix if too long ---
        if prefix_len + target_len > max_total:
            max_prefix_len = max_total - target_len
            if max_prefix_len <= 0:
                prefix_tokens = []
                prefix_len = 0
            else:
                prefix_tokens = prefix_tokens[-max_prefix_len:]
                prefix_len = len(prefix_tokens)

        seq = prefix_tokens + t_out_flat.tolist()
        L = len(seq)

        seq_ids = torch.full((self.max_seq_len,), self.pad_token_id, dtype=torch.long)
        seq_ids[:L] = torch.tensor(seq, dtype=torch.long)

        target_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        start = prefix_len
        end = min(prefix_len + target_len, self.max_seq_len)
        target_mask[start:end] = True

        out_shape = torch.tensor([H_out, W_out], dtype=torch.long)
        sample_id = f"{row['id']}:{target_idx}"
        prefix_len_tensor = torch.tensor(prefix_len, dtype=torch.long)

        return seq_ids, prefix_len_tensor, target_mask, out_shape, sample_id

    @staticmethod
    def collate_fn(batch):
        seq_ids = torch.stack([b[0] for b in batch], dim=0)      # (B, L)
        prefix_lens = torch.stack([b[1] for b in batch], dim=0)  # (B,)
        target_masks = torch.stack([b[2] for b in batch], dim=0) # (B, L)
        out_shapes = torch.stack([b[3] for b in batch], dim=0)   # (B, 2)
        sample_ids = [b[4] for b in batch]                       # list[str]
        return seq_ids, prefix_lens, target_masks, out_shapes, sample_ids
