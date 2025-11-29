# arc_model.py

from __future__ import annotations

import copy
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.training_utils import compute_density_for_timestep_sampling


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: torch.device | None = None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class ArcFlux(nn.Module):
    """
    Diffusion model for ARC.

    Training:
      - We build a sequence [input_cells ; output_cells].
      - We treat the input cells as a FROZEN PREFIX.
      - We noise only the output region and train the model
        to predict the flow (noise - latents) on that region.

    Inference:
      - Given a test input grid, we use its flattened cells as the prefix.
      - We append random noise tokens for the (unknown) output cells.
      - We run rectified-flow sampling, freezing the prefix at every step
        and infilling the suffix.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.in_channels = config.get("in_channels", 64)  # token embedding dim
        self.num_layers = config.get("num_layers", 6)
        self.num_single_layers = config.get("num_single_layers", 18)
        self.attention_head_dim = config.get("attention_head_dim", 128)
        self.num_attention_heads = config.get("num_attention_heads", 8)
        self.joint_attention_dim = config.get("joint_attention_dim", 512)
        self.pooled_projection_dim = config.get(
            "pooled_projection_dim", self.joint_attention_dim
        )

        self.num_colors = config.get("num_colors", 10)  # 0..9
        self.pad_token_id = config.get("pad_token_id", 10)  # +1 for PAD
        self.max_seq_len = config.get("max_seq_len", 30 * 30 * 2)  # input+output

        # Noise scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        # Token embedding for colors (0..9) + PAD
        vocab_size = self.num_colors + 1
        self.token_embed = nn.Embedding(vocab_size, self.in_channels)

        # Project prefix tokens to joint_attention_dim for cross-attn
        self.cond_proj = nn.Linear(self.in_channels, self.joint_attention_dim)

        # Global pooled projection (from prefix tokens)
        self.fc = nn.Sequential(
            nn.Linear(self.joint_attention_dim, self.pooled_projection_dim),
            nn.ReLU(),
        )

        # Flux Transformer (DiT-ish core)
        self.transformer = FluxTransformer2DModel(
            in_channels=self.in_channels,
            num_layers=self.num_layers,
            num_single_layers=self.num_single_layers,
            attention_head_dim=self.attention_head_dim,
            num_attention_heads=self.num_attention_heads,
            joint_attention_dim=self.joint_attention_dim,
            pooled_projection_dim=self.pooled_projection_dim,
            guidance_embeds=False,
        )

        # Make PAD embedding nicer: start as zero vector
        with torch.no_grad():
            if 0 <= self.pad_token_id < vocab_size:
                self.token_embed.weight[self.pad_token_id].zero_()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_sigmas(self, timesteps, n_dim=3, dtype=torch.float32):
        device = self.device
        sigmas = self.noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(
        self,
        seq_ids: torch.LongTensor,     # (B, L_max)  [input ; output ; PAD]
        prefix_lens: torch.LongTensor, # (B,)        number of input tokens per sample
        target_mask: torch.BoolTensor, # (B, L_max)  True only on output tokens
    ):
        """
        One training step, Flow Matching style.

        seq_ids contains color IDs (0..9) and PAD (10).
        prefix_lens[i] tells us how many initial tokens belong to the input grid.
        target_mask[i,j] = True iff seq_ids[i,j] is part of the output grid.
        """
        device = seq_ids.device
        bsz, L_max = seq_ids.shape

        # --- embeddings for the full [input; output] sequence ---
        latents = self.token_embed(seq_ids)  # (B, L_max, D)

        # --- build encoder_hidden_states from PREFIX tokens only ---
        prefix_lens = prefix_lens.to(device)
        max_prefix = int(prefix_lens.max().item())

        cond_hidden = latents.new_zeros(bsz, max_prefix, self.joint_attention_dim)
        cond_mask = torch.zeros(bsz, max_prefix, dtype=torch.bool, device=device)

        for i in range(bsz):
            Lp = int(prefix_lens[i].item())
            if Lp == 0:
                continue
            prefix_lat = latents[i, :Lp, :]                        # (Lp, D)
            cond_hidden[i, :Lp, :] = self.cond_proj(prefix_lat)    # (Lp, joint_dim)
            cond_mask[i, :Lp] = True

        cond_mask_float = cond_mask.float()
        denom = cond_mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (cond_hidden * cond_mask_float.unsqueeze(-1)).sum(dim=1) / denom
        pooled_projection = self.fc(pooled)                       # (B, pooled_dim)

        # --- sample timesteps & sigmas ---
        noise = torch.randn_like(latents)

        u = compute_density_for_timestep_sampling(
            weighting_scheme="logit_normal",
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=None,
        )
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=device)
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)

        sigma_full = sigmas.expand(-1, L_max, -1)  # (B, L_max, 1)

        # --- apply noise ONLY to target (output) tokens ---
        target_mask = target_mask.to(device)
        target_mask_3d = target_mask.unsqueeze(-1)                 # (B, L_max, 1)

        noisy_model_input = torch.where(
            target_mask_3d,
            (1.0 - sigma_full) * latents + sigma_full * noise,
            latents,
        )

        # --- positional ids for Flux ---
        img_ids = (
            torch.arange(L_max, device=device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(bsz, 1, 3)
        )  # (B, L_max, 3)

        txt_ids = torch.zeros(
            bsz, cond_hidden.shape[1], 3, device=device
        )  # (B, max_prefix, 3)

        # --- forward Flux transformer ---
        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            encoder_hidden_states=cond_hidden,
            pooled_projections=pooled_projection,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
            timestep=timesteps / 1000.0,   # scale like your audio model
            return_dict=False,
        )[0]  # (B, L_max, D)

        # Flow Matching target: vector field from latents to noise
        flow_target = noise - latents

        mse = (model_pred.float() - flow_target.float()) ** 2
        mse = mse * target_mask_3d.float()     # ignore prefix & PAD
        loss = mse.mean()

        return loss

    @torch.no_grad()
    def sample_with_prefix(
        self,
        prefix_ids: torch.LongTensor,  # (B, L_prefix)
        out_height: int,
        out_width: int,
        num_inference_steps: int = 64,
    ) -> torch.LongTensor:
        """
        Prefix + infill sampling.

        prefix_ids: flattened input grid colors
        out_height/out_width: shape of desired output grid

        Returns:
            (B, out_height, out_width) integer color grids in [0..9]
        """
        device = self.device
        prefix_ids = prefix_ids.to(device)

        bsz, Lp = prefix_ids.shape
        L_out = out_height * out_width
        L_total = Lp + L_out

        # fixed prefix latents
        prefix_latents = self.token_embed(prefix_ids)              # (B, Lp, D)

        # initial unknown region (pure noise)
        unknown_latents = torch.randn(bsz, L_out, self.in_channels, device=device)
        latents = torch.cat([prefix_latents, unknown_latents], dim=1)  # (B, L_total, D)

        # encoder_hidden_states from prefix only
        cond_hidden = self.cond_proj(prefix_latents)               # (B, Lp, joint_dim)
        cond_mask = torch.ones(bsz, Lp, dtype=torch.bool, device=device)
        cond_mask_float = cond_mask.float()
        denom = cond_mask_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (cond_hidden * cond_mask_float.unsqueeze(-1)).sum(dim=1) / denom
        pooled_projection = self.fc(pooled)

        scheduler = self.noise_scheduler
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
        )

        img_ids = (
            torch.arange(L_total, device=device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(bsz, 1, 3)
        )

        for t in timesteps:
            txt_ids = torch.zeros(
                bsz, cond_hidden.shape[1], 3, device=device
            )
            noise_pred = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=cond_hidden,
                pooled_projections=pooled_projection,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=None,
                timestep=torch.tensor([t / 1000.0], device=device),
                return_dict=False,
            )[0]

            latents = scheduler.step(noise_pred, t, latents).prev_sample
            # re-freeze prefix at each step
            latents[:, :Lp, :] = prefix_latents

        # decode only the infilled suffix to discrete colors
        unknown_final = latents[:, Lp:, :]                          # (B, L_out, D)
        codebook = self.token_embed.weight[: self.num_colors]       # (10, D)
        logits = torch.einsum("bld,kd->blk", unknown_final, codebook)
        pred_ids = logits.argmax(dim=-1)                            # (B, L_out)

        return pred_ids.view(bsz, out_height, out_width)
