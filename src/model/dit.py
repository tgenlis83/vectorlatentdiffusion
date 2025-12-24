from __future__ import annotations

import argparse
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from src.dataset.dataset import QuickDrawGraphDataModule
from src.model.vae import VectorGraphRVQVAE

# =============================================================================
# ------------------------ RoPE + Patchify (DiT-XL/2) -------------------------
# =============================================================================

def rope_angles(pos: torch.Tensor, dim: int, base: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pos: [...], can be float or int positions
    returns cos,sin with shape [..., dim/2]
    """
    half = dim // 2
    inv_freq = (base ** (-torch.arange(0, half, device=pos.device, dtype=torch.float32) / half))
    angles = pos.unsqueeze(-1).float() * inv_freq  # [..., half]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    x: [*, dim] where dim even, applies rotary embedding using pos: [*] broadcastable.
    """
    dim = x.shape[-1]
    assert dim % 2 == 0, "RoPE requires even dim"
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    cos, sin = rope_angles(pos, dim)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Patchify1D(nn.Module):
    """
    Patchify along token axis with patch size P=2 (DiT-XL/2).
    Input:  x [B,T,D], mask [B,T]
    Output: xp [B,Tp,D*P], mp [B,Tp]
    """
    def __init__(self, patch_size: int = 2):
        super().__init__()
        self.p = int(patch_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        B, T, D = x.shape
        p = self.p
        pad = (p - (T % p)) % p
        if pad > 0:
            x = torch.cat([x, x.new_zeros((B, pad, D))], dim=1)
            mask = torch.cat([mask, torch.zeros((B, pad), device=mask.device, dtype=torch.bool)], dim=1)
        T2 = x.shape[1]
        Tp = T2 // p
        xp = x.view(B, Tp, p * D)
        mp = mask.view(B, Tp, p).any(dim=-1)
        return xp, mp, pad

    def unpatchify(self, xp_out: torch.Tensor, T_orig: int, pad: int, D: int) -> torch.Tensor:
        B, Tp, _ = xp_out.shape
        p = self.p
        x = xp_out.view(B, Tp * p, D)
        if pad > 0:
            x = x[:, : T_orig, :]
        return x


# =============================================================================
# ------------- Multi-Scale Deformable Attention (1D) + adaLN-Zero ------------
# =============================================================================

def downsample_1d(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [B,T,C], mask: [B,T]
    -> x2: [B,ceil(T/2),C], mask2: [B,ceil(T/2)]
    Uses masked average pooling with stride 2.
    """
    B, T, C = x.shape
    pad = T % 2
    if pad:
        x = torch.cat([x, x.new_zeros((B, 1, C))], dim=1)
        mask = torch.cat([mask, torch.zeros((B, 1), device=mask.device, dtype=torch.bool)], dim=1)
        T = T + 1

    x0 = x[:, 0::2, :]
    x1 = x[:, 1::2, :]
    m0 = mask[:, 0::2].unsqueeze(-1)
    m1 = mask[:, 1::2].unsqueeze(-1)
    denom = (m0.float() + m1.float()).clamp(min=1.0)
    x2 = (x0 * m0.float() + x1 * m1.float()) / denom
    mask2 = (m0.squeeze(-1) | m1.squeeze(-1))
    return x2, mask2


class AdaLNZero(nn.Module):
    """
    adaLN-Zero modulation like DiT:
      shift, scale, gate for attn and mlp
    Zero-initialized last layer => starts as identity (gates ~ 0).
    """
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        # zero-init last linear
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # returns shift1, scale1, gate1, shift2, scale2, gate2 each [B,C]
        return self.net(c).chunk(6, dim=-1)


class MSDeformAttn1D(nn.Module):
    """
    Multi-Scale Deformable Attention (1D).
    - Builds L scales via pooling.
    - For each query token, predicts K sampling offsets per scale per head.
    - Samples K keys/values from each scale by linear interpolation.
    - Computes dot-product attention over S = L*K sampled points (per head).
    - Uses RoPE on q and sampled k based on (possibly fractional) positions.

    Complexity: O(B * T * H * L * K * dh).
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_scales: int = 3,
        num_points: int = 4,
        max_offset_frac: float = 0.25,   # as fraction of scale length
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.h = num_heads
        self.dh = dim // num_heads

        self.L = int(num_scales)
        self.K = int(num_points)
        self.max_offset_frac = float(max_offset_frac)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # offsets per token: [B,T, H, L, K] (normalized, later scaled by length)
        self.off_proj = nn.Linear(dim, num_heads * self.L * self.K)

        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _sample_1d_linear(
        self,
        kv: torch.Tensor,        # [B,Ts,H,dh]
        m: torch.Tensor,         # [B,Ts] bool
        idx: torch.Tensor,       # [B,Tq,H,L,K] float indices in [0,Ts-1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples kv at idx via linear interpolation.
        Returns:
          samp: [B,Tq,H,L,K,dh]
          pos:  [B,Tq,H,L,K] float (for RoPE)
          valid:[B,Tq,H,L,K] bool
        """
        B, Ts, H, dh = kv.shape
        _, Tq, _, L, K = idx.shape

        idx = idx.clamp(0.0, float(Ts - 1) - 1e-6)
        i0 = idx.floor().to(torch.long)
        i1 = (i0 + 1).clamp(max=Ts - 1)
        w1 = (idx - i0.float()).to(kv.dtype)
        w0 = (1.0 - w1).to(kv.dtype)

        # gather masks
        i0_flat = i0.reshape(B, -1)
        i1_flat = i1.reshape(B, -1)
        m0 = m.gather(1, i0_flat).reshape(B, Tq, H, L, K)
        m1 = m.gather(1, i1_flat).reshape(B, Tq, H, L, K)
        valid = m0 | m1

        # gather kv: expand indices to [B, Tq,H,L,K,dh] over dim=1 (Ts)
        # kv: [B,Ts,H,dh] -> [B,H,Ts,dh]
        kvh = kv.permute(0, 2, 1, 3).contiguous()  # [B,H,Ts,dh]
        # indices should be [B,H,Tq,L,K,dh] on dim=2
        i0g = i0.permute(0, 2, 1, 3, 4).unsqueeze(-1).expand(B, H, Tq, L, K, dh)
        i1g = i1.permute(0, 2, 1, 3, 4).unsqueeze(-1).expand(B, H, Tq, L, K, dh)

        kvh_exp = kvh.unsqueeze(3).unsqueeze(3)  # [B,H,Ts,1,1,dh]
        kvh_exp = kvh_exp.expand(B, H, Ts, L, K, dh)

        v0 = torch.gather(kvh_exp, 2, i0g)
        v1 = torch.gather(kvh_exp, 2, i1g)

        w0h = w0.permute(0, 2, 1, 3, 4).unsqueeze(-1)  # [B,H,Tq,L,K,1]
        w1h = w1.permute(0, 2, 1, 3, 4).unsqueeze(-1)

        # interpolate
        samp = (w0h * v0 + w1h * v1)  # [B,H,Tq,L,K,dh]
        samp = samp.permute(0, 2, 1, 3, 4, 5).contiguous()  # [B,Tq,H,L,K,dh]

        # for RoPE positions we use idx itself
        pos = idx
        return samp, pos, valid

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C], mask: [B,T]
        returns [B,T,C]
        """
        B, T, C = x.shape
        H, dh = self.h, self.dh
        L, K = self.L, self.K

        # build multi-scale pyramids
        feats = [x]
        masks = [mask]
        for _ in range(1, L):
            x2, m2 = downsample_1d(feats[-1], masks[-1])
            feats.append(x2)
            masks.append(m2)

        # projections
        q = self.q_proj(x).view(B, T, H, dh)
        # RoPE on q using base positions (0..T-1)
        pos_q = torch.arange(T, device=x.device, dtype=torch.float32).view(1, T, 1).expand(B, T, H)
        q = apply_rope(q.reshape(B * T * H, dh), pos_q.reshape(B * T * H)).view(B, T, H, dh)

        # offsets predicted from x: [B,T, H*L*K] -> [B,T,H,L,K]
        off = self.off_proj(x).view(B, T, H, L, K)
        off = torch.tanh(off) * self.max_offset_frac  # fraction in [-max_offset_frac, max_offset_frac]

        # reference point in normalized [0,1] per token: i/(T-1)
        if T <= 1:
            ref = torch.zeros((B, T, H, 1, 1), device=x.device, dtype=torch.float32)
        else:
            ref = (torch.arange(T, device=x.device, dtype=torch.float32) / float(T - 1)).view(1, T, 1, 1, 1)
            ref = ref.expand(B, T, H, 1, 1)  # [B,T,H,1,1]

        # sample keys/values per scale
        k_samps = []
        v_samps = []
        valid_samps = []
        pos_samps = []

        for li in range(L):
            f = feats[li]     # [B,Ts,C]
            m = masks[li]     # [B,Ts]
            Ts = f.shape[1]

            k = self.k_proj(f).view(B, Ts, H, dh)
            v = self.v_proj(f).view(B, Ts, H, dh)

            # build sample indices in this scale:
            # normalized sample positions s = ref + off (off is fraction), clamp [0,1]
            s_norm = (ref + off[:, :, :, li:li+1, :].unsqueeze(3).squeeze(3)).clamp(0.0, 1.0)  # [B,T,H,1,K]
            s_norm = s_norm.squeeze(3)  # [B,T,H,K]

            # convert to float index in [0,Ts-1]
            if Ts <= 1:
                idx = torch.zeros((B, T, H, 1, K), device=x.device, dtype=torch.float32)
            else:
                idx = s_norm * float(Ts - 1)
                idx = idx.unsqueeze(3)  # [B,T,H,1,K] treat as L=1 within sampler

            # sampler expects [B,T,H,L,K] => use L=1 for this scale
            idx_full = idx  # [B,T,H,1,K]
            idx_full = idx_full.expand(B, T, H, 1, K)
            # reshape to [B,T,H,L,K] with L=1
            k_s, pos_s, val_s = self._sample_1d_linear(k, m, idx_full)
            v_s, _, _ = self._sample_1d_linear(v, m, idx_full)

            # apply RoPE to sampled keys using fractional positions "pos_s"
            # pos_s: [B,T,H,1,K], rotate k_s on dh
            k_s2 = apply_rope(k_s.reshape(B * T * H * 1 * K, dh), pos_s.reshape(B * T * H * 1 * K)).view(B, T, H, 1, K, dh)
            k_s = k_s2

            k_samps.append(k_s)     # [B,T,H,1,K,dh]
            v_samps.append(v_s)     # [B,T,H,1,K,dh]
            valid_samps.append(val_s.unsqueeze(3))  # [B,T,H,1,K]
            pos_samps.append(pos_s)  # unused beyond RoPE

        # concat across scales -> [B,T,H,L,K,dh]
        k_all = torch.cat(k_samps, dim=3)
        v_all = torch.cat(v_samps, dim=3)
        valid_all = torch.cat(valid_samps, dim=3)  # [B,T,H,L,K]

        # dot-product attention over S=L*K samples
        # q: [B,T,H,dh] -> [B,T,H,1,1,dh]
        qh = q.unsqueeze(3).unsqueeze(3)
        logits = (qh * k_all).sum(dim=-1) * (dh ** -0.5)  # [B,T,H,L,K]

        # mask invalid samples + invalid queries
        logits = logits.masked_fill(~valid_all, -1e9)
        logits = logits.masked_fill(~mask.unsqueeze(2).unsqueeze(3).unsqueeze(4), -1e9)

        attn = torch.softmax(logits.view(B, T, H, L * K), dim=-1).view(B, T, H, L, K)
        attn = self.attn_drop(attn)

        out = (attn.unsqueeze(-1) * v_all).sum(dim=4).sum(dim=3)  # sum over K then L -> [B,T,H,dh]
        out = out.reshape(B, T, C)
        out = self.proj_drop(self.out(out))
        return out * mask.unsqueeze(-1)


class DiTMSDeformBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        num_scales: int = 3,
        num_points: int = 4,
        max_offset_frac: float = 0.25,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MSDeformAttn1D(
            dim=dim,
            num_heads=num_heads,
            num_scales=num_scales,
            num_points=num_points,
            max_offset_frac=max_offset_frac,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.mod = AdaLNZero(cond_dim=cond_dim, hidden_dim=dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.mod(c)

        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        a = self.attn(h, mask=mask)
        x = x + torch.tanh(gate1.unsqueeze(1)) * a
        x = x * mask.unsqueeze(-1)

        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        m = self.mlp(h)
        x = x + torch.tanh(gate2.unsqueeze(1)) * m
        return x * mask.unsqueeze(-1)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DiTMSDeform1D(nn.Module):
    """
    DiT-XL/2-like backbone on 1D latent sequences:
      - patchify (P=2)
      - embed to model_dim
      - blocks: adaLN-Zero + MSDeformAttn + MLP
      - output: velocity in original token space via unpatchify
    """
    def __init__(
        self,
        in_dim: int,                # latent channel dim (RVQ-VAE d_model)
        model_dim: int = 1152,      # DiT-XL width
        depth: int = 28,            # DiT-XL depth
        heads: int = 16,            # DiT-XL heads
        patch_size: int = 2,        # DiT-XL/2 patch size
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_scales: int = 3,
        num_points: int = 4,
        max_offset_frac: float = 0.25,
        cond_vocab_size: int = 1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.model_dim = int(model_dim)
        self.patch = Patchify1D(patch_size=patch_size)
        self.p = int(patch_size)

        self.patch_in = nn.Linear(self.in_dim * self.p, model_dim)
        self.time_mlp = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))

        self.style_emb = nn.Embedding(int(cond_vocab_size), model_dim)

        self.blocks = nn.ModuleList([
            DiTMSDeformBlock(
                dim=model_dim,
                num_heads=heads,
                cond_dim=model_dim,
                num_scales=num_scales,
                num_points=num_points,
                max_offset_frac=max_offset_frac,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.out_norm = nn.LayerNorm(model_dim)

        # output head to patch space (zero-init like DiT)
        self.out_proj = nn.Linear(model_dim, self.in_dim * self.p)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor, cond_id: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,in_dim] current noised latent tokens x_t
        mask: [B,T] bool
        t: [B] float in [0,1]
        cond_id: [B] long
        returns v_pred: [B,T,in_dim]
        """
        B, T, D = x.shape
        xp, mp, pad = self.patch(x, mask)  # [B,Tp,D*p], [B,Tp]
        h = self.patch_in(xp) * mp.unsqueeze(-1)

        c = self.time_mlp(timestep_embedding(t, self.model_dim)) + self.style_emb(cond_id)

        for blk in self.blocks:
            h = blk(h, mask=mp, c=c)

        h = self.out_norm(h) * mp.unsqueeze(-1)
        vp = self.out_proj(h) * mp.unsqueeze(-1)  # [B,Tp,D*p]
        v = self.patch.unpatchify(vp, T_orig=T, pad=pad, D=D)
        return v * mask.unsqueeze(-1)


# =============================================================================
# ------------------------ Lightning: Rectified Flow --------------------------
# =============================================================================

class LatentFlowMSDeformDiT(L.LightningModule):
    """
    Velocity matching / rectified flow on RVQ-VAE latent tokens.
    """
    def __init__(
        self,
        vqvae_ckpt: str,
        cond_vocab_size: int,
        # DiT-XL/2 defaults
        model_preset: str = "xl",
        model_dim: int = 1152,
        depth: int = 28,
        heads: int = 16,
        patch_size: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_scales: int = 3,
        num_points: int = 4,
        max_offset_frac: float = 0.25,
        # optim
        lr: float = 2e-4,
        weight_decay: float = 1e-2,
        ema_decay: float = 0.999,   # optional EMA for sampling
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load + freeze RVQ-VAE
        self.vqvae = VectorGraphRVQVAE.load_from_checkpoint(vqvae_ckpt, map_location="cpu")
        self.vqvae.eval()
        for p in self.vqvae.parameters():
            p.requires_grad = False

        in_dim = int(self.vqvae.encoder.out_norm.normalized_shape[0])  # d_model

        # presets (rough DiT sizes)
        if model_preset.lower() == "xl":
            model_dim = 1152
            depth = 28
            heads = 16
        elif model_preset.lower() == "l":
            model_dim = 1024
            depth = 24
            heads = 16
        elif model_preset.lower() == "b":
            model_dim = 768
            depth = 12
            heads = 12
        elif model_preset.lower() == "s":
            model_dim = 384
            depth = 8
            heads = 6

        self.net = DiTMSDeform1D(
            in_dim=in_dim,
            model_dim=model_dim,
            depth=depth,
            heads=heads,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_scales=num_scales,
            num_points=num_points,
            max_offset_frac=max_offset_frac,
            cond_vocab_size=cond_vocab_size,
        )

        # EMA copy for sampling stability
        self.use_ema = ema_decay > 0
        self.ema_decay = float(ema_decay)
        self.net_ema = DiTMSDeform1D(
            in_dim=in_dim,
            model_dim=model_dim,
            depth=depth,
            heads=heads,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_scales=num_scales,
            num_points=num_points,
            max_offset_frac=max_offset_frac,
            cond_vocab_size=cond_vocab_size,
        )
        self.net_ema.load_state_dict(self.net.state_dict())
        for p in self.net_ema.parameters():
            p.requires_grad = False
        self.net_ema.eval()

    @torch.no_grad()
    def _update_ema(self):
        if not self.use_ema:
            return
        d = self.ema_decay
        msd = self.net.state_dict()
        esd = self.net_ema.state_dict()
        for k in esd.keys():
            esd[k].mul_(d).add_(msd[k], alpha=1.0 - d)

    @torch.no_grad()
    def encode_latents(self, batch: Dict[str, Any]) -> torch.Tensor:
        x_pad = batch["x_pad"].to(self.device)          # [B,T,2]
        mask = batch["mask"].to(self.device)            # [B,T]
        neighbors = batch["neighbors"].to(self.device)  # [B,T,M]
        hq = self.vqvae.encode_latents(x_pad, mask, neighbors)  # [B,T,d_model]
        return hq

    def forward(self, xt: torch.Tensor, mask: torch.Tensor, t: torch.Tensor, cond_id: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        net = self.net_ema if use_ema else self.net
        return net(xt, mask, t, cond_id)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        mask = batch["mask"].to(self.device)          # [B,T]
        cond_id = batch["cond_id"].to(self.device)    # [B]

        x1 = self.encode_latents(batch)               # [B,T,D]
        x1 = x1 * mask.unsqueeze(-1)

        # Rectified Flow / velocity matching
        x0 = torch.randn_like(x1)
        B = x1.size(0)
        t = torch.rand((B,), device=self.device, dtype=torch.float32)

        tb = t.view(B, 1, 1)
        xt = (1.0 - tb) * x0 + tb * x1
        v_target = (x1 - x0)

        v_pred = self.forward(xt, mask, t, cond_id, use_ema=False)

        mse = (v_pred - v_target).pow(2).mean(dim=-1)  # [B,T]
        loss = (mse * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/Tavg", mask.float().sum(dim=1).mean(), prog_bar=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        self._update_ema()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        mask = batch["mask"].to(self.device)
        cond_id = batch["cond_id"].to(self.device)
        x1 = self.encode_latents(batch) * mask.unsqueeze(-1)

        x0 = torch.randn_like(x1)
        B = x1.size(0)
        t = torch.rand((B,), device=self.device, dtype=torch.float32)
        tb = t.view(B, 1, 1)
        xt = (1.0 - tb) * x0 + tb * x1
        v_target = (x1 - x0)

        v_pred = self.forward(xt, mask, t, cond_id, use_ema=False)
        mse = (v_pred - v_target).pow(2).mean(dim=-1)
        loss = (mse * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay), betas=(0.9, 0.99))

    # --------------------- Sampling (optional) ---------------------

    @torch.no_grad()
    def sample_latents(
        self,
        T: int,
        cond_id: int = 0,
        steps: int = 60,
        method: str = "heun",
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Integrate dz/dt = v_theta(z,t) from t=0..1 starting z(0) ~ N(0,I).
        Returns x1_hat latents: [1,T,D]
        """
        D = self.net.in_dim
        device = self.device

        z = torch.randn((1, T, D), device=device)
        mask = torch.ones((1, T), device=device, dtype=torch.bool)
        cond = torch.tensor([cond_id], device=device, dtype=torch.long)

        dt = 1.0 / float(steps)
        for i in range(steps):
            t0 = torch.tensor([i * dt], device=device, dtype=torch.float32)
            v0 = self.forward(z, mask, t0, cond, use_ema=use_ema)
            if method == "euler":
                z = z + dt * v0
            else:
                z_pred = z + dt * v0
                t1 = torch.tensor([(i + 1) * dt], device=device, dtype=torch.float32)
                v1 = self.forward(z_pred, mask, t1, cond, use_ema=use_ema)
                z = z + dt * 0.5 * (v0 + v1)
        return z

    @torch.no_grad()
    def decode_sample_to_graph(
        self,
        hq: torch.Tensor,
        mem_mask: torch.Tensor,
        exist_thr: float = 0.5,
        edge_thr: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Uses the RVQ-VAE decoder + edge head to produce coords+edges from latent memory hq.
        Returns a dict with predicted coords [Q,2], exist prob [Q], adjacency [Q,Q] bool.
        """
        coords_pred, exist_logits, edge_logits = self.vqvae.decode_graph(hq, mem_mask)
        exist_prob = torch.sigmoid(exist_logits)[0]  # [Q]
        coords = coords_pred[0]                      # [Q,2]
        exist = exist_prob > exist_thr
        edge_prob = torch.sigmoid(edge_logits[0])
        edges = (edge_prob > edge_thr) & exist.unsqueeze(0) & exist.unsqueeze(1)
        edges.fill_diagonal_(False)
        return {"coords": coords, "exist_prob": exist_prob, "exist": exist, "adj": edges}
