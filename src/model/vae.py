#!/usr/bin/env python3
"""
Geometry-aware Vector Graph RVQ-VAE (Graph-Transformer encoder + Residual VQ + DETR-style query decoder)
for your *vectorized street graphs* dataset.

This implements the geometry-aware improvements you described:

✅ Fourier features for continuous (x,y) inputs
✅ Graph-Transformer encoder with *graph-structured* sparse attention over neighbors
✅ Geometry-biased attention logits using relative dx, dy, distance, sin/cos(angle)
✅ Residual Vector Quantization (RVQ): multiple codebooks quantize residuals for high-fidelity coords
✅ Query-based decoder (DETR-style):
    - fixed Q learnable queries
    - cross-attention reads encoded latents
    - outputs:
        * coordinate head: (x,y)
        * existence head: node present probability
        * edge head: predicts adjacency with geometry-aware pair features
✅ Proper permutation handling via Hungarian matching (scipy if available; greedy fallback)
✅ Edge loss computed in query-space after matching (masked, class-imbalance aware)

You can later plug a diffusion/flow model on the *quantized latent embeddings* (continuous) or on indices.

Install:
  pip install torch lightning numpy tqdm matplotlib
Optional (better matching):
  pip install scipy

Run (train):
  python rvq_graph_vqvae.py --data city_street_graphs.pkl --max-epochs 200 --q 512 --max-neighbors 16

Notes:
- Q is the max number of nodes the decoder can represent. For city graphs, 256-1024 are common.
- The edge head here is dense (QxQ) because it gives best quality; if Q is huge, add sparse edges later.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.dataset.dataset import QuickDrawGraphDataModule

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import grad_norm
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path

from scipy.optimize import linear_sum_assignment  # type: ignore

def edge_index_to_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    edge_index: [2,E]
    returns adjacency matrix: [N,N]
    """
    A = torch.zeros((num_nodes, num_nodes), device=edge_index.device, dtype=torch.float32)
    A[edge_index[0], edge_index[1]] = 1.0
    return A

# =============================================================================
# Fourier Features (geometry-aware input lift)
# =============================================================================

class FourierEmbedder(nn.Module):
    """
    x in [-1,1] (your normalized coords), returns Fourier features.
    """
    def __init__(self, num_freqs: int = 10, max_freq_log2: int = 9):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0.0, float(max_freq_log2), steps=num_freqs)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,2]
        # -> [B,T,2,num_freqs]
        x_proj = x.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)
        # sin/cos -> [B,T,2,2*num_freqs]
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        # flatten -> [B,T,4*num_freqs]
        return emb.flatten(-2)


# =============================================================================
# Geometry-aware Graph-Transformer layer (sparse neighbor attention)
# =============================================================================

class NeighborGraphAttention(nn.Module):
    """
    Sparse multi-head attention over a fixed neighbor list per token:
      neighbors: [B,T,M], -1 padded
    Geometry bias is computed from dx,dy,dist,sin/cos(angle) between (i -> neighbor).
    """
    def __init__(self, d_model: int, n_heads: int, geo_hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.scale = self.dh ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        # geometry -> per-head bias
        self.geo_mlp = nn.Sequential(
            nn.Linear(6, geo_hidden),
            nn.SiLU(),
            nn.Linear(geo_hidden, geo_hidden),
            nn.SiLU(),
            nn.Linear(geo_hidden, n_heads),
        )

    def forward(
        self,
        h: torch.Tensor,          # [B,T,C]
        coords: torch.Tensor,     # [B,T,2]
        mask: torch.Tensor,       # [B,T] bool
        neighbors: torch.Tensor,  # [B,T,M] long, -1 padded
    ) -> torch.Tensor:
        B, T, C = h.shape
        M = neighbors.shape[-1]
        device = h.device

        # gather neighbor features
        nbr = neighbors.clamp(min=0)  # -1 -> 0 (masked later)
                
        idx = nbr + (torch.arange(B, device=device) * T).view(B, 1, 1)
        flat_h = h.view(B * T, C)
        nbr_feat = flat_h[idx.view(-1)].view(B, T, M, C)
        
        flat_coords = coords.view(B * T, 2)
        nbr_coords = flat_coords[idx.view(-1)].view(B, T, M, 2)

        # validity: query must be real + neighbor index valid + neighbor token real
        flat_mask = mask.view(B * T)
        nbr_mask = flat_mask[idx.view(-1)].view(B, T, M)
        nbr_valid = (neighbors >= 0) & mask.unsqueeze(-1) & nbr_mask   # [B,T,M]

        # Q,K,V
        q = self.q_proj(h).view(B, T, self.n_heads, self.dh)                      # [B,T,H,dh]
        k = self.k_proj(nbr_feat).view(B, T, M, self.n_heads, self.dh)            # [B,T,M,H,dh]
        v = self.v_proj(nbr_feat).view(B, T, M, self.n_heads, self.dh)

        # geometry features between query i and neighbor j
        dx = nbr_coords[..., 0] - coords[..., 0].unsqueeze(-1)                    # [B,T,M]
        dy = nbr_coords[..., 1] - coords[..., 1].unsqueeze(-1)
        dist2 = dx * dx + dy * dy
        dist = torch.sqrt(dist2 + 1e-8)
        
        # Safe atan2 to avoid NaN gradients at (0,0)
        mask_zero = (dx.abs() < 1e-6) & (dy.abs() < 1e-6)
        dx_safe = torch.where(mask_zero, torch.ones_like(dx) * 1e-6, dx)
        dy_safe = torch.where(mask_zero, torch.ones_like(dy) * 1e-6, dy)
        ang = torch.atan2(dy_safe, dx_safe)
        
        geo = torch.stack([dx, dy, dist, dist2, torch.sin(ang), torch.cos(ang)], dim=-1)  # [B,T,M,6]

        bias = self.geo_mlp(geo)                                                  # [B,T,M,H]
        bias = bias.permute(0, 1, 3, 2).contiguous()                               # [B,T,H,M]

        # attention logits: [B,T,H,M]
        # (q · k) across dh
        qh = q.unsqueeze(3)                                                        # [B,T,H,1,dh]
        kh = k.permute(0, 1, 3, 2, 4).contiguous()                                 # [B,T,H,M,dh]
        logits = (qh * kh).sum(-1) * self.scale                                    # [B,T,H,M]
        logits = logits + bias

        # mask invalid neighbors
        logits = logits.masked_fill(~nbr_valid.unsqueeze(2), -1e9)
        # also mask invalid queries
        logits = logits.masked_fill(~mask.unsqueeze(2).unsqueeze(-1), -1e9)

        attn = torch.softmax(logits, dim=-1)                                       # [B,T,H,M]
        # attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)

        vh = v.permute(0, 1, 3, 2, 4).contiguous()                                 # [B,T,H,M,dh]
        out = (attn.unsqueeze(-1) * vh).sum(dim=-2)                                # [B,T,H,dh]
        out = out.reshape(B, T, C)
        out = self.drop(self.out(out))
        out = out * mask.unsqueeze(-1)
        return out


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = NeighborGraphAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.norm1(h), coords=coords, mask=mask, neighbors=neighbors)
        h = h * mask.unsqueeze(-1)
        h = h + self.mlp(self.norm2(h))
        h = h * mask.unsqueeze(-1)
        return h


class VectorGraphEncoder(nn.Module):
    """
    Fourier features + degree feature + graph-structured sparse attention blocks.
    Produces node embeddings ready for RVQ.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        depth: int = 8,
        fourier_freqs: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ff = FourierEmbedder(num_freqs=fourier_freqs, max_freq_log2=fourier_freqs - 1)
        in_dim = 4 * fourier_freqs + 1  # fourier + degree
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList([GraphTransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d_model)

    @staticmethod
    def degree_feature(neighbors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # neighbors [B,T,M], -1 padded; mask [B,T]
        deg = (neighbors >= 0).sum(dim=-1).float()  # [B,T] includes self
        deg = torch.log1p(deg).unsqueeze(-1)        # [B,T,1]
        return deg * mask.unsqueeze(-1).float()

    def forward(self, x_pad: torch.Tensor, mask: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        # x_pad: [B,T,2] coords (normalized); mask [B,T]; neighbors [B,T,M]
        ff = self.ff(x_pad)  # [B,T,4F]
        deg = self.degree_feature(neighbors, mask)  # [B,T,1]
        h = self.in_proj(torch.cat([ff, deg], dim=-1))
        h = h * mask.unsqueeze(-1)
        for blk in self.blocks:
            h = blk(h, coords=x_pad, mask=mask, neighbors=neighbors)
        return self.out_norm(h) * mask.unsqueeze(-1)


# =============================================================================
# Residual Vector Quantization (RVQ)
# =============================================================================

class ResidualVectorQuantizer(nn.Module):
    """
    Residual VQ:
      residual_0 = x
      q_i = nearest(codebook_i, residual_{i})
      residual_{i+1} = residual_i - q_i
      output = sum_i q_i
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int = 1024,
        num_codebooks: int = 4,
        commitment_weight: float = 0.25,
        ema: bool = False,  # left False for simplicity (STE + standard codebook updates)
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.commitment_weight = commitment_weight
        self.ema = ema

        self.codebooks = nn.Parameter(torch.randn(num_codebooks, codebook_size, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B,T,C], mask [B,T]
        returns:
          x_q: [B,T,C]
          indices: [B,T,num_codebooks] (long; -1 for padding)
          vq_loss: scalar
        """
        B, T, C = x.shape
        device = x.device
        x_q = torch.zeros_like(x)
        residual = x

        indices = torch.full((B, T, self.num_codebooks), -1, device=device, dtype=torch.long)

        vq_loss = torch.tensor(0.0, device=device)
        valid = mask.view(B * T)

        x_flat = residual.view(B * T, C)
        xq_flat = x_q.view(B * T, C)

        for i in range(self.num_codebooks):
            cb = self.codebooks[i]  # [K,C]
            # distances: ||a-b||^2 = a^2 + b^2 - 2ab
            # only compute for valid entries
            xf = x_flat[valid]  # [Nv,C]
            if xf.numel() == 0:
                break

            a2 = (xf * xf).sum(dim=-1, keepdim=True)         # [Nv,1]
            b2 = (cb * cb).sum(dim=-1).unsqueeze(0)          # [1,K]
            ab = xf @ cb.t()                                  # [Nv,K]
            dist = a2 + b2 - 2.0 * ab                         # [Nv,K]
            idx = torch.argmin(dist, dim=-1)                  # [Nv]

            # quantize
            q = cb[idx]                                       # [Nv,C]

            # losses
            # codebook loss: move cb toward xf; commitment: move xf toward q
            # (standard VQ-VAE formulation)
            vq_loss = vq_loss + F.mse_loss(q, xf.detach())
            vq_loss = vq_loss + self.commitment_weight * F.mse_loss(xf, q.detach())

            # straight-through
            q_st = xf + (q - xf).detach()

            # write back
            xq_flat_valid = xq_flat[valid] + q_st
            xq_flat[valid] = xq_flat_valid

            # store indices in [B,T,i]
            indices.view(B * T, self.num_codebooks)[valid, i] = idx

            # update residual for next stage
            x_flat_valid = (x_flat[valid] - q_st)
            x_flat[valid] = x_flat_valid

        x_q = xq_flat.view(B, T, C) * mask.unsqueeze(-1)
        return x_q, indices, vq_loss


# =============================================================================
# DETR-style Query Decoder + geometry-aware edge head
# =============================================================================

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class QueryDecoder(nn.Module):
    """
    DETR-like: learnable queries cross-attend to encoded tokens.
    Outputs:
      coords_pred: [B,Q,2]
      exist_logits: [B,Q]
      q_embed: [B,Q,D]
    """
    def __init__(self, d_model: int, q: int = 512, n_heads: int = 8, depth: int = 6, dropout: float = 0.0):
        super().__init__()
        self.q = q
        self.query = nn.Parameter(torch.randn(q, d_model) / math.sqrt(d_model))

        self.cross = nn.ModuleList([
            nn.ModuleDict({
                "norm_q": nn.LayerNorm(d_model),
                "norm_kv": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "mlp": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(depth)
        ])

        self.exist_head = nn.Linear(d_model, 1)
        self.coord_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, memory: torch.Tensor, mem_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        memory: [B,T,D]
        mem_mask: [B,T] bool
        """
        B, T, D = memory.shape
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B,Q,D]

        key_padding_mask = ~mem_mask  # True means ignore
        for blk in self.cross:
            qq = blk["norm_q"](q)
            mm = blk["norm_kv"](memory)
            attn_out, _ = blk["attn"](qq, mm, mm, key_padding_mask=key_padding_mask, need_weights=False)
            q = q + attn_out
            q = q + blk["mlp"](q)

        exist_logits = self.exist_head(q).squeeze(-1)  # [B,Q]
        coords_pred = torch.tanh(self.coord_head(q))   # [B,Q,2] in [-1,1]
        return coords_pred, exist_logits, q


class DenseGeometryEdgeHead(nn.Module):
    """
    Dense adjacency logits [B,Q,Q] with geometry-aware features.
    Uses:
      - bilinear query interaction
      - plus MLP over (dx,dy,dist,dist^2,sin,cos)
    """
    def __init__(self, d_model: int, geo_hidden: int = 64):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.geo = nn.Sequential(
            nn.Linear(6, geo_hidden),
            nn.SiLU(),
            nn.Linear(geo_hidden, geo_hidden),
            nn.SiLU(),
            nn.Linear(geo_hidden, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, q: torch.Tensor, coords: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        """
        q: [B,Q,D]
        coords: [B,Q,2]
        active_mask: [B,Q] bool -> which queries are "in play" for edges
        returns logits: [B,Q,Q]
        """
        B, Qn, D = q.shape

        qW = q @ self.W               # [B,Q,D]
        bil = qW @ q.transpose(1, 2)  # [B,Q,Q]

        dx = coords[:, :, 0].unsqueeze(2) - coords[:, :, 0].unsqueeze(1)
        dy = coords[:, :, 1].unsqueeze(2) - coords[:, :, 1].unsqueeze(1)
        dist2 = dx * dx + dy * dy
        dist = torch.sqrt(dist2 + 1e-8)
        
        # Safe atan2 to avoid NaN gradients at (0,0)
        mask_zero = (dx.abs() < 1e-6) & (dy.abs() < 1e-6)
        dx_safe = torch.where(mask_zero, torch.ones_like(dx) * 1e-6, dx)
        dy_safe = torch.where(mask_zero, torch.ones_like(dy) * 1e-6, dy)
        ang = torch.atan2(dy_safe, dx_safe)
        
        geo = torch.stack([dx, dy, dist, dist2, torch.sin(ang), torch.cos(ang)], dim=-1)  # [B,Q,Q,6]
        geo_logits = self.geo(geo).squeeze(-1)

        logits = bil + geo_logits + self.bias
        logits = logits - torch.diag_embed(torch.diagonal(logits, dim1=1, dim2=2))

        mm = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
        logits = logits.masked_fill(~mm, -1e9)
        return logits

# =============================================================================
# Matching + losses (permutation invariance)
# =============================================================================

def _hungarian(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    cost: [Q, N] on CPU device ok; uses scipy if available; greedy fallback.
    """
    device = cost.device
    Q, N = cost.shape
    c = cost.detach().cpu().numpy()
    r, c2 = linear_sum_assignment(c)
    return torch.tensor(r, device=device, dtype=torch.long), torch.tensor(c2, device=device, dtype=torch.long)


@dataclass
class MatchResult:
    pred_to_gt: torch.Tensor   # [B,Q] gt index or -1
    pred_matched: torch.Tensor # [B,Q] bool


def match_nodes_by_coords(
    coords_pred: torch.Tensor,   # [B,Q,2]
    exist_logits: torch.Tensor,  # [B,Q]
    coords_gt: List[torch.Tensor],   # list length B, each [N,2]
    exist_cost_weight: float = 0.1,
    coord_cost_weight: float = 1.0,
) -> MatchResult:
    B, Q, _ = coords_pred.shape
    device = coords_pred.device

    pred_to_gt = torch.full((B, Q), -1, device=device, dtype=torch.long)
    pred_matched = torch.zeros((B, Q), device=device, dtype=torch.bool)

    exist_prob = torch.sigmoid(exist_logits)

    for b in range(B):
        gt = coords_gt[b].to(device)
        N = gt.size(0)
        if N == 0:
            continue

        cp = coords_pred[b].unsqueeze(1)         # [Q,1,2]
        cg = gt.unsqueeze(0)                     # [1,N,2]
        coord_cost = (cp - cg).abs().sum(dim=-1) # [Q,N]

        exist_cost = (1.0 - exist_prob[b]).unsqueeze(1).expand(Q, N)
        cost = coord_cost_weight * coord_cost + exist_cost_weight * exist_cost

        r, c = _hungarian(cost)
        pred_to_gt[b, r] = c
        pred_matched[b, r] = True

    return MatchResult(pred_to_gt=pred_to_gt, pred_matched=pred_matched)


def lift_adj_to_queries(A_gt: torch.Tensor, pred_to_gt: torch.Tensor) -> torch.Tensor:
    """
    A_gt: [N,N]
    pred_to_gt: [Q] in [0,N-1] or -1
    returns A_q: [Q,Q] masked for unmatched.
    """
    device = A_gt.device
    Q = pred_to_gt.numel()
    A_q = torch.zeros((Q, Q), device=device, dtype=A_gt.dtype)

    valid = pred_to_gt >= 0
    if valid.sum() == 0:
        return A_q

    gi = pred_to_gt.clamp(min=0)
    Aj = A_gt.index_select(0, gi).index_select(1, gi)  # [Q,Q]
    vv = valid.unsqueeze(0) & valid.unsqueeze(1)
    A_q = Aj * vv.float()
    A_q.fill_diagonal_(0.0)
    return A_q


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal loss for logits.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def dense_edge_bce_loss(
    logits: torch.Tensor,     # [B,Q,Q]
    A_target: torch.Tensor,   # [B,Q,Q]
    active_mask: torch.Tensor,# [B,Q] bool
    pos_weight: float = 5.0,
) -> torch.Tensor:
    B, Q, _ = logits.shape
    mm = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
    tri = torch.triu(torch.ones((Q, Q), device=logits.device, dtype=torch.bool), diagonal=1)
    valid = mm & tri.unsqueeze(0)

    x = logits[valid]
    y = A_target[valid]
    if x.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    pw = torch.tensor(pos_weight, device=logits.device)
    return F.binary_cross_entropy_with_logits(x, y, pos_weight=pw, reduction="mean")


# =============================================================================
# LightningModule: RVQ-VAE with query decoder + edge head
# =============================================================================

class VectorGraphRVQVAE(L.LightningModule):
    def __init__(
        self,
        d_model: int = 256,
        enc_heads: int = 8,
        enc_depth: int = 8,
        dec_heads: int = 8,
        dec_depth: int = 6,
        fourier_freqs: int = 10,
        dropout: float = 0.0,
        q: int = 512,
        codebook_size: int = 1024,
        num_codebooks: int = 4,
        commitment_weight: float = 0.25,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        # loss weights
        w_coord: float = 10.0,
        w_exist: float = 1.0,
        w_edge: float = 1.0,
        pos_weight_edge: float = 10.0,
        exist_alpha: float = 0.25,
        exist_gamma: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = VectorGraphEncoder(
            d_model=d_model,
            n_heads=enc_heads,
            depth=enc_depth,
            fourier_freqs=fourier_freqs,
            dropout=dropout,
        )
        self.rvq = ResidualVectorQuantizer(
            dim=d_model,
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            commitment_weight=commitment_weight,
        )
        self.decoder = QueryDecoder(d_model=d_model, q=q, n_heads=dec_heads, depth=dec_depth, dropout=dropout)
        self.edge_head = DenseGeometryEdgeHead(d_model=d_model)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.9, 0.99))
        return opt

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)
        
        # Check for NaNs in gradients
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradient of {name}")
                if torch.isinf(param.grad).any():
                    print(f"Inf detected in gradient of {name}")


    def forward(self, x_pad: torch.Tensor, mask: torch.Tensor, neighbors: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x_pad, mask=mask, neighbors=neighbors)         # [B,T,D]
        hq, indices, vq_loss = self.rvq(h, mask=mask)                   # [B,T,D], [B,T,Cb], scalar
        coords_pred, exist_logits, q_embed = self.decoder(hq, mem_mask=mask)  # [B,Q,2], [B,Q], [B,Q,D]
        return {
            "hq": hq,
            "indices": indices,
            "vq_loss": vq_loss,
            "coords_pred": coords_pred,
            "exist_logits": exist_logits,
            "q_embed": q_embed,
        }

    def log_reconstruction(self, batch: Dict[str, Any], batch_idx: int):
        if batch_idx != 0:
            return
        
        # Check if logger exists and has experiment (TensorBoard)
        if not self.logger or not hasattr(self.logger, "experiment"):
            return

        x_pad = batch["x_pad"].to(self.device)
        mask = batch["mask"].to(self.device)
        neighbors = batch["neighbors"].to(self.device)
        ei_list = batch["edge_index"]

        # Forward pass
        out = self.forward(x_pad, mask=mask, neighbors=neighbors)
        coords_pred = out["coords_pred"]   # [B,Q,2]
        exist_logits = out["exist_logits"] # [B,Q]
        q_embed = out["q_embed"]           # [B,Q,D]

        # Visualize first sample in batch
        idx = 0
        
        # 1. Ground Truth
        gt_coords = x_pad[idx]
        gt_mask = mask[idx]
        gt_edges = ei_list[idx] # [2, E]
        
        gt_coords_np = gt_coords[gt_mask].detach().float().cpu().numpy()
        gt_edges_np = gt_edges.detach().cpu().numpy()

        # 2. Prediction
        pred_coords = coords_pred[idx]
        pred_exist = torch.sigmoid(exist_logits[idx])
        
        # Filter by existence threshold
        exist_thr = 0.5
        keep_mask = pred_exist > exist_thr # [Q]
        
        # Compute edge probabilities for this sample
        # We treat 'keep_mask' as the active nodes
        active_mask = keep_mask.unsqueeze(0) # [1, Q]
        q_embed_1 = q_embed[idx].unsqueeze(0)
        coords_pred_1 = pred_coords.unsqueeze(0)
        
        # Edge head expects batch
        edge_logits = self.edge_head(q_embed_1, coords_pred_1, active_mask=active_mask) # [1, Q, Q]
        edge_probs = torch.sigmoid(edge_logits[0]) # [Q, Q]
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # GT Plot
        ax = axes[0]
        if len(gt_coords_np) > 0:
            ax.scatter(gt_coords_np[:, 0], gt_coords_np[:, 1], c='blue', s=10, label='Node')
            for k in range(gt_edges_np.shape[1]):
                u, v = gt_edges_np[:, k]
                # Sanity check indices
                if u < len(gt_coords_np) and v < len(gt_coords_np):
                    ax.plot([gt_coords_np[u, 0], gt_coords_np[v, 0]], 
                            [gt_coords_np[u, 1], gt_coords_np[v, 1]], 'b-', alpha=0.3, linewidth=0.5)
        ax.set_title("Ground Truth")
        ax.invert_yaxis() # SVG coords usually y-down
        ax.axis('equal')

        # Pred Plot
        ax = axes[1]
        pred_coords_np = pred_coords.detach().float().cpu().numpy()
        keep_idx = torch.where(keep_mask)[0].cpu().numpy()
        
        if len(keep_idx) > 0:
            kept_coords = pred_coords_np[keep_idx]
            ax.scatter(kept_coords[:, 0], kept_coords[:, 1], c='red', s=10, label='Pred')
            
            # Edges between kept nodes
            edge_thr = 0.5
            
            # Get upper triangular indices
            r, c = torch.triu_indices(len(pred_exist), len(pred_exist), offset=1, device=self.device)
            
            # Filter by probability
            p_vals = edge_probs[r, c]
            valid_edges = p_vals > edge_thr
            
            r_valid = r[valid_edges].cpu().numpy()
            c_valid = c[valid_edges].cpu().numpy()
            
            for u, v in zip(r_valid, c_valid):
                ax.plot([pred_coords_np[u, 0], pred_coords_np[v, 0]], 
                        [pred_coords_np[u, 1], pred_coords_np[v, 1]], 'r-', alpha=0.3, linewidth=0.5)

        ax.set_title(f"Reconstruction (Thr={exist_thr})")
        ax.invert_yaxis()
        ax.axis('equal')
        
        # Log to tensorboard
        self.logger.experiment.add_figure("reconstruction", fig, global_step=self.global_step)
        plt.close(fig)

    def _step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        x_pad = batch["x_pad"].to(self.device)         # [B,T,2]
        mask = batch["mask"].to(self.device)           # [B,T]
        neighbors = batch["neighbors"].to(self.device) # [B,T,M]
        ei_list: List[torch.Tensor] = batch["edge_index"]

        out = self.forward(x_pad, mask=mask, neighbors=neighbors)
        coords_pred = out["coords_pred"]        # [B,Q,2]
        exist_logits = out["exist_logits"]      # [B,Q]
        q_embed = out["q_embed"]                # [B,Q,D]

        # build per-graph GT coords list (unpadded)
        coords_gt_list: List[torch.Tensor] = []
        B = x_pad.size(0)
        for b in range(B):
            n = int(mask[b].sum().item())
            coords_gt_list.append(x_pad[b, :n].detach())  # coords are already normalized target

        # Hungarian matching: map predicted queries to GT nodes
        match = match_nodes_by_coords(coords_pred, exist_logits, coords_gt_list)

        # Existence targets: matched queries are 1 else 0
        tgt_exist = match.pred_matched.float()  # [B,Q]

        # Existence loss (focal)
        loss_exist = focal_bce_with_logits(
            exist_logits, tgt_exist,
            alpha=float(self.hparams.exist_alpha),
            gamma=float(self.hparams.exist_gamma),
            reduction="mean",
        )

        # Coordinate reconstruction loss on matched queries
        loss_coord = torch.tensor(0.0, device=self.device)
        denom = 0.0
        for b in range(B):
            pred_to_gt = match.pred_to_gt[b]  # [Q]
            valid_q = torch.where(pred_to_gt >= 0)[0]
            if valid_q.numel() == 0:
                continue
            gt_idx = pred_to_gt[valid_q]
            gt_coords = coords_gt_list[b].to(self.device)[gt_idx]
            pred_coords = coords_pred[b, valid_q]
            # Smooth L1 is robust for geometry
            loss_coord = loss_coord + F.smooth_l1_loss(pred_coords, gt_coords, reduction="sum")
            denom += float(valid_q.numel())
        if denom > 0:
            loss_coord = loss_coord / denom

        # Edge targets + edge prediction
        # We compute dense adjacency in query space (only matched queries participate)
        Q = coords_pred.size(1)
        A_target = torch.zeros((B, Q, Q), device=self.device, dtype=torch.float32)
        for b in range(B):
            n = coords_gt_list[b].size(0)
            A_gt = edge_index_to_adj(ei_list[b].to(self.device), num_nodes=n)  # [N,N]
            A_q = lift_adj_to_queries(A_gt, match.pred_to_gt[b])
            A_target[b] = A_q

        active_mask = match.pred_matched  # strict: only matched queries
        edge_logits = self.edge_head(q_embed, coords_pred, active_mask=active_mask)  # [B,Q,Q]
        loss_edge = dense_edge_bce_loss(
            edge_logits, A_target, active_mask=active_mask,
            pos_weight=float(self.hparams.pos_weight_edge),
        )

        loss = (
            out["vq_loss"]
            + float(self.hparams.w_exist) * loss_exist
            + float(self.hparams.w_coord) * loss_coord
            + float(self.hparams.w_edge) * loss_edge
        )

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/vq", out["vq_loss"], prog_bar=False)
        self.log(f"{stage}/exist", loss_exist, prog_bar=False)
        self.log(f"{stage}/coord", loss_coord, prog_bar=False)
        self.log(f"{stage}/edge", loss_edge, prog_bar=False)

        # a couple diagnostics
        with torch.no_grad():
            avg_n = mask.float().sum(dim=1).mean()
            avg_matched = match.pred_matched.float().sum(dim=1).mean()
            self.log(f"{stage}/avg_nodes", avg_n, prog_bar=False)
            self.log(f"{stage}/avg_matched", avg_matched, prog_bar=False)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._step(batch, "val")
        if batch_idx == 0:
            self.log_reconstruction(batch, batch_idx)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="./data/quickdraw_graphs.pkl")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-nodes", type=int, default=0, help="Optional node subsample (0 disables)")
    p.add_argument("--max-neighbors", type=int, default=4, help="Max neighbors per node")
    p.add_argument("--seed", type=int, default=42)

    # model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--enc-heads", type=int, default=8)
    p.add_argument("--enc-depth", type=int, default=8)
    p.add_argument("--dec-heads", type=int, default=8)
    p.add_argument("--dec-depth", type=int, default=6)
    p.add_argument("--fourier-freqs", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--q", type=int, default=512, help="decoder query count (max nodes output)")
    p.add_argument("--codebook-size", type=int, default=1024)
    p.add_argument("--num-codebooks", type=int, default=4)
    p.add_argument("--commitment-weight", type=float, default=0.25)

    # loss weights
    p.add_argument("--w-coord", type=float, default=10.0)
    p.add_argument("--w-exist", type=float, default=1.0)
    p.add_argument("--w-edge", type=float, default=1.0)
    p.add_argument("--pos-weight-edge", type=float, default=10.0)

    # train
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--gradient-clip-val", type=float, default=1.0)

    # output
    p.add_argument("--logdir", type=str, default="./logs_rvqvae")
    p.add_argument("--ckptdir", type=str, default="./ckpt_rvqvae")
    args = p.parse_args()

    L.seed_everything(args.seed, workers=True)

    dm = QuickDrawGraphDataModule(
        data_path=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        max_nodes=args.max_nodes,
        max_neighbors=args.max_neighbors,
        seed=args.seed,
    )

    model = VectorGraphRVQVAE(
        d_model=args.d_model,
        enc_heads=args.enc_heads,
        enc_depth=args.enc_depth,
        dec_heads=args.dec_heads,
        dec_depth=args.dec_depth,
        fourier_freqs=args.fourier_freqs,
        dropout=args.dropout,
        q=args.q,
        codebook_size=args.codebook_size,
        num_codebooks=args.num_codebooks,
        commitment_weight=args.commitment_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        w_coord=args.w_coord,
        w_exist=args.w_exist,
        w_edge=args.w_edge,
        pos_weight_edge=args.pos_weight_edge,
    )

    ckptdir = Path(args.ckptdir)
    ckptdir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckptdir),
            filename="rvqvae-{epoch:03d}-{val/loss:.6f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
        )
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision="bf16-mixed",
        default_root_dir=args.logdir,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=20,
        limit_train_batches=0.01,
        limit_val_batches=0.01
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
