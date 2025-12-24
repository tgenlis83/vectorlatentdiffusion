#!/usr/bin/env python3
"""
SOTA-ish Vector-Graph Latent Diffusion (Rectified Flow / Velocity Matching) with:

Backbone      : DiT-XL/2 style (Extra Large, patch size = 2) adapted to 1D sequences
Conditioning  : adaLN-Zero (Adaptive LayerNorm w/ zero-init gates) + "City Style" embedding
Positional    : RoPE (Rotary Positional Embeddings)
Latent shape  : 1D sequence (flattened latent tokens); variable length via padding mask
Attention     : Multi-Scale Deformable Attention (MSDeformAttn) for O(T * L * K) compute
Loss          : Velocity matching (rectified flow): predict v that moves x_t toward x_1

This script expects:
- Your vectorized street-graph dataset pickle: city_street_graphs.pkl (x [N,2], edge_index [2,E])
- A trained geometry-aware RVQ-VAE checkpoint (the one from your RVQ-VAE script)
  which provides encoder + RVQ quantizer to produce latent tokens x1.

We diffuse in the *continuous quantized latent embeddings* (the RVQ output vectors), like modern VQ-latent diffusion.

Train:
  python dit_msdeform_flow.py \
    --data city_street_graphs.pkl \
    --vqvae-ckpt ckpt_rvqvae/rvqvae-....ckpt \
    --batch-size 6 --max-epochs 100 \
    --max-nodes 0 \
    --cond-field country \
    --model-preset xl

Sample (optional demo):
  python dit_msdeform_flow.py ... --sample-after-train --sample-steps 60 --sample-nodes 800

Notes:
- Patch size 2 means we patchify the latent token sequence in pairs before the Transformer.
- MSDeformAttn here is 1D multi-scale deformable attention (Deformable-DETR style) with
  RoPE applied to q/k for sampled keys.
- "City Style" conditioning is implemented as an embedding over a chosen metadata field
  (country/city). You can swap to a real style label later.

Deps:
  pip install torch lightning numpy tqdm
Optional:
  pip install scipy   (not needed here; only for VAE matching, which diffusion doesn't use)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from src.dataset.dataset import QuickDrawGraphDataModule
from src.model.dit import LatentFlowMSDeformDiT

# =============================================================================
# ---------------------------------- Main ------------------------------------
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="city_street_graphs.pkl")
    p.add_argument("--vqvae-ckpt", type=str, required=True, help="RVQ-VAE checkpoint (.ckpt)")

    # data
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-nodes", type=int, default=0)
    p.add_argument("--max-neighbors", type=int, default=16)
    p.add_argument("--node-order", type=str, default="xy", choices=["xy", "none"])

    # conditioning
    p.add_argument("--cond-field", type=str, default="country", choices=["country", "city", "none"])

    # model
    p.add_argument("--model-preset", type=str, default="xl", choices=["xl", "l", "b", "s"])
    p.add_argument("--patch-size", type=int, default=2)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)

    # msdeform
    p.add_argument("--num-scales", type=int, default=3)
    p.add_argument("--num-points", type=int, default=4)
    p.add_argument("--max-offset-frac", type=float, default=0.25)

    # optim/train
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--precision", type=str, default="32-true")
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")

    # output
    p.add_argument("--logdir", type=str, default="./logs_dit_msdeform")
    p.add_argument("--ckptdir", type=str, default="./ckpt_dit_msdeform")

    # sampling demo
    p.add_argument("--sample-after-train", action="store_true")
    p.add_argument("--sample-nodes", type=int, default=800)
    p.add_argument("--sample-cond-id", type=int, default=0)
    p.add_argument("--sample-steps", type=int, default=60)
    p.add_argument("--sample-method", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument("--exist-thr", type=float, default=0.5)
    p.add_argument("--edge-thr", type=float, default=0.5)
    args = p.parse_args()

    L.seed_everything(42, workers=True)

    dm = QuickDrawGraphDataModule(
        data_path=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        max_nodes=args.max_nodes,
        max_neighbors=args.max_neighbors,
        node_order=args.node_order,
        cond_field=args.cond_field,
        seed=42,
    )
    dm.setup()
    assert dm.cond_vocab_size is not None

    model = LatentFlowMSDeformDiT(
        vqvae_ckpt=args.vqvae_ckpt,
        cond_vocab_size=dm.cond_vocab_size,
        model_preset=args.model_preset,
        patch_size=args.patch_size,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        num_scales=args.num_scales,
        num_points=args.num_points,
        max_offset_frac=args.max_offset_frac,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
    )

    ckptdir = Path(args.ckptdir)
    ckptdir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckptdir),
            filename="dit-msdeform-{epoch:03d}-{val/loss:.6f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
        )
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        default_root_dir=args.logdir,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=20,
    )
    trainer.fit(model, dm)

    if args.sample_after_train:
        # Sample latents then decode to a graph with the RVQ-VAE decoder
        import matplotlib.pyplot as plt

        model.eval()
        hq = model.sample_latents(
            T=int(args.sample_nodes),
            cond_id=int(args.sample_cond_id),
            steps=int(args.sample_steps),
            method=str(args.sample_method),
            use_ema=True,
        )  # [1,T,D]
        mem_mask = torch.ones((1, hq.shape[1]), device=hq.device, dtype=torch.bool)

        out = model.decode_sample_to_graph(
            hq=hq,
            mem_mask=mem_mask,
            exist_thr=float(args.exist_thr),
            edge_thr=float(args.edge_thr),
        )

        coords = out["coords"].detach().cpu()
        exist = out["exist"].detach().cpu()
        adj = out["adj"].detach().cpu()

        # plot predicted edges among existing nodes
        keep = torch.where(exist)[0]
        coords_k = coords[keep]
        idx_map = {int(old): i for i, old in enumerate(keep.tolist())}

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        # edges
        for i_old in keep.tolist():
            for j_old in keep.tolist():
                if i_old < j_old and adj[i_old, j_old].item():
                    i = idx_map[int(i_old)]
                    j = idx_map[int(j_old)]
                    ax.plot([coords_k[i, 0], coords_k[j, 0]], [coords_k[i, 1], coords_k[j, 1]], linewidth=0.3)
        ax.scatter(coords_k[:, 0], coords_k[:, 1], s=3)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(f"Sample (cond_id={args.sample_cond_id})")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
