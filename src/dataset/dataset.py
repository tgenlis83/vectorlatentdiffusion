from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from tqdm import tqdm


# =============================================================================
# --------------------------- Graph utilities ---------------------------------
# =============================================================================

def pad_list_3d(xs: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(xs)
    T = max(x.shape[0] for x in xs) if B else 0
    D = xs[0].shape[1] if B else 0
    out = xs[0].new_full((B, T, D), float(pad_value))
    mask = torch.zeros((B, T), dtype=torch.bool, device=xs[0].device)
    for b, x in enumerate(xs):
        n = x.shape[0]
        out[b, :n] = x
        mask[b, :n] = True
    return out, mask


def pad_list_neighbors(neis: List[torch.Tensor], T: int, pad_value: int = -1) -> torch.Tensor:
    B = len(neis)
    M = neis[0].shape[1] if B else 0
    out = torch.full((B, T, M), pad_value, dtype=torch.long, device=neis[0].device)
    for b, n in enumerate(neis):
        out[b, : n.shape[0]] = n
    return out


# =============================================================================
# --------------------------- Dataset + Collate -------------------------------
# =============================================================================

class QuickDrawGraphDataset(Dataset):
    """
    Loads pickle samples from create_quickdraw_graph_dataset.py and returns:
      x: [N,2]
      neighbors: [N,M]
      cond_id: int (e.g., category word)
      meta: dict (word/country/etc.)
    """
    def __init__(
        self,
        pickle_path: str,
        max_nodes: int = 0,
        max_neighbors: int = 16,
        node_order: str = "none",   # "none" keeps stroke order; "xy" sorts by x/y
        cond_field: str = "word",   # "word" | "country" | "none" | any key present
        seed: int = 42,
        augment_rotate: bool = False,
        rotate_deg: float = 15.0,
        augment_jitter: bool = False,
        jitter_std: float = 0.01,
    ):
        print(f"Loading QuickDrawGraphDataset from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            self.samples: List[Dict[str, Any]] = pickle.load(f)
        print(f"Loaded {len(self.samples)} samples.")
        if len(self.samples) == 0:
            raise RuntimeError("Pickle contains 0 samples.")

        # These parameters are now handled in create_dataset.py, but kept here for compatibility
        self.max_nodes = int(max_nodes)
        self.max_neighbors = int(max_neighbors)
        self.node_order = str(node_order)
        self.cond_field = str(cond_field)
        self.seed = int(seed)

        self.augment_rotate = bool(augment_rotate)
        self.rotate_deg = float(rotate_deg)
        self.augment_jitter = bool(augment_jitter)
        self.jitter_std = float(jitter_std)

        # Build conditioning vocab (stable ordering)
        if self.cond_field == "none":
            self.cond_vocab = {"<none>": 0}
        else:
            vals: List[str] = []
            for s in tqdm(self.samples):
                if self.cond_field in s:
                    vals.append(str(s[self.cond_field]))
                elif "meta" in s and self.cond_field in s["meta"]:
                    vals.append(str(s["meta"][self.cond_field]))
            uniq = sorted(set(vals))
            self.cond_vocab = {"<unk>": 0}
            for i, v in enumerate(uniq, start=1):
                self.cond_vocab[v] = i

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cond_id(self, s: Dict[str, Any]) -> int:
        if self.cond_field == "none":
            return 0
        v = None
        if self.cond_field in s:
            v = s[self.cond_field]
        elif "meta" in s and self.cond_field in s["meta"]:
            v = s["meta"][self.cond_field]
        if v is None:
            return 0
        return self.cond_vocab.get(str(v), 0)

    def _maybe_augment(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        # deterministic per-sample RNG
        g = torch.Generator(device=x.device)
        g.manual_seed(self.seed * 1_000_003 + idx)

        out = x
        if self.augment_rotate:
            # random angle in [-rotate_deg, rotate_deg]
            a = (torch.rand((), generator=g, device=x.device) * 2 - 1) * (self.rotate_deg * np.pi / 180.0)
            ca = torch.cos(a)
            sa = torch.sin(a)
            R = torch.stack([torch.stack([ca, -sa]), torch.stack([sa, ca])])  # [2,2]
            out = out @ R.T

        if self.augment_jitter:
            out = out + torch.randn_like(out, generator=g) * self.jitter_std

        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        x = s["x"].float()
        ei = s["edge_index"].long()
        neighbors = s["neighbors"].long()
        x = self._maybe_augment(x, idx)
        cond_id = self._get_cond_id(s)
        meta = {k: v for k, v in s.items() if k not in ("x", "edge_index", "neighbors")}
        return {"x": x, "edge_index": ei, "neighbors": neighbors, "mask_len": x.size(0), "cond_id": cond_id, "meta": meta}



def collate_graphs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs = [b["x"] for b in batch]
    eis = [b["edge_index"] for b in batch]
    neis = [b["neighbors"] for b in batch]
    cond = torch.tensor([b["cond_id"] for b in batch], dtype=torch.long)
    metas = [b["meta"] for b in batch]

    x_pad, mask = pad_list_3d(xs, pad_value=0.0)   # [B,T,2], [B,T]
    T = x_pad.shape[1]
    neighbors = pad_list_neighbors(neis, T=T)      # [B,T,M]

    return {"x_pad": x_pad, "mask": mask, "neighbors": neighbors, "edge_index": eis, "cond_id": cond, "meta": metas}


class QuickDrawGraphDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        max_nodes: int = 0,
        max_neighbors: int = 16,
        node_order: str = "none",
        cond_field: str = "word",
        seed: int = 42,
        augment_rotate: bool = False,
        rotate_deg: float = 15.0,
        augment_jitter: bool = False,
        jitter_std: float = 0.01,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.val_split = float(val_split)
        self.max_nodes = int(max_nodes)
        self.max_neighbors = int(max_neighbors)
        self.node_order = str(node_order)
        self.cond_field = str(cond_field)
        self.seed = int(seed)

        self.augment_rotate = bool(augment_rotate)
        self.rotate_deg = float(rotate_deg)
        self.augment_jitter = bool(augment_jitter)
        self.jitter_std = float(jitter_std)

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.cond_vocab_size: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        full = QuickDrawGraphDataset(
            self.data_path,
            max_nodes=self.max_nodes,
            max_neighbors=self.max_neighbors,
            node_order=self.node_order,
            cond_field=self.cond_field,
            seed=self.seed,
            augment_rotate=self.augment_rotate,
            rotate_deg=self.rotate_deg,
            augment_jitter=self.augment_jitter,
            jitter_std=self.jitter_std,
        )
        self.cond_vocab_size = len(full.cond_vocab)

        n = len(full)
        idx = list(range(n))
        rng = random.Random(self.seed)
        rng.shuffle(idx)
        n_val = int(round(n * self.val_split))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        self.train_ds = torch.utils.data.Subset(full, train_idx)
        self.val_ds = torch.utils.data.Subset(full, val_idx) if n_val > 0 else None

    def train_dataloader(self):
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_graphs,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_graphs,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )
