#!/usr/bin/env python3
"""
Download + preprocess the Google Quick, Draw! dataset into *graph samples*.

Multiprocessing version:
- downloads each category NDJSON (cached)
- converts drawings -> graph samples using multiprocessing (default 24 workers)

Output: pickle list[dict] with:
  {
    "x": torch.FloatTensor [N,2],
    "edge_index": torch.LongTensor [2,E],
    "neighbors": torch.LongTensor [N,M],
    "word": str,
    "country": str,
    "key_id": str/int,
    "timestamp": str,
    "recognized": bool,
    "norm": {...},
    "stroke_id": torch.LongTensor [N]   (optional)
  }
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import requests
import torch
from tqdm import tqdm
import multiprocessing as mp

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


CATEGORIES_TXT_RAW = (
    "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
)

QUICKDRAW_SIMPLIFIED_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
QUICKDRAW_RAW_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/raw/"


# ----------------------------- Download helpers -----------------------------

def download_url(url: str, dst_path: Path, chunk_size: int = 1 << 20, force: bool = False) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and dst_path.stat().st_size > 0 and not force:
        return dst_path

    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        with open(tmp_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {dst_path.name}"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    tmp_path.replace(dst_path)
    return dst_path


def fetch_categories_list() -> List[str]:
    r = requests.get(CATEGORIES_TXT_RAW, timeout=60)
    r.raise_for_status()
    return [line.strip() for line in r.text.splitlines() if line.strip()]


def iter_ndjson_lines(path: Path) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


# ---------------------------- Geometry helpers -----------------------------

def normalize_positions(xy: np.ndarray, mode: str = "unit_circle") -> Tuple[np.ndarray, Dict[str, float]]:
    if xy.shape[0] == 0:
        return xy, {"mode": mode, "cx": 0.0, "cy": 0.0, "scale": 1.0}

    if mode == "unit_box":
        mins = xy.min(axis=0)
        maxs = xy.max(axis=0)
        denom = np.maximum(maxs - mins, 1e-6)
        out = (xy - mins) / denom
        meta = {
            "mode": mode,
            "minx": float(mins[0]),
            "miny": float(mins[1]),
            "scalex": float(denom[0]),
            "scaley": float(denom[1]),
        }
        return out.astype(np.float32), meta

    center = xy.mean(axis=0)
    centered = xy - center
    radii = np.linalg.norm(centered, axis=1)
    scale = float(np.max(radii)) if radii.size else 1.0
    if scale < 1e-6:
        scale = 1.0
    out = centered / scale
    meta = {"mode": mode, "cx": float(center[0]), "cy": float(center[1]), "scale": float(scale)}
    return out.astype(np.float32), meta


def _alloc_points_per_stroke(lengths: List[int], budget: int) -> List[int]:
    total = sum(lengths)
    if total <= budget:
        return lengths[:]

    alloc = [max(1, int(round(budget * (L / total)))) for L in lengths]
    alloc = [min(a, L) for a, L in zip(alloc, lengths)]

    cur = sum(alloc)
    if cur > budget:
        while cur > budget:
            for i in range(len(alloc)):
                if cur <= budget:
                    break
                if alloc[i] > 1:
                    alloc[i] -= 1
                    cur -= 1
    elif cur < budget:
        while cur < budget:
            progressed = False
            for i in range(len(alloc)):
                if cur >= budget:
                    break
                if alloc[i] < lengths[i]:
                    alloc[i] += 1
                    cur += 1
                    progressed = True
            if not progressed:
                break
    return alloc


def subsample_strokes(strokes_xy: List[np.ndarray], max_points: int) -> List[np.ndarray]:
    if max_points <= 0:
        return strokes_xy

    lengths = [s.shape[0] for s in strokes_xy]
    total = sum(lengths)
    if total <= max_points:
        return strokes_xy

    alloc = _alloc_points_per_stroke(lengths, max_points)

    out: List[np.ndarray] = []
    for s, k in zip(strokes_xy, alloc):
        L = s.shape[0]
        if L == 0:
            continue
        if k >= L:
            out.append(s)
        elif k == 1:
            out.append(s[[0]])
        else:
            idx = np.linspace(0, L - 1, num=k, dtype=np.int32)
            out.append(s[idx])
    return out


# --------------------------- Graph utilities (Numpy) -------------------------

def reorder_xy_np(x: np.ndarray, edge_index: np.ndarray, stroke_id: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Sort nodes by x then y; remap edge_index accordingly."""
    if x.size == 0:
        return x, edge_index, stroke_id
    # lexsort: last key is primary. We want x primary, then y.
    # So we pass (y, x).
    order = np.lexsort((x[:, 1], x[:, 0]))
    
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size, dtype=np.int64)
    
    x2 = x[order]
    ei2 = inv[edge_index] if edge_index.size > 0 else edge_index
    sid2 = stroke_id[order] if stroke_id is not None else None
    return x2, ei2, sid2


def build_neighbors_np(
    edge_index: np.ndarray,
    num_nodes: int,
    max_neighbors: int,
    include_self: bool = True,
) -> np.ndarray:
    nbrs: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.size > 0:
        src = edge_index[0]
        dst = edge_index[1]
        for s, d in zip(src, dst):
            if 0 <= s < num_nodes and 0 <= d < num_nodes and s != d:
                nbrs[s].append(int(d))

    if include_self:
        for i in range(num_nodes):
            nbrs[i].append(i)

    out = np.full((num_nodes, max_neighbors), -1, dtype=np.int64)
    for i in range(num_nodes):
        u = sorted(list(set(nbrs[i])))
        if len(u) > max_neighbors:
            u = u[:max_neighbors]
        out[i, :len(u)] = u
    return out


def augment_rotate_np(x: np.ndarray, deg: float, rng: np.random.RandomState) -> np.ndarray:
    a = (rng.rand() * 2 - 1) * (deg * np.pi / 180.0)
    ca = np.cos(a)
    sa = np.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return x @ R.T


def augment_jitter_np(x: np.ndarray, std: float, rng: np.random.RandomState) -> np.ndarray:
    return x + rng.randn(*x.shape) * std


# ----------------------- Multiprocessing worker code -----------------------

_WORKER_CFG: Dict[str, Any] = {}


def _worker_init(cfg: Dict[str, Any]):
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _drawing_to_graph_numpy(drawing: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      x_np: [N,2] float32
      ei_np: [2,E] int64
      stroke_id_np: [N] int64
      norm_meta: dict
    """
    max_nodes = int(_WORKER_CFG["max_nodes"])
    normalize = str(_WORKER_CFG["normalize"])
    bidirectional = bool(_WORKER_CFG["bidirectional"])

    strokes_xy: List[np.ndarray] = []
    for stroke in drawing:
        xs = stroke[0]
        ys = stroke[1]
        L = min(len(xs), len(ys))
        if L <= 0:
            continue
        pts = np.stack(
            [np.asarray(xs[:L], dtype=np.float32), np.asarray(ys[:L], dtype=np.float32)],
            axis=1,
        )
        strokes_xy.append(pts)

    strokes_xy = subsample_strokes(strokes_xy, max_points=max_nodes)

    points: List[List[float]] = []
    edges: List[Tuple[int, int]] = []
    stroke_id: List[int] = []
    base = 0
    for sid, s in enumerate(strokes_xy):
        L = s.shape[0]
        if L == 0:
            continue
        for i in range(L):
            points.append([float(s[i, 0]), float(s[i, 1])])
            stroke_id.append(sid)
            if i > 0:
                a = base + i - 1
                b = base + i
                edges.append((a, b))
                if bidirectional:
                    edges.append((b, a))
        base += L

    if len(points) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            {"mode": normalize, "empty": True},
        )

    xy = np.asarray(points, dtype=np.float32)
    xy_norm, norm_meta = normalize_positions(xy, mode=normalize)
    
    if len(edges) == 0:
        ei_np = np.zeros((2, 0), dtype=np.int64)
    else:
        ei_np = np.asarray(edges, dtype=np.int64).T
        
    stroke_id_np = np.asarray(stroke_id, dtype=np.int64)
    return xy_norm, ei_np, stroke_id_np, norm_meta


def _process_line(args: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    """
    args: (json_line, category_default)
    Returns a lightweight sample dict with numpy arrays (to reduce IPC overhead).
    """
    line, cat_default = args
    try:
        rec = json.loads(line)
    except Exception:
        return None

    if _WORKER_CFG["recognized_only"] and not rec.get("recognized", True):
        return None

    drawing = rec.get("drawing", None)
    if drawing is None:
        return None

    x_np, ei_np, sid_np, norm_meta = _drawing_to_graph_numpy(drawing)
    if x_np.shape[0] < int(_WORKER_CFG["min_nodes"]) or ei_np.shape[1] == 0:
        return None

    # --- Deterministic processing (moved from dataset.py) ---

    # 1. Node order
    if _WORKER_CFG.get("node_order") == "xy":
        x_np, ei_np, sid_np = reorder_xy_np(x_np, ei_np, sid_np)

    # 2. Neighbors
    max_neighbors = int(_WORKER_CFG.get("max_neighbors", 16))
    neighbors_np = build_neighbors_np(ei_np, x_np.shape[0], max_neighbors)

    return {
        "x_np": x_np,
        "edge_index_np": ei_np,
        "neighbors_np": neighbors_np,
        "stroke_id_np": sid_np,
        "word": rec.get("word", cat_default),
        "country": rec.get("countrycode", "<unk>"),
        "key_id": rec.get("key_id"),
        "timestamp": rec.get("timestamp"),
        "recognized": bool(rec.get("recognized", True)),
        "norm": norm_meta,
    }


# ------------------------------ Visualization ------------------------------

def plot_sample(sample: Dict[str, Any], max_edges: int = 4000):
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return
    x = sample["x"].detach().cpu().numpy()
    ei = sample["edge_index"].detach().cpu().numpy()

    plt.figure(figsize=(4, 4))
    E = ei.shape[1]
    step = max(1, E // max_edges) if E > max_edges else 1
    for k in range(0, E, step):
        u = ei[0, k]
        v = ei[1, k]
        plt.plot([x[u, 0], x[v, 0]], [x[u, 1], x[v, 1]], linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.title(f"{sample.get('word','?')} ({sample.get('country','?')})", fontsize=10)
    plt.show()


# ----------------------------------- Main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./data/quickdraw_graphs.pkl", help="Output pickle path")
    ap.add_argument("--cache-dir", type=str, default="./data/quickdraw_cache", help="Where to cache downloaded ndjson files")

    ap.add_argument("--format", type=str, default="simplified", choices=["simplified", "raw"])
    ap.add_argument("--categories", type=str, default="all",
                    help="Comma-separated categories, or 'all' to use the official categories list.")
    ap.add_argument("--categories-file", type=str, default="",
                    help="Optional path to a text file with one category per line (overrides --categories).")

    ap.add_argument("--max-per-category", type=int, default=1024, help="Max drawings to keep per category (0=no limit)")
    ap.add_argument("--max-total", type=int, default=0, help="Hard cap on total drawings across categories (0=no limit)")
    ap.add_argument("--recognized-only", action="store_true", help="Keep only recognized drawings")

    ap.add_argument("--max-nodes", type=int, default=128, help="Max points per drawing after subsampling (0=no cap)")
    ap.add_argument("--min-nodes", type=int, default=10, help="Skip drawings with fewer points than this")
    ap.add_argument("--normalize", type=str, default="unit_circle", choices=["unit_circle", "unit_box"])

    ap.add_argument("--node-order", type=str, default="none", choices=["none", "xy"], help="Sort nodes by coordinate")
    ap.add_argument("--max-neighbors", type=int, default=4, help="Max neighbors per node")

    ap.add_argument("--workers", type=int, default=1, help="Multiprocessing workers")
    ap.add_argument("--chunksize", type=int, default=512, help="imap_unordered chunksize (tune for speed/memory)")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-download", action="store_true")

    ap.add_argument("--plot", action="store_true", help="Plot a few random samples at end")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # categories
    if args.categories_file:
        cats = [c.strip() for c in Path(args.categories_file).read_text(encoding="utf-8").splitlines() if c.strip()]
    else:
        if args.categories.strip().lower() == "all":
            cats = fetch_categories_list()
        else:
            cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    if not cats:
        raise SystemExit("No categories specified.")

    base = QUICKDRAW_SIMPLIFIED_BASE if args.format == "simplified" else QUICKDRAW_RAW_BASE

    # worker config (shared via initializer)
    worker_cfg = {
        "max_nodes": int(args.max_nodes),
        "min_nodes": int(args.min_nodes),
        "normalize": str(args.normalize),
        "bidirectional": True,
        "recognized_only": bool(args.recognized_only),
        "node_order": str(args.node_order),
        "max_neighbors": int(args.max_neighbors),
    }

    all_samples: List[Dict[str, Any]] = []
    total_kept = 0

    # Safer start method for many platforms (esp. macOS / notebooks)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    for cat in cats:
        filename = f"{cat}.ndjson"
        url = base + quote(filename)
        local_path = cache_dir / args.format / filename
        download_url(url, local_path, force=args.force_download)

        max_this = int(args.max_per_category)
        kept_this = 0

        # Prepare work items: (line, cat_default)
        # We stream lines but must feed them into the pool; this keeps memory moderate.
        lines_iter = iter_ndjson_lines(local_path)

        with mp.Pool(processes=int(args.workers), initializer=_worker_init, initargs=(worker_cfg,)) as pool:
            pbar = tqdm(desc=f"Processing {cat} (mp x{args.workers})", unit="draw")
            # imap_unordered over a generator of (line, cat)
            for out in pool.imap_unordered(_process_line, ((ln, cat) for ln in lines_iter), chunksize=int(args.chunksize)):
                pbar.update(1)

                if out is None:
                    continue

                # Convert numpy -> torch in the main process
                sample = {
                    "x": torch.from_numpy(out["x_np"]).float(),
                    "edge_index": torch.from_numpy(out["edge_index_np"]).long(),
                    "neighbors": torch.from_numpy(out["neighbors_np"]).long(),
                    "stroke_id": torch.from_numpy(out["stroke_id_np"]).long(),
                    "word": out["word"],
                    "country": out["country"],
                    "key_id": out["key_id"],
                    "timestamp": out["timestamp"],
                    "recognized": out["recognized"],
                    "norm": out["norm"],
                }
                all_samples.append(sample)
                kept_this += 1
                total_kept += 1

                pbar.set_postfix({"kept_cat": kept_this, "kept_total": total_kept})

                if max_this > 0 and kept_this >= max_this:
                    # stop early: terminate pool quickly
                    pool.terminate()
                    pool.join()
                    break
                if args.max_total and args.max_total > 0 and total_kept >= int(args.max_total):
                    pool.terminate()
                    pool.join()
                    break

            pbar.close()

        if args.max_total and args.max_total > 0 and total_kept >= int(args.max_total):
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(all_samples)} samples to: {out_path.resolve()}")

    if args.plot and len(all_samples) > 0:
        k = min(9, len(all_samples))
        chosen = random.sample(all_samples, k=k)
        for s in chosen:
            plot_sample(s)


if __name__ == "__main__":
    main()
