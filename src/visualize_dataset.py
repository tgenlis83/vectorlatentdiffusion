#!/usr/bin/env python3
"""
Script to visualize 3x3 random samples from the dataset pickle file created by create_dataset.py.
"""

import argparse
import pickle
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import torch  # Needed for unpickling torch tensors

def plot_graph_sample(ax, sample: Dict):
    x = sample["x"]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    
    edge_index = sample["edge_index"]
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()

    # Draw edges
    # We iterate over edges to draw lines. 
    # Note: For very dense graphs, LineCollection is faster, but this matches the original script's style.
    for k in range(edge_index.shape[1]):
        u = edge_index[0, k]
        v = edge_index[1, k]
        # Draw line from u to v
        ax.plot([x[u, 0], x[v, 0]], [x[u, 1], x[v, 1]], linewidth=0.3, color='black', alpha=0.5)

    # Draw nodes
    ax.scatter(x[:, 0], x[:, 1], s=2, c='red', alpha=0.8, edgecolors='none')
    
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    
    city = sample.get('city', 'Unknown')
    country = sample.get('country', 'Unknown')
    ax.set_title(f"{city}, {country}", fontsize=8)

def main():
    parser = argparse.ArgumentParser(description="Visualize 3x3 samples from the dataset.")
    parser.add_argument("--input", type=str, default="city_street_graphs.pkl", help="Path to input pickle file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection")
    args = parser.parse_args()

    random.seed(args.seed)

    p = Path(args.input)
    if not p.exists():
        print(f"Error: File not found: {p}")
        print("Please run create_dataset.py first to generate the dataset.")
        return

    print(f"Loading dataset from {p} ...")
    try:
        with open(p, "rb") as f:
            samples = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return
    
    n = len(samples)
    print(f"Loaded {n} samples.")
    if n == 0:
        print("Dataset is empty.")
        return

    k = min(9, n)
    chosen = random.sample(samples, k)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    print("Plotting samples...")
    for i, ax in enumerate(axes):
        if i < k:
            plot_graph_sample(ax, chosen[i])
        else:
            ax.axis("off")
    
    plt.tight_layout()
    print("Displaying plot window...")
    plt.show()

if __name__ == "__main__":
    main()
