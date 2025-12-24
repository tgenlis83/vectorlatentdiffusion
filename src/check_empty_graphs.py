#!/usr/bin/env python3
"""
Script to check for empty or small graphs in the dataset and visualize them.
"""

import argparse
import pickle
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np

def plot_graph_sample(ax, sample: Dict, title_prefix: str = ""):
    x = sample["x"]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    
    edge_index = sample["edge_index"]
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()

    # Draw edges
    if edge_index.shape[1] > 0:
        for k in range(edge_index.shape[1]):
            u = edge_index[0, k]
            v = edge_index[1, k]
            if u < len(x) and v < len(x):
                ax.plot([x[u, 0], x[v, 0]], [x[u, 1], x[v, 1]], linewidth=0.5, color='black', alpha=0.6)

    # Draw nodes
    if len(x) > 0:
        ax.scatter(x[:, 0], x[:, 1], s=10, c='red', alpha=0.8, edgecolors='none')
    
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    
    word = sample.get('word', 'Unknown')
    num_nodes = len(x)
    num_edges = edge_index.shape[1] if edge_index.ndim > 1 else 0
    ax.set_title(f"{title_prefix}{word}\nN={num_nodes}, E={num_edges}", fontsize=8)

def main():
    parser = argparse.ArgumentParser(description="Check for empty graphs in the dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to input pickle file")
    parser.add_argument("--threshold", type=int, default=5, help="Threshold for 'small' graph (number of nodes)")
    args = parser.parse_args()

    p = Path(args.data)
    if not p.exists():
        print(f"Error: File {p} not found.")
        return

    print(f"Loading dataset from {p}...")
    with open(p, "rb") as f:
        samples: List[Dict[str, Any]] = pickle.load(f)
    
    print(f"Loaded {len(samples)} samples.")

    node_counts = []
    edge_counts = []
    max_neighbor_counts = []
    
    # For calculating potential padding waste
    total_nodes_if_padded_to_max = 0
    actual_total_nodes = 0
    
    total_neighbors_if_padded_to_16 = 0 # Assuming default max_neighbors=16
    actual_total_neighbors = 0

    empty_nodes = []
    empty_edges = []
    small_graphs = []
    normal_graphs = []

    for i, sample in enumerate(samples):
        x = sample["x"]
        edge_index = sample["edge_index"]
        neighbors = sample.get("neighbors", None)
        
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1] if edge_index.ndim > 1 else 0
        
        node_counts.append(num_nodes)
        edge_counts.append(num_edges)
        actual_total_nodes += num_nodes

        if neighbors is not None:
            # neighbors is [N, M] or list of lists? 
            # In dataset.py it seems to be a tensor [N, M] or similar structure
            if torch.is_tensor(neighbors):
                # Count valid neighbors (assuming -1 is padding)
                valid_neighbors = (neighbors >= 0).sum(dim=1)
                max_n = valid_neighbors.max().item() if len(valid_neighbors) > 0 else 0
                max_neighbor_counts.append(max_n)
                actual_total_neighbors += valid_neighbors.sum().item()
                total_neighbors_if_padded_to_16 += num_nodes * 16
            elif isinstance(neighbors, list):
                 # If it's a list of lists
                max_n = max((len(n) for n in neighbors), default=0)
                max_neighbor_counts.append(max_n)
                current_neighbors = sum(len(n) for n in neighbors)
                actual_total_neighbors += current_neighbors
                total_neighbors_if_padded_to_16 += num_nodes * 16

        if num_nodes == 0:
            empty_nodes.append(i)
        elif num_nodes < args.threshold:
            small_graphs.append(i)
        else:
            normal_graphs.append(i)

        if num_edges == 0:
            empty_edges.append(i)

    # Statistics
    node_counts = np.array(node_counts)
    edge_counts = np.array(edge_counts)
    max_neighbor_counts = np.array(max_neighbor_counts)
    
    max_nodes_in_dataset = node_counts.max() if len(node_counts) > 0 else 0
    total_nodes_if_padded_to_max = len(samples) * max_nodes_in_dataset
    
    print("\n" + "="*40)
    print("DATASET STATISTICS")
    print("="*40)
    print(f"Total Samples: {len(samples)}")
    
    print(f"\nNode Counts:")
    print(f"  Min: {node_counts.min()}")
    print(f"  Max: {node_counts.max()}")
    print(f"  Mean: {node_counts.mean():.2f}")
    print(f"  Median: {np.median(node_counts):.2f}")
    print(f"  95th Percentile: {np.percentile(node_counts, 95):.2f}")
    print(f"  99th Percentile: {np.percentile(node_counts, 99):.2f}")

    print(f"\nEdge Counts:")
    print(f"  Min: {edge_counts.min()}")
    print(f"  Max: {edge_counts.max()}")
    print(f"  Mean: {edge_counts.mean():.2f}")
    print(f"  Median: {np.median(edge_counts):.2f}")

    if len(max_neighbor_counts) > 0:
        print(f"\nMax Neighbors per Node (per graph):")
        print(f"  Min: {max_neighbor_counts.min()}")
        print(f"  Max: {max_neighbor_counts.max()}")
        print(f"  Mean: {max_neighbor_counts.mean():.2f}")
        print(f"  Median: {np.median(max_neighbor_counts):.2f}")
        print(f"  95th Percentile: {np.percentile(max_neighbor_counts, 95):.2f}")
        print(f"  99th Percentile: {np.percentile(max_neighbor_counts, 99):.2f}")

    print(f"\nPadding / Waste Analysis:")
    if total_nodes_if_padded_to_max > 0:
        waste_nodes = 1.0 - (actual_total_nodes / total_nodes_if_padded_to_max)
        print(f"  If padded to max nodes ({max_nodes_in_dataset}): {waste_nodes*100:.2f}% of node tensor would be padding.")
    
    if total_neighbors_if_padded_to_16 > 0:
        waste_neighbors = 1.0 - (actual_total_neighbors / total_neighbors_if_padded_to_16)
        print(f"  If padded to 16 neighbors: {waste_neighbors*100:.2f}% of neighbor tensor would be padding.")

    print("="*40 + "\n")

    print(f"Found {len(empty_nodes)} samples with 0 nodes.")
    print(f"Found {len(empty_edges)} samples with 0 edges.")
    print(f"Found {len(small_graphs)} samples with < {args.threshold} nodes (but > 0).")

    # Visualize
    # We want to show a grid of problematic graphs.
    # Priority: Empty nodes -> Small graphs -> Empty edges (if nodes > 0)
    
    indices_to_plot = []
    labels = []

    # Add some empty node graphs
    for idx in empty_nodes[:9]:
        indices_to_plot.append(idx)
        labels.append("Empty Nodes")
    
    # Add some small graphs
    remaining = 9 - len(indices_to_plot)
    if remaining > 0:
        for idx in small_graphs[:remaining]:
            indices_to_plot.append(idx)
            labels.append("Small Graph")

    # Add some empty edge graphs (that have nodes)
    remaining = 9 - len(indices_to_plot)
    if remaining > 0:
        # Filter empty_edges to exclude those already in empty_nodes or small_graphs
        valid_empty_edges = [idx for idx in empty_edges if idx not in empty_nodes and idx not in small_graphs]
        for idx in valid_empty_edges[:remaining]:
            indices_to_plot.append(idx)
            labels.append("No Edges")

    # If we still have space, add normal graphs
    remaining = 9 - len(indices_to_plot)
    if remaining > 0:
        random.shuffle(normal_graphs)
        for idx in normal_graphs[:remaining]:
            indices_to_plot.append(idx)
            labels.append("Normal")

    if not indices_to_plot:
        print("No graphs to plot based on criteria.")
        return

    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    for i, idx in enumerate(indices_to_plot):
        if i >= len(axes):
            break
        sample = samples[idx]
        plot_graph_sample(axes[i], sample, title_prefix=f"[{labels[i]}] ")

    plt.tight_layout()
    output_file = "empty_graphs_check.png"
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")
    # plt.show() # Cannot show in headless environment

if __name__ == "__main__":
    main()
