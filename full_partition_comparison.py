from platform import node

import csv
import json
import numpy as np
from export_to_julia import export_xgi_to_julia_format
import xgi
import os
from modularitypruning import prune_to_stable_partitions
from modularitypruning.leiden_utilities import repeated_leiden_from_gammas
from simple_graphs import *
from graph_mod_pruning import *
import pandas as pd
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRAPH_TYPES = ["total_weight", "degree_preserving", "1to1", "total_weight_betas", "degree_preserving_betas", "1to1_betas"]
# GRAPH_TYPES = ["total_weight", "degree_preserving", "1to1"]
GAMMA_RANGE = np.linspace(0, 2, 1000)
DATA_DIR    = "data"


# ---------------------------------------------------------------------------
# Path helpers
#
# All outputs live under a single top-level folder:
#
#   partition_comparisons/
#   └── <dataname>/
#       ├── adjacency_<graph_type>.csv
#       ├── adjacency_<graph_type>_nodes.json
#       ├── <dataname>.julia
#       ├── vi_summary.csv
#       ├── confusion_<graph_type>_partition_<n>.png
#       └── partitions/
#           └── communities_<graph_type>.csv
# ---------------------------------------------------------------------------

ROOT_DIR = "partition_comparisons"


def results_dir(dataname: str) -> str:
    """Return the dataset folder inside the top-level partition_comparisons/ directory."""
    return os.path.join(ROOT_DIR, dataname)


def adjacency_path(dataname: str, graph_type: str) -> str:
    """Return the path to the adjacency matrix CSV for a given graph type."""
    return os.path.join(results_dir(dataname), f"adjacency_{graph_type}.csv")


def nodes_path(dataname: str, graph_type: str) -> str:
    """Return the path to the node-label JSON for a given graph type."""
    return os.path.join(results_dir(dataname), f"adjacency_{graph_type}_nodes.json")


def partitions_dir(dataname: str) -> str:
    """Return the folder where community-partition CSVs are stored."""
    return os.path.join(results_dir(dataname), "partitions")


def partition_path(dataname: str, graph_type: str) -> str:
    """Return the path to the detected-communities CSV for a given graph type."""
    return os.path.join(partitions_dir(dataname), f"communities_{graph_type}.csv")


def julia_export_path(dataname: str) -> str:
    """Return the path to the Julia-format hypergraph export file."""
    return os.path.join(results_dir(dataname), f"{dataname}.julia")


def vi_summary_path(dataname: str) -> str:
    """Return the path to the VI summary CSV."""
    return os.path.join(results_dir(dataname), "vi_summary.csv")


def confusion_imgs_dir(dataname: str) -> str:
    """Return the folder where confusion heatmap images are stored."""
    return os.path.join(results_dir(dataname), "confusion_imgs")


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_partition_with_labels(G, partitions, path):
    """Save every partition as (partition_index, node_id, community) rows."""
    node_names = G.vs["name"]
    with open(path, "w") as f:
        f.write("partition_index,node_id,community\n")
        for partition_index, partition in enumerate(partitions):
            for node_name, community in zip(node_names, partition):
                f.write(f"{partition_index},{node_name},{community}\n")


# ---------------------------------------------------------------------------
# Phase 1: build adjacency matrices and export to Julia
# ---------------------------------------------------------------------------

def build_and_export(dataname: str, min_edge_size: int = None, max_edge_size: int = None):
    """
    Load the XGI hypergraph for *dataname*, compute the three simple-graph
    projections (total-weight, degree-preserving, 1-to-1), save their
    adjacency matrices and node-label files, and export the raw hypergraph
    to Julia format.

    Always regenerates all output files from scratch.

    Parameters
    ----------
    dataname       : XGI dataset name (e.g. "email-enron")
    min_edge_size  : optional lower bound on hyperedge size to keep
    max_edge_size  : optional upper bound on hyperedge size to keep
    """
    out_dir = results_dir(dataname)

    # create the results folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # load the hypergraph
    H = xgi.load_xgi_data(dataname)

    # optionally filter by edge size
    if min_edge_size is not None or max_edge_size is not None:
        lo = min_edge_size if min_edge_size is not None else 0
        hi = max_edge_size if max_edge_size is not None else float("inf")
        H = xgi.subhypergraph(
            H,
            edges=[e for e in H.edges if lo <= len(H.edges.members(e)) <= hi],
        )

    nodes_list = list(H.nodes)

    # compute the three simple-graph projections
    A_total_weight      = to_simple_total_weight(H)
    A_degree_preserving = to_simple_degree_preserving(H)
    A_1to1              = to_simple_graph_1to1(H)

    # load betas from data/<dataname>/<dataname>_betas(.csv)
    betas = []
    betas_base = os.path.join(DATA_DIR, dataname, f"{dataname}_betas.csv")
    betas_path = betas_base if os.path.exists(betas_base) else f"{betas_base}.csv"
    with open(betas_path) as f:
        for row in csv.DictReader(f):
            beta_value = row.get("beta") or row.get("Beta_values")
            if beta_value is None:
                raise KeyError(
                    f"Expected a 'beta' or 'Beta_values' column in {betas_path}. "
                    f"Found columns: {list(row.keys())}"
                )
            betas.append(float(beta_value))
    A_degree_preserving_betas = to_simple_degree_preserving(H, betas=betas)
    A_total_weight_betas = to_simple_total_weight(H, betas=betas)
    A_1to1_betas = to_simple_graph_1to1(H, betas=betas)
    print("wrote beta adjacency matrices")

    # save the betas versions to files
    save_matrix_with_labels(
        A_degree_preserving_betas, nodes_list,
        adjacency_path(dataname, "degree_preserving_betas"),
        nodes_path(dataname, "degree_preserving_betas"),
    )

    save_matrix_with_labels(
        A_total_weight_betas, nodes_list,
        adjacency_path(dataname, "total_weight_betas"),
        nodes_path(dataname, "total_weight_betas"),
    )

    save_matrix_with_labels(
        A_1to1_betas, nodes_list,
        adjacency_path(dataname, "1to1_betas"),
        nodes_path(dataname, "1to1_betas"),
    )

    

    # save adjacency matrices and node labels
    save_matrix_with_labels(
        A_total_weight, nodes_list,
        adjacency_path(dataname, "total_weight"),
        nodes_path(dataname, "total_weight"),
    )
    save_matrix_with_labels(
        A_degree_preserving, nodes_list,
        adjacency_path(dataname, "degree_preserving"),
        nodes_path(dataname, "degree_preserving"),
    )
    save_matrix_with_labels(
        A_1to1, nodes_list,
        adjacency_path(dataname, "1to1"),
        nodes_path(dataname, "1to1"),
    )

    # export the original hypergraph to Julia format
    export_xgi_to_julia_format(H, julia_export_path(dataname))

    print(f"[build_and_export] Done. Results saved to '{out_dir}/'.")


# ---------------------------------------------------------------------------
# Phase 2: community detection on each simple-graph projection
# ---------------------------------------------------------------------------

def detect_communities(dataname: str):
    """
    Load each simple-graph adjacency matrix for *dataname*, run modularity
    community detection, and save the resulting partitions to CSV files.

    Always regenerates all partition files from scratch.
    """
    os.makedirs(partitions_dir(dataname), exist_ok=True)

    for graph_type in GRAPH_TYPES:
        # load the graph
        G = load_graph(adjacency_path(dataname, graph_type), nodes_path(dataname, graph_type))

        # run leiden clustering across the gamme range
        communities = cluster(G, graph_type)
        out_path = partition_path(dataname, graph_type)

        # save the stable partitions to CSV files with node labels
        save_partition_with_labels(G, communities, out_path)
        print(f"[detect_communities] Saved partitions for '{graph_type}' → {out_path}")


# ---------------------------------------------------------------------------
# Phase 3: evaluate against ground truth and produce confusion plots
# ---------------------------------------------------------------------------

def evaluate_partitions(dataname: str, gt_partition_file: str):
    """
    Compare every detected partition against the ground truth, compute
    variation of information (VI), plot confusion heatmaps, and save a
    summary CSV.

    Parameters
    ----------
    dataname           : dataset name (used to locate result files)
    gt_partition_file  : path to the ground-truth partition CSV
    """
    # load node ordering from the 1-to-1 nodes file (consistent reference)
    with open(nodes_path(dataname, "1to1"), "r") as f:
        node_order = json.load(f)

    # load and align the ground truth partition to node_order
    julia_to_igraph = load_node_mapping(dataname, data_dir=DATA_DIR)
    gt_dict = load_ground_truth(gt_partition_file, julia_to_igraph, len(julia_to_igraph))

    print("Ground truth partition:")
    print(gt_dict)

    # get the ground truth partition in the same order as the node labels
    gt_part = []
    for i in range(len(node_order)):
        node_id = node_order[i]

        # printing to confirm node labels/community mapping
        print("node_id:", node_id, "community:", gt_dict.get(int(node_id), -1))
        gt_part.append(gt_dict.get(int(node_id), -1))

    print("Ground truth partition in node order:")
    print(gt_part)

    # now compare the ground truth partition to each detected partition using
    # variation of information and confusion heatmaps
    os.makedirs(confusion_imgs_dir(dataname), exist_ok=True)
    n_nodes = len(node_order)
    summary_rows = []

    for graph_type in GRAPH_TYPES:
        pred_df = pd.read_csv(partition_path(dataname, graph_type))

        # loop through each partition in the predictions and compare to the ground truth
        for partition_index, partition_df in pred_df.groupby("partition_index"):
            # build pred_partition in the same node_order as gt_part
            pred_dict = dict(zip(partition_df["node_id"], partition_df["community"]))
            pred_part = [pred_dict.get(int(nid), -1) for nid in node_order]

            # restrict to nodes that have valid labels in both partitions
            valid = [i for i in range(n_nodes) if gt_part[i] != -1 and pred_part[i] != -1]
            gt_valid   = [gt_part[i]   for i in valid]
            pred_valid = [pred_part[i] for i in valid]

            vi_score        = vi(gt_valid, pred_valid)
            num_communities = len(set(pred_valid))
            ari             = adjusted_rand_score(gt_valid, pred_valid)

            summary_rows.append({
                "graph_type":      graph_type,
                "partition_index": int(partition_index),
                "vi_score":        vi_score,
                "adjusted_rand_index": ari,
                "num_communities": num_communities,
            })

            print(f"VI between ground truth and {graph_type} partition {partition_index}: {vi_score}")

            plot_confusion(
                gt_valid,
                pred_valid,
                f"{graph_type} partition {partition_index}",
                os.path.join(confusion_imgs_dir(dataname), f"confusion_{graph_type}_partition_{partition_index}.png"),
            )

    # save VI summary table
    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["graph_type", "partition_index"])
        .reset_index(drop=True)
    )
    summary_df.to_csv(vi_summary_path(dataname), index=False)
    print(f"Saved VI summary table to {vi_summary_path(dataname)}")
    print(summary_df)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(dataname: str, gt_partition_file: str,
         min_edge_size: int = None, max_edge_size: int = None):
    """
    Run the full pipeline for a given dataset:
      1. Build adjacency matrices and export to Julia format.
      2. Run community detection on each simple-graph projection.
      3. Evaluate detected partitions against the ground truth.

    Parameters
    ----------
    dataname           : XGI dataset name (e.g. "email-enron")
    gt_partition_file  : path to the ground-truth node-partition CSV
    min_edge_size      : optional lower bound on hyperedge size
    max_edge_size      : optional upper bound on hyperedge size
    """
    build_and_export(dataname, min_edge_size=min_edge_size, max_edge_size=max_edge_size)
    detect_communities(dataname)
    evaluate_partitions(dataname, gt_partition_file)


if __name__ == "__main__":
    main(
        dataname="house-committees",
        gt_partition_file="house-committees.csv",
        min_edge_size=2,
        max_edge_size=10,
    )

    print("done!")
