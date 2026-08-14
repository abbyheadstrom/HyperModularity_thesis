# This code creates a synthetic hypergraph, runs the clustering algorithm on it 
# exports it to Julia and then imports it back to Python to check if the results are consistent.


from platform import node

import numpy as np
from export_to_julia import export_xgi_to_julia_format
import xgi
import os
from modularitypruning import prune_to_stable_partitions
from modularitypruning.leiden_utilities import repeated_leiden_from_gammas
from simple_graphs import *
from graph_mod_pruning import *

GRAPH_FILES = {
    "total_weight":      "adjacency_total_weight.csv",
    "degree_preserving": "adjacency_degree_preserving.csv",
    "1to1":              "adjacency_1to1.csv",
}

GAMMA_RANGE  = np.linspace(0, 2, 1000)
DATANAME     = "enron"
DATA_DIR     = "data"
GT_PARTITION = "enron_node_partition.csv"



def test_julia_link():

    # create the enron hypergraph
    H = xgi.load_xgi_data("email-enron")
    nodes_list = list(H.nodes)[:20]
    node_subhypergraph = xgi.subhypergraph(H, nodes=nodes_list, keep_isolates=False)


    A_total_weight      = to_simple_total_weight(node_subhypergraph)
    A_degree_preserving = to_simple_degree_preserving(node_subhypergraph)
    A_1to1              = to_simple_graph_1to1(node_subhypergraph)

    save_matrix_with_labels(
        A_total_weight, list(node_subhypergraph.nodes),
        "test_results/adjacency_total_weight.csv",
        "test_results/adjacency_total_weight_nodes.json"
    )
    save_matrix_with_labels(
        A_degree_preserving, list(node_subhypergraph.nodes),
        "test_results/adjacency_degree_preserving.csv",
        "test_results/adjacency_degree_preserving_nodes.json"
    )
    save_matrix_with_labels(
        A_1to1, list(node_subhypergraph.nodes),
        "test_results/adjacency_1to1.csv",
        "test_results/adjacency_1to1_nodes.json"
    )

    # export the original hypergraph to julia format
    export_xgi_to_julia_format(node_subhypergraph, "test_results/enron_subhypergraph.julia")

    # now run modularity community detection on the simple graphs and save them to test_results/partitions/ (may have to create the folder)
    # create the folder if it doesn't exist
    os.makedirs("test_results/partitions", exist_ok=True)
    # do total weight preserving first
    G_total_weight = load_graph("test_results/adjacency_total_weight.csv", "test_results/adjacency_total_weight_nodes.json")
    communities_total_weight = cluster(G_total_weight, "total_weight")
    # do degree preserving next
    G_degree_preserving = load_graph("test_results/adjacency_degree_preserving.csv", "test_results/adjacency_degree_preserving_nodes.json")
    communities_degree_preserving = cluster(G_degree_preserving, "degree_preserving")
    # do 1to1 
    G_1to1 = load_graph("test_results/adjacency_1to1.csv", "test_results/adjacency_1to1_nodes.json")
    communities_1to1 = cluster(G_1to1, "1to1")
    

    # save the communities to files in test_results/partitions/ with names like "communities_total_weight.csv" and "communities_degree_preserving.csv" and "communities_1to1.csv"
    np.savetxt("test_results/partitions/communities_total_weight.csv", communities_total_weight, delimiter=",", fmt="%d")
    np.savetxt("test_results/partitions/communities_degree_preserving.csv", communities_degree_preserving, delimiter=",", fmt="%d")
    np.savetxt("test_results/partitions/communities_1to1.csv", communities_1to1, delimiter=",", fmt="%d")

    # now load the grouund truth partition (enron_node_subhypergraph_partition.csv) 
    julia_to_igraph = load_node_mapping("enron_subhypergraph.julia", data_dir="data/test_results")
    gt_dict = load_ground_truth(
        "enron_subhypergraph_node_partition.csv", 
        julia_to_igraph, 
        20)
    # print the ground truth partition
    print("Ground truth partition:")
    print(gt_dict)

    # now create an array in the same order as the node labels from the .json files 
    # convert data/test_results/adjacency_total_weight_nodes.json to a list of node labels in the order they appear in the adjacency matrix
    with open("test_results/adjacency_total_weight_nodes.json", "r") as f:
        node_order = json.load(f)
    
    # get the groud truth partiition in the same order as the node labels
    gt_partition = []
    for i in range(len(node_order)):
        node_id = node_order[i]
        gt_partition.append(gt_dict.get(int(node_id)))
    print("Ground truth partition in node order:")
    print(gt_partition)


    # now compare the ground truth partitions to the decected communities using variation of information and heatmaps
    # load the detected communities from the .csv files
    communities_total_weight = np.loadtxt("test_results/partitions/communities_total_weight.csv", delimiter=",", dtype=int)
    communities_degree_preserving = np.loadtxt("test_results/partitions/communities_degree_preserving.csv", delimiter=",", dtype=int)
    communities_1to1 = np.loadtxt("test_results/partitions/communities_1to1.csv", delimiter=",", dtype=int)

    # compare the detected communities to the ground truth using variation of information
    variation_i = vi(gt_partition, communities_total_weight)
    print(f"Variation of Information between ground truth and total weight preserving partition: {variation_i}")
    variation_i = vi(gt_partition, communities_degree_preserving)
    print(f"Variation of Information between ground truth and degree preserving partition: {variation_i}")
    variation_i = vi(gt_partition, communities_1to1)
    print(f"Variation of Information between ground truth and 1to1 partition: {variation_i}")

    plot_confusion(gt_partition, communities_total_weight, "total weight", "test_results/confusion_total_weight.png")
    plot_confusion(gt_partition, communities_degree_preserving, "degree preserving", "test_results/confusion_degree_preserving.png")
    plot_confusion(gt_partition, communities_1to1, "1to1", "test_results/confusion_1to1.png")

    

if __name__ == "__main__":
    test_julia_link()

    print("done!")