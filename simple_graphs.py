# Abby Headstrom 
# April 2026
# This graph contains code for the different methods of converting hypergraphs to simple graphs
# Using the different weighting schemes

import os
import xgi as xgi
import numpy as np
from collections import defaultdict
from math import comb
import json

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', FutureWarning)

# This function converts the hypergraph to a simple graph using the total weight preserving method
def to_simple_total_weight(H, betas=None):
    '''
    Takes an XGI hypergraph and converts it to a simple graph 
    
    TAKE ONE: jsut using the total weight preserving method (will add others later)
      [ w(e) / C(|e|, 2) ] 

    Using vectorized approach: graph = B @ W  @ B.T where B is the incidence matrix
    '''

    # Get the incidence matrix 
    B = xgi.incidence_matrix(H, sparse=False)  # get the incidence matrix as a dense numpy array

    # make the weight matrix W
    # diagonal matrix st the diagonal entry for edge e is w(e) / C(|e|, 2)
    # edge_weights = np.array([1.0] * len(H.edges)) 
        # if betas is not none, use the provided betas to calculate the edge weights 

    edge_weights = np.array([1.0] * len(H.edges))
    if betas is not None:
        betas_array = np.array(betas)

    # loop through each edge to get the weights
    for i, edge in enumerate(H.edges):  
        edge_size = len(H.edges.members()[i])
        # get the edge size (make sure it is greater than 1)
        # edge_size = len(H.edges[edge])
        if edge_size >= 2:
            if betas is not None:
                edge_weights[i] = betas_array[edge_size -1] / comb(edge_size, 2)
            else:
                edge_weights[i] = 1.0 / comb(edge_size, 2)
        else:
            edge_weights[i] = 0.0  # or skip these edges entirely

    # make the diagonal weight matrix W 
    W = np.diag(edge_weights)

    # get the adjacency matrix for the simple graph 
    A = B @ W @ B.T
    np.fill_diagonal(A, 0)  # remove self-loops

    return A



def to_simple_degree_preserving(H, betas=None):
    # get the incidence matrix
    B = xgi.incidence_matrix(H, sparse=False)
    print("incidence matrix B:")
    print(B)

    # if betas is not none, use the provided betas to calculate the edge weights 
    if betas is not None:
        edge_weights = np.array(betas)

    # make the weight matrix W 
    edge_weights = np.array([1.0] * len(H.edges))

    # loop through each edge 
    # this method preserves the degree of each edge (w(e) / (|e| - 1))
    for i, edge in enumerate(H.edges):
        edge_size = len(H.edges.members()[i])
        if edge_size >= 2:
            if betas is not None:
                edge_weights[i] = betas[edge_size -1] / (edge_size - 1)
            else:
                edge_weights[i] = 1.0 / (edge_size - 1)
        else:
            edge_weights[i] = 0.0

    # make the diagonal weight matrix W 
    W = np.diag(edge_weights)

    # get the adjacency matrix for the simple graph 
    A = B @ W @ B.T
    np.fill_diagonal(A, 0)

    return A

# This method makes every edge in the simple graph have the same weight as its corresponding
# hyperedge in the original hypergraph
def to_simple_graph_1to1(H, betas=None): 
    # get the incidence matrix
    B = xgi.incidence_matrix(H, sparse=False)

    # if betas is not none, use the provided betas to calculate the edge weights 
    if betas is not None:
        betas_array = np.array(betas)
    
    edge_weights = np.array([1.0] * len(H.edges))


    # make the weight matrix W 
    # edge_weights = np.array([1.0] * len(H.edges))

    # make the diagonal weight matrix W 
    W = np.diag(edge_weights)
    
    # if betas is not none, scale the edge weights by their size
    if betas is not None:
        for i, edge in enumerate(H.edges):
            edge_size = len(H.edges.members()[i])
            if edge_size >= 2:
                W[i, i] = betas_array[edge_size -1]
            else:
                W[i, i] = 0.0
    # get the adjacency matrix for the simple graph 
    A = B @ W @ B.T
    np.fill_diagonal(A, 0)

    return A

# The following functions are for saving the adjacency matrices and node labels to files
# so that they can be compared with the saved Julia node labels more easily.
def save_matrix_with_labels(A, node_order, matrix_path, labels_path):
    np.savetxt(matrix_path, A, delimiter=",")
    with open(labels_path, "w") as f:
        json.dump(node_order, f)
    print(f"Saved matrix to {matrix_path}")
    print(f"Saved node order ({len(node_order)} nodes) to {labels_path}")


def load_matrix_with_labels(matrix_path, labels_path):
    A = np.loadtxt(matrix_path, delimiter=",")
    with open(labels_path, "r") as f:
        node_order = json.load(f)
    return A, node_order


def community_indices_to_node_labels(communities, node_order):
    return [[node_order[i] for i in community] for community in communities]




if __name__ == "__main__":
    # Load the hypergraph
    H = xgi.read_edgelist("network_storage/xgi_graphs/enron.edgelist")

    # Convert to simple graph using total weight preserving method
    A_total_weight = to_simple_total_weight(H)

    # Convert to simple graph using degree preserving method
    A_degree_preserving = to_simple_degree_preserving(H)

    # Convert to simple graph using 1-to-1 method
    A_1to1 = to_simple_graph_1to1(H)


    # write the adjacency matrices to files
    np.savetxt("adjacency_total_weight.csv", A_total_weight, delimiter=",")
    np.savetxt("adjacency_degree_preserving.csv", A_degree_preserving, delimiter=",")
    np.savetxt("adjacency_1to1.csv", A_1to1, delimiter=",")

    # print some summary statistics about the adjacency matrices
    print("Total Weight Preserving Adjacency Matrix:")
    print(f"Shape: {A_total_weight.shape}")
    print(f"Number of non-zero entries: {np.count_nonzero(A_total_weight)}")
    print(f"Average weight: {A_total_weight.sum() / np.count_nonzero(A_total_weight)}")
    print("\nDegree Preserving Adjacency Matrix:")
    print(f"Shape: {A_degree_preserving.shape}")
    print(f"Number of non-zero entries: {np.count_nonzero(A_degree_preserving)}")
    print(f"Average weight: {A_degree_preserving.sum() / np.count_nonzero(A_degree_preserving)}")
    print("\n1-to-1 Adjacency Matrix:")
    print(f"Shape: {A_1to1.shape}")
    print(f"Number of non-zero entries: {np.count_nonzero(A_1to1)}")
    print(f"Average weight: {A_1to1.sum() / np.count_nonzero(A_1to1)}") 

    # convert the adjacency matrices to Networ