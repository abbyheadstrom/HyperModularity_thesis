# This file takes the hypergraph in xgi and exports it to the format
# expected by HyperModularity.jl's read_hypergraph_data() function


import xgi
import os

def export_xgi_to_julia_format(H, dataname, output_dir="data", 
                                 node_labels=None, label_names=None):
    """
    Exports an XGI hypergraph to the file format expected by 
    HyperModularity.jl's read_hypergraph_data():
    
        data/
        └── dataname/
            ├── hyperedges-dataname.txt       (one edge per line, comma-separated node IDs)
            ├── node-labels-dataname.txt      (one integer label per line)
            └── label-names-dataname.txt      (one label name per line)
    
    Node IDs are remapped to 1-indexed integers to match Julia's convention.
    """

    out_path = os.path.join(output_dir, dataname)
    os.makedirs(out_path, exist_ok=True)

    # Use a path-safe base name for output files in case dataname includes folders.
    file_stem = os.path.basename(dataname.rstrip(os.sep))

    # ── Remap node IDs to 1-indexed contiguous integers ──────────────────
    # XGI node IDs can be arbitrary — Julia expects 1:n
    all_nodes = sorted(H.nodes)
    node_map = {old: new for new, old in enumerate(all_nodes, start=1)}
    n = len(all_nodes)

    # ── Write hyperedges ──────────────────────────────────────────────────
    # Format: each line is a comma-separated list of 1-indexed node IDs
    edges_path = os.path.join(out_path, f"hyperedges-{file_stem}.txt")
    with open(edges_path, "w") as f:
        for edge in H.edges.members():
            if len(edge) < 2:
                continue  # skip singletons, Julia's minsize=2 by default
            remapped = sorted(node_map[v] for v in edge)
            f.write(",".join(map(str, remapped)) + "\n")
    print(f"Wrote {H.num_edges} edges to {edges_path}")

    # ── Write node labels ─────────────────────────────────────────────────
    # Format: one integer per line, in order of 1-indexed node IDs
    labels_path = os.path.join(out_path, f"node-labels-{file_stem}.txt")
    with open(labels_path, "w") as f:
        if node_labels is not None:
            # node_labels should be a dict: original_node_id -> integer label
            for old_id in all_nodes:
                f.write(str(node_labels.get(old_id, 0)) + "\n")
        else:
            # Default: label every node as community 1
            for _ in range(n):
                f.write("1\n")
    print(f"Wrote {n} node labels to {labels_path}")

    # ── Write label names ─────────────────────────────────────────────────
    # Format: one string per line, index corresponds to integer label
    names_path = os.path.join(out_path, f"label-names-{file_stem}.txt")
    with open(names_path, "w") as f:
        if label_names is not None:
            for name in label_names:
                f.write(str(name) + "\n")
        else:
            f.write("unknown\n")
    print(f"Wrote label names to {names_path}")

    # ── Write the node ID mapping so you can align results later ─────────
    # This is critical for comparing Julia vs Python partitions afterwards
    mapping_path = os.path.join(out_path, f"node-mapping-{file_stem}.csv")
    with open(mapping_path, "w") as f:
        f.write("Julia_ID,XGI_ID\n")
        for old_id, new_id in node_map.items():
            f.write(f"{new_id},{old_id}\n")
    print(f"Wrote node mapping to {mapping_path}")

    return node_map

if __name__ == "__main__":
    H = xgi.load_xgi_data('email-enron')
    export_xgi_to_julia_format(H, "enron", output_dir="data")