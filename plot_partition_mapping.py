import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── Load ─────────────────────────────────────────────────────────────────────
edges = pd.read_csv("map_edges_9.csv")
nodes = pd.read_csv("map_nodes_9.csv")


# ── Build igraph ──────────────────────────────────────────────────────────────
# igraph needs vertices indexed 0..N-1, so remap partition_id
id_to_idx = {int(row.partition_id): i for i, row in nodes.iterrows()}

G = ig.Graph(directed=True)
G.add_vertices(len(nodes))

# Add vertex attributes
G.vs["partition_id"] = list(nodes.partition_id.astype(int))
G.vs["n_clusters"]   = list(nodes.n_clusters.astype(int))
G.vs["is_fixed_point"] = list(nodes.is_fixed_point.astype(bool))
G.vs["attractor"]    = list(nodes.attractor.astype(int))

# Add edges (skip self-loops)
edge_list = [
    (id_to_idx[int(row.src)], id_to_idx[int(row.dst)])
    for _, row in edges.iterrows()
    if not row.is_self_loop
]


G.add_edges(edge_list)

# ── Discrete color palette keyed by K ────────────────────────────────────────
unique_k   = sorted(set(G.vs["n_clusters"]))
palette    = plt.cm.tab10.colors
k_to_color = {k: palette[i % len(palette)] for i, k in enumerate(unique_k)}


vertex_colors = [k_to_color[k] for k in G.vs["n_clusters"]]

# Fixed points get stars and bigger size
vertex_shapes = ["rectangle" if fp else "circle" for fp in G.vs["is_fixed_point"]]

# ── Layout ────────────────────────────────────────────────────────────────────
layout = G.layout("fr")   # Fruchterman-Reingold; try "kk" for Kamada-Kawai or graphopt

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))

ig.plot(G,
    target          = ax,
    layout          = layout,
    vertex_color    = vertex_colors,
    vertex_shape    = vertex_shapes,
    vertex_frame_color = ["black" if fp else "white" for fp in G.vs["is_fixed_point"]],
    vertex_frame_width = [2.0 if fp else 0.3 for fp in G.vs["is_fixed_point"]],
    edge_color      = "grey",
    edge_width      = 0.5,
    edge_arrow_size = 0.5,
    edge_arrow_width = 0.8,
    autocurve       = True,   # curves overlapping edges
    vertex_size = 25
)

# ig.plot(G,
#     target             = ax,
#     layout             = layout,
#     vertex_color       = vertex_colors,
#     # vertex_size        = vertex_sizes,
#     vertex_shape       = vertex_shapes,
#     vertex_frame_color = ["black" if fp else "white" for fp in G.vs["is_fixed_point"]],
#     vertex_frame_width = [2.0 if fp else 0.3 for fp in G.vs["is_fixed_point"]],
#     edge_color         = ["#E24B4A" if int(G.vs[e.target]["partition_id"]) in fixed_ids
#                           else "lightgrey" for e in G.es],
#     edge_width         = [2.5 if int(G.vs[e.target]["partition_id"]) in fixed_ids
#                           else 0.4 for e in G.es],
#     edge_arrow_size    = 1.5,
#     edge_arrow_width   = 1.5,
#     autocurve          = True,
# )

# ── Legend ────────────────────────────────────────────────────────────────────
k_patches    = [mpatches.Patch(color=k_to_color[k], label=f"K = {k}") for k in unique_k]
circle_entry = mlines.Line2D([], [], color="grey", marker="o", linestyle="None",
                              markersize=8, label="Transient (circle)")
star_entry   = mlines.Line2D([], [], color="black", marker="s", linestyle="None",
                              markersize=8, label="Fixed point (square)")

ax.legend(handles=k_patches + [circle_entry, star_entry],
          loc="upper left",
          bbox_to_anchor=(1.01, 1),   # just outside the right edge of the axes
          borderaxespad=0,
          framealpha=0.9,
          fontsize=9)

# make room for the legend on the right
plt.subplots_adjust(right=0.82)

ax.set_title("Partition map  —  squares = fixed points, color = K")
ax.axis("off")

plt.savefig("school_results/map8.png", dpi=150, bbox_inches="tight")
plt.show()


###################################################
# OTHER OPTION FOR THE LAYOUT

# import pandas as pd
# import igraph as ig
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines

# # ── Load ─────────────────────────────────────────────────────────────────────
# edges = pd.read_csv("ACTUALmap_edges_1.csv")
# nodes = pd.read_csv("ACTUALmap_nodes_1.csv")

# # ── Build igraph ──────────────────────────────────────────────────────────────
# # igraph needs vertices indexed 0..N-1, so remap partition_id
# id_to_idx = {int(row.partition_id): i for i, row in nodes.iterrows()}

# G = ig.Graph(directed=True)
# G.add_vertices(len(nodes))

# # Add vertex attributes
# G.vs["partition_id"] = list(nodes.partition_id.astype(int))
# G.vs["n_clusters"]   = list(nodes.n_clusters.astype(int))
# G.vs["is_fixed_point"] = list(nodes.is_fixed_point.astype(bool))
# G.vs["attractor"]    = list(nodes.attractor.astype(int))

# # Add edges (skip self-loops)
# edge_list = [
#     (id_to_idx[int(row.src)], id_to_idx[int(row.dst)])
#     for _, row in edges.iterrows()
#     if not row.is_self_loop
# ]


# G.add_edges(edge_list)

# # ── Discrete color palette keyed by K ────────────────────────────────────────
# unique_k   = sorted(set(G.vs["n_clusters"]))
# palette    = plt.cm.tab10.colors
# k_to_color = {k: palette[i % len(palette)] for i, k in enumerate(unique_k)}


# vertex_colors = [k_to_color[k] for k in G.vs["n_clusters"]]

# # Fixed points get stars and bigger size
# vertex_shapes = ["rectangle" if fp else "circle" for fp in G.vs["is_fixed_point"]]

# # ── Layout ────────────────────────────────────────────────────────────────────
# layout = G.layout("kk")   # Fruchterman-Reingold; try "kk" for Kamada-Kawai

# # ── Plot ──────────────────────────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(14, 10))

# ig.plot(G,
#     target          = ax,
#     layout          = layout,
#     vertex_color    = vertex_colors,
#     vertex_shape    = vertex_shapes,
#     vertex_frame_color = ["black" if fp else "white" for fp in G.vs["is_fixed_point"]],
#     vertex_frame_width = [2.0 if fp else 0.3 for fp in G.vs["is_fixed_point"]],
#     edge_color      = "grey",
#     edge_width      = 0.5,
#     edge_arrow_size = 0.5,
#     edge_arrow_width = 0.8,
#     autocurve       = True,   # curves overlapping edges
#     vertex_size = 25
# )

# # ig.plot(G,
# #     target             = ax,
# #     layout             = layout,
# #     vertex_color       = vertex_colors,
# #     # vertex_size        = vertex_sizes,
# #     vertex_shape       = vertex_shapes,
# #     vertex_frame_color = ["black" if fp else "white" for fp in G.vs["is_fixed_point"]],
# #     vertex_frame_width = [2.0 if fp else 0.3 for fp in G.vs["is_fixed_point"]],
# #     edge_color         = ["#E24B4A" if int(G.vs[e.target]["partition_id"]) in fixed_ids
# #                           else "lightgrey" for e in G.es],
# #     edge_width         = [2.5 if int(G.vs[e.target]["partition_id"]) in fixed_ids
# #                           else 0.4 for e in G.es],
# #     edge_arrow_size    = 1.5,
# #     edge_arrow_width   = 1.5,
# #     autocurve          = True,
# # )

# # ── Legend ────────────────────────────────────────────────────────────────────
# k_patches    = [mpatches.Patch(color=k_to_color[k], label=f"K = {k}") for k in unique_k]
# circle_entry = mlines.Line2D([], [], color="grey", marker="o", linestyle="None",
#                               markersize=8, label="Transient (circle)")
# star_entry   = mlines.Line2D([], [], color="black", marker="s", linestyle="None",
#                               markersize=8, label="Fixed point (square)")

# ax.legend(handles=k_patches + [circle_entry, star_entry],
#           loc="upper left",
#           bbox_to_anchor=(1.01, 1),   # just outside the right edge of the axes
#           borderaxespad=0,
#           framealpha=0.9,
#           fontsize=9)

# # make room for the legend on the right
# plt.subplots_adjust(right=0.82)

# ax.set_title("Partition map  —  squares = fixed points, color = K, size = modularity score")
# ax.axis("off")

# plt.savefig("ACTUAL_map1.png", dpi=150, bbox_inches="tight")
# plt.show()

