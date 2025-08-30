import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from scipy.ndimage import binary_closing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Degree simplification (cycle protect)
def simplify_graph_preserve_cycles(G, protected_nodes):
    G_simple = nx.Graph()
    visited = set()

    for node in G.nodes():
        if node in visited or node in protected_nodes or G.degree(node) != 1:
            continue

        path = [node]
        current = node
        prev = None

        while True:
            neighbors = list(G.neighbors(current))
            next_nodes = [n for n in neighbors if n != prev]
            if len(next_nodes) != 1:
                break
            next_node = next_nodes[0]
            path.append(next_node)
            prev, current = current, next_node

            if current in protected_nodes or G.degree(current) != 2:
                break

        visited.update(path)
        if len(path) > 1:
            G_simple.add_edge(path[0], path[-1])

    for u, v in G.edges():
        if (G.degree(u) != 2 or G.degree(v) != 2) or (u in protected_nodes or v in protected_nodes):
            G_simple.add_edge(u, v)

    return G_simple


# Input data
vol = np.load('file_name')
mask = vol.astype(bool)
Z, Y, X = mask.shape
print(f"Layer number: {Z}")

out_dir = 'test'
os.makedirs(out_dir, exist_ok=True)

neighbors = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),          (0, 1),
             (1, -1),  (1, 0), (1, 1)]

min_cycle_length = 0
nodes_number = 4
hole_area_threshold = 4
small_object_size = 60

all_cycles = []

for z in range(Z):
    slice_2d = mask[z]
    if not np.any(slice_2d):
        continue


    cleaned = binary_closing(slice_2d, structure=np.ones((5,5)))  # Fill in the small black dots
    cleaned = remove_small_holes(cleaned, area_threshold=hole_area_threshold)
    cleaned = remove_small_objects(cleaned, min_size=small_object_size)
    skeleton_2d = skeletonize(cleaned)
    coords = np.transpose(np.nonzero(skeleton_2d))

    # Build G
    G = nx.Graph()
    for y, x in coords:
        G.add_node((y, x))
        for dy, dx in neighbors:
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < Y and 0 <= nx_ < X and skeleton_2d[ny, nx_]:
                G.add_edge((y, x), (ny, nx_))

    # find all cycles
    original_cycles = nx.cycle_basis(G)
    protected_nodes = set(node for cycle in original_cycles for node in cycle)

    # simplify (protect cycles)
    G_simple = simplify_graph_preserve_cycles(G, protected_nodes)

    # save
    fig_graph, ax_graph = plt.subplots(figsize=(4, 4), dpi=300)
    pos = {node: (node[1], -node[0]) for node in G_simple.nodes()}
    nx.draw(G_simple, pos, node_size=10, node_color='red', edge_color='blue', ax=ax_graph)
    ax_graph.set_title(f"Simplified Graph - Z={z}")
    ax_graph.axis('off')
    plt.savefig(f"{out_dir}/layer_{z:03d}_graph.png")
    plt.close(fig_graph)

    # extract cycle
    cycles = nx.cycle_basis(G_simple)

    # plot
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.imshow(slice_2d, cmap='gray')
    cycle_info = []
    cycle_id = 0
    valid_cycles = 0

    for i, cycle in enumerate(cycles):

        if len(cycle) < nodes_number:
            continue

        points = np.array(cycle)
        dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        length = np.max(dists)

        if length < min_cycle_length:
            continue

        valid_cycles += 1

        path = np.array(cycle + [cycle[0]])
        ax.plot(path[:, 1], path[:, 0], linewidth=0.5, alpha=0.8)

        cycle_info.append({
            'layer': z,
            'cycle_id': cycle_id,
            'length': length,
            'positions': cycle
        })
        cycle_id += 1

        all_cycles.append({
            'layer': z,
            'length': length
        })

    ax.set_title(f'Z = {z} | {valid_cycles} cycles')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/layer_{z:03d}_cycles.png")
    plt.close()


    with open(f"{out_dir}/layer_{z:03d}_cycles.txt", 'w') as f:
        for c in cycle_info:
            f.write(f"Cycle {c['cycle_id']} (length={c['length']:.2f}):\n")
            for yx in c['positions']:
                f.write(f"  {yx}\n")
            f.write('\n')

print("Simplified graphs and cycles saved.")


all_lengths = [c['length'] for c in all_cycles]
plt.figure(figsize=(6,4))
plt.hist(all_lengths, bins=35, color="lightcoral", edgecolor="black", alpha=0.7)
plt.xlabel("Cycle length (diameter)")
plt.ylabel("Frequency")
plt.title("Overall cycle length distribution across all layers")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()