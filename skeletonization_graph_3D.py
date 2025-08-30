import numpy as np
from skan import csr
import networkx as nx
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def visualize_node_neighborhood(G, node_id, radius=2):

    # Obtain a local subgraph centered on node with radius
    sub_nodes = nx.single_source_shortest_path_length(G, node_id, cutoff=radius).keys()
    subG = G.subgraph(sub_nodes)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for u, v in subG.edges():
        p1 = np.array(G.nodes[u]['coord_zyx'])[::-1]  # (x,y,z)
        p2 = np.array(G.nodes[v]['coord_zyx'])[::-1]
        ax.plot(*zip(p1, p2), color='blue', alpha=0.8)

    ax.set_title(f"Neighborhood of node {node_id}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def print_adjacency_info(G, node_id):
    print(f"Central node {node_id} degree: {G.degree[node_id]}")
    print("Neighbor node list:")
    for nbr in G.neighbors(node_id):
        print(f" - {nbr}, degree = {G.degree[nbr]}")



def compress_degree_with_angle(G, length_attr='length', coord_attr='coord_zyx', angle_exclude_range=(85, 95), degree_set={2}):

    def angle_between_edges(G, node):
        if G.degree(node) != 2:
            return None
        nbrs = list(G.neighbors(node))
        p0 = np.array(G.nodes[node][coord_attr], dtype=float)
        p1 = np.array(G.nodes[nbrs[0]][coord_attr], dtype=float)
        p2 = np.array(G.nodes[nbrs[1]][coord_attr], dtype=float)
        v1, v2 = p1 - p0, p2 - p0
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(theta)

    G = G.copy()
    anchor = set()
    for n in G.nodes:
        d = G.degree(n)
        if d not in degree_set:
            anchor.add(n)
        else:
            # Perform angle judgment only for nodes degree ∈ degree_set
            theta = angle_between_edges(G, n) if d == 2 else None
            if theta is None or (angle_exclude_range[0] <= theta <= angle_exclude_range[1]):
                anchor.add(n)

    H = nx.Graph()
    mapping = {}

    for n in anchor:
        H.add_node(n, **G.nodes[n])

    visited_edges = set()
    for a in anchor:
        for nbr in list(G.neighbors(a)):
            e = tuple(sorted((a, nbr)))
            if e in visited_edges:
                continue

            path_nodes = [a]
            total_len = 0.0
            coords_list = [G.nodes[a][coord_attr]]

            prev, curr = a, nbr
            while True:
                visited_edges.add(tuple(sorted((prev, curr))))
                path_nodes.append(curr)
                edge_len = G.edges[prev, curr].get(length_attr, 1.0)
                total_len += edge_len
                coords_list.append(G.nodes[curr][coord_attr])

                if curr in anchor:
                    break
                nxts = [x for x in G.neighbors(curr) if x != prev]
                if len(nxts) != 1:
                    break
                prev, curr = curr, nxts[0]

            u, v = path_nodes[0], path_nodes[-1]
            if u == v:
                continue

            H.add_edge(u, v,
                       length=total_len,
                       path=path_nodes,
                       coords=coords_list)
            for mid in path_nodes[1:-1]:
                mapping[mid] = None

    for n in anchor:
        mapping[n] = n

    return H, mapping

def prune_skeleton(G, min_path_length=5, max_iter=100):

    G = G.copy()
    for _ in range(max_iter):
        to_remove = []

        for node in G.nodes:
            if G.degree(node) != 1:
                continue

            path = [node]
            curr = node
            prev = None

            while True:
                neighbors = list(G.neighbors(curr))
                next_nodes = [n for n in neighbors if n != prev]
                if len(next_nodes) != 1:
                    break

                next_node = next_nodes[0]
                path.append(next_node)
                prev, curr = curr, next_node

                if G.degree(curr) != 2:
                    break

            total_len = 0.0
            for u, v in zip(path[:-1], path[1:]):
                total_len += G.edges[u, v].get('length', 1.0)

            if total_len < min_path_length:
                to_remove.extend(path[:-1])

        if not to_remove:
            break

        G.remove_nodes_from(set(to_remove))

    return G

def plot_graph_3d_lines(H, coord_attr='coords', linewidth=0.8, alpha=0.8):
    segments = []
    for u, v, data in H.edges(data=True):
        coords_zyx = data[coord_attr]  # list of (z,y,x)
        if len(coords_zyx) < 2:
            continue
        coords_xyz = [(x, y, z) for z, y, x in coords_zyx]
        segments.append(coords_xyz)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    line_collection = Line3DCollection(segments, colors =((0.1, 0.5, 0.4)) , linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(line_collection)

    all_coords = np.concatenate(segments)
    X, Y, Z = all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]

    max_range = np.array([X.max() - X.min(),
                          Y.max() - Y.min(),
                          Z.max() - Z.min()]).max() / 2.0

    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("Degree Simplified 3D Graph")

    plt.tight_layout()
    plt.show()


def plot_edge_length_distribution(G, title="", bins=50):
    edge_lengths = [data['length'] for _, _, data in G.edges(data=True)]

    if not edge_lengths:
        print("Graph has no edges.")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(edge_lengths, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Edge Length")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    # Load the 3D binary mask
    vol = np.load('file_name')
    mask = vol > 0

    # Skeletonise in 3D
    skeleton = skeletonize(mask)

    # Extract voxel coordinates of the skeleton
    coords = np.column_stack(np.nonzero(skeleton))
    n_points = coords.shape[0]

    # Down‑sample for plotting if too dense
    max_points = 10000000000000000
    if n_points > max_points:
        idx = np.random.choice(n_points, size=max_points, replace=False)
        coords_plot = coords[idx]
    else:
        coords_plot = coords

    point_size = 1
    point_alpha = 0.9


    # Plot a 3‑D scatter of the skeleton voxels
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords_plot[:, 2], coords_plot[:, 1], coords_plot[:, 0], s=point_size, marker='.', edgecolors='none',
               alpha=point_alpha)  # note axis order (z,y,x)→(x,y,z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_title('label_tr_LI-2016-03-04-emb5-pos3_tp70_.npy')

    # Improve aspect ratio
    max_range = np.array([coords_plot[:, 2].max() - coords_plot[:, 2].min(),
                          coords_plot[:, 1].max() - coords_plot[:, 1].min(),
                          coords_plot[:, 0].max() - coords_plot[:, 0].min()]).max() / 2.0

    mid_x = (coords_plot[:, 2].max() + coords_plot[:, 2].min()) * 0.5
    mid_y = (coords_plot[:, 1].max() + coords_plot[:, 1].min()) * 0.5
    mid_z = (coords_plot[:, 0].max() + coords_plot[:, 0].min()) * 0.5

    scale = 1
    ax.set_xlim(mid_x - max_range * scale, mid_x + max_range * scale)
    ax.set_ylim(mid_y - max_range * scale, mid_y + max_range * scale)
    ax.set_zlim(mid_z - max_range * scale, mid_z + max_range * scale)

    plt.show()




    # Extract graph
    spacing = (1,1,1)  # set voxel
    sk = csr.Skeleton(skeleton.astype(np.uint8), spacing=spacing)
    A = sk.graph
    coords = sk.coordinates
    deg = sk.degrees


    # Build graph
    G_full = nx.from_scipy_sparse_array(A)

    for i, (z, y, x) in enumerate(coords):
        G_full.nodes[i]['coord_zyx'] = (z, y, x)
        G_full.nodes[i]['degree'] = int(deg[i])

    spacing_arr = np.array(spacing, dtype=float)

    for u, v in G_full.edges():
        vec = (coords[u] - coords[v]) * spacing_arr
        G_full.edges[u, v]['length'] = float(np.linalg.norm(vec))

    print(G_full.number_of_nodes(), G_full.number_of_edges())
    print('max degree:', max(dict(G_full.degree()).values()))
    edge_lengths = [data['length'] for _, _, data in G_full.edges(data=True)]
    if edge_lengths:
        print("min edge length:", min(edge_lengths))
        print("max edge length:", max(edge_lengths))
    else:
        print("Graph has no edges.")



    # Degree and angle method to simplification
    H, node_map = compress_degree_with_angle(G_full,angle_exclude_range=(85, 95), degree_set={2})
    print("Number of nodes before compression：", G_full.number_of_nodes())
    print("Number of nodes after compression：", H.number_of_nodes())
    print("Compressed number of digits：", H.number_of_edges())
    edge_lengths_H = [data['length'] for _, _, data in H.edges(data=True)]
    if edge_lengths_H:
        print("After compression:")
        print("  min edge length:", min(edge_lengths_H))
        print("  max edge length:", max(edge_lengths_H))
    else:
        print("Compressed graph has no edges.")
    plot_graph_3d_lines(H)

    #plot_edge_length_distribution(G_full, title="Original Graph Edge Length Distribution")
    #plot_edge_length_distribution(H, title="Compressed Graph Edge Length Distribution")
    plot_edge_length_distribution(G_full)
    plot_edge_length_distribution(H)

    # Pruning
    H_pruned = prune_skeleton(H, min_path_length=5)
    plot_graph_3d_lines(H_pruned)


