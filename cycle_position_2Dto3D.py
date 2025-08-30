import numpy as np
import networkx as nx
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from scipy.ndimage import binary_closing
import matplotlib.pyplot as plt
from matplotlib import cm

# Extract cycle and its information
def extract_cycles_information(vol, hole_area_threshold=4, small_object_size=60,
                                  nodes_number=4, min_cycle_length=0):
    mask = vol.astype(bool)
    Z, Y, X = mask.shape

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]

    cycles_info = []

    for z in range(Z):
        slice_2d = mask[z]
        if not np.any(slice_2d):
            continue

        # Preprocessing
        cleaned = binary_closing(slice_2d, structure=np.ones((5,5)))
        cleaned = remove_small_holes(cleaned, area_threshold=hole_area_threshold)
        cleaned = remove_small_objects(cleaned, min_size=small_object_size)

        # Skeletonization
        skeleton_2d = skeletonize(cleaned)
        coords = np.transpose(np.nonzero(skeleton_2d))

        # Build graph
        G = nx.Graph()
        for y, x in coords:
            G.add_node((y, x))
            for dy, dx in neighbors:
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < Y and 0 <= nx_ < X and skeleton_2d[ny, nx_]:
                    G.add_edge((y, x), (ny, nx_))

        # Extract cycles
        cycles = nx.cycle_basis(G)
        for cycle in cycles:
            if len(cycle) < nodes_number:
                continue

            points = np.array(cycle)
            dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
            length = np.max(dists)

            if length < min_cycle_length:
                continue

            # Calculaten medium point
            center_yx = points.mean(axis=0)
            cycles_info.append({
                'z': z,
                'y': center_yx[0],
                'x': center_yx[1],
                'length': length
            })

    return cycles_info



file = "file_name"
print(f"Processing {file} ...")
vol = np.load(file)
cycles_info = extract_cycles_information(vol)

print(f"Got {len(cycles_info)} cycles")


if len(cycles_info) > 0:
    xs = [c['x'] for c in cycles_info]
    ys = [c['y'] for c in cycles_info]
    zs = [c['z'] for c in cycles_info]
    lengths = [c['length'] for c in cycles_info]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(xs, ys, zs, c=lengths, cmap='viridis',
                    vmin=min(lengths), vmax=max(lengths))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Spatial Distribution of Cycles")

    cbar = plt.colorbar(sc, ax=ax, pad=0.07)
    cbar.set_label("Cycle length (diameter)")

    plt.show()
else:
    print("No cycles found")