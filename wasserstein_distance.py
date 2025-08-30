import numpy as np
import networkx as nx
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from scipy.ndimage import binary_closing
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns

# Extract cycle and its information
def extract_cycle_information(vol, hole_area_threshold=4, small_object_size=60,
                          nodes_number=4, min_cycle_length=0):
    mask = vol.astype(bool)
    Z, Y, X = mask.shape

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]

    cycle_lengths = []

    for z in range(Z):
        slice_2d = mask[z]
        if not np.any(slice_2d):
            continue

        # Preprocessing
        cleaned = binary_closing(slice_2d, structure=np.ones((5,5)))
        cleaned = remove_small_holes(cleaned, area_threshold=hole_area_threshold)
        cleaned = remove_small_objects(cleaned, min_size=small_object_size)

        # Skeletonize
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

            cycle_lengths.append(length)

    return cycle_lengths



files = [
    "file_name1",
    "file_name2",
    "file_name3",
    "file_name4",
    "file_name5",
    "file_name6"
]

all_groups = []

for f in files:
    print(f"Processing {f} ...")
    vol = np.load(f)
    lengths = extract_cycle_information(vol)
    all_groups.append(lengths)
    print(f"Got {len(lengths)} cycles")

# Wasserstein distance
n = len(all_groups)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = wasserstein_distance(all_groups[i], all_groups[j])

print("\nWasserstein distance matrix:")
print(dist_matrix)

plt.figure(figsize=(6,5))
sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="Reds",
            xticklabels=[f"G{i+1}" for i in range(n)],
            yticklabels=[f"G{i+1}" for i in range(n)])
# plt.title("Wasserstein distance between cycle length distributions")
plt.show()