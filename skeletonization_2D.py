import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from scipy.ndimage import binary_closing
from skan import Skeleton
import os

vol = np.load('file_name')
mask = vol.astype(bool)
Z, Y, X = mask.shape
print(f"Layer number: {Z}")


out_dir = 'file_path'
os.makedirs(out_dir, exist_ok=True)

# Parameter
hole_area_threshold = 4
small_object_size = 60

# Iterate through each layer and plot
for z in range(Z):
    slice_2d = mask[z]
    if not np.any(slice_2d):
        continue

    # Closing operation + filling small holes
    cleaned = binary_closing(slice_2d, structure=np.ones((5,5)))
    cleaned = remove_small_holes(cleaned, area_threshold=hole_area_threshold)
    cleaned = remove_small_objects(cleaned, min_size=small_object_size)

    # Skeleton
    # skeleton_2d = skeletonize(slice_2d)
    skeleton_2d = skeletonize(cleaned)
    skel = Skeleton(skeleton_2d)
    coords = skel.coordinates
    paths = skel.paths_list()

    # Plot
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(slice_2d, cmap='gray')
    for path in paths:
        pts = coords[path]
        ax.plot(pts[:, 1], pts[:, 0], c='red', linewidth= 0.5)

    ax.set_title(f'Layer Z = {z}')
    ax.axis('off')
    plt.savefig(f"{out_dir}/layer_{z:03d}.png", dpi=300)
    plt.close()

print(f"Cleaned skeleton images saved to: {out_dir}")