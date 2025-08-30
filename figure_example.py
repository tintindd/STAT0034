from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

# Invert the horse image
image = invert(data.horse())

# perform skeletonization
skeleton = skeletonize(image)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()

################################################################################################################################################

import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def classify_offset(dx, dy, dz):
    nonzero = (dx != 0) + (dy != 0) + (dz != 0)
    if nonzero == 1:
        return "Face"   # 6
    elif nonzero == 2:
        return "Edge"   # 12
    else:
        return "Vertex" # 8

def make_mask_and_colors(connectivity=26):
    assert connectivity in (6, 18, 26)
    n = 3
    filled = np.zeros((n, n, n), dtype=bool)
    facecolors = np.zeros((n, n, n, 4), dtype=float)

    cmap = {
        "Face":   (0.85, 0.20, 0.20, 0.95),
        "Edge":   (0.30, 0.80, 0.30, 0.95),
        "Vertex": (0.20, 0.35, 0.95, 0.95),
    }

    for ix, iy, iz in itertools.product(range(n), repeat=3):
        dx, dy, dz = ix - 1, iy - 1, iz - 1
        if (dx, dy, dz) == (0, 0, 0):
            filled[ix, iy, iz] = False
            continue

        nonzero = (dx != 0) + (dy != 0) + (dz != 0)
        include = (
            (connectivity == 6  and nonzero == 1) or
            (connectivity == 18 and nonzero <= 2) or
            (connectivity == 26 and nonzero <= 3)
        )

        if include:
            filled[ix, iy, iz] = True
            cat = classify_offset(dx, dy, dz)
            facecolors[ix, iy, iz] = cmap[cat]

    return filled, facecolors

def plot_voxel_neighborhood(connectivity=26, add_labels=False, elev=22, azim=35,
                            out_prefix="voxel_neighborhood"):
    filled, facecolors = make_mask_and_colors(connectivity)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled, facecolors=facecolors, edgecolor="k", linewidth=0.75)

    if add_labels:
        n = 3
        for ix, iy, iz in itertools.product(range(n), repeat=3):
            if not filled[ix, iy, iz]:
                continue
            dx, dy, dz = ix - 1, iy - 1, iz - 1
            x, y, z = ix + 0.5, iy + 0.5, iz + 0.5
            ax.text(x, y, z, f"({dx},{dy},{dz})",
                    fontsize=8, ha="center", va="center")

    ticks = [0.5, 1.5, 2.5]
    labels = ["-1", "0", "1"]
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    ax.set_zticks(ticks); ax.set_zticklabels(labels)
    ax.set_xlabel("Δx"); ax.set_ylabel("Δy"); ax.set_zlabel("Δz")

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)

    legend_elems = [
        Patch(facecolor=(0.85, 0.20, 0.20, 0.95), edgecolor="k", label="Face (6)"),
        Patch(facecolor=(0.30, 0.80, 0.30, 0.95), edgecolor="k", label="Edge (12)"),
        Patch(facecolor=(0.20, 0.35, 0.95, 0.95), edgecolor="k", label="Vertex (8)"),
    ]
    if connectivity == 6:
        legend_elems = legend_elems[:1]
    elif connectivity == 18:
        legend_elems = legend_elems[:2]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    fig.tight_layout()

    png = f"{out_prefix}_{connectivity}conn.png"
    pdf = f"{out_prefix}_{connectivity}conn.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}\nSaved: {pdf}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Render a 3D voxel diagram for 6/18/26-connectivity."
    )
    parser.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26],
                        help="Neighborhood type (default: 26).")
    parser.add_argument("--labels", action="store_true",
                        help="Add (Δx,Δy,Δz) labels at voxel centers.")
    parser.add_argument("--out", dest="out_prefix", default="voxel_neighborhood",
                        help="Output filename prefix (default: voxel_neighborhood).")
    parser.add_argument("--elev", type=float, default=22, help="Elevation angle.")
    parser.add_argument("--azim", type=float, default=35, help="Azimuth angle.")
    args = parser.parse_args()

    plot_voxel_neighborhood(connectivity=args.connectivity,
                            add_labels=args.labels,
                            elev=args.elev, azim=args.azim,
                            out_prefix=args.out_prefix)

if __name__ == "__main__":
    main()