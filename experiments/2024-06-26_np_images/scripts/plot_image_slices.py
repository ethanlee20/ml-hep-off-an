
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ml_hep_off_an_lib.plot import setup_mpl_params

setup_mpl_params()

cmap = plt.cm.magma
norm = Normalize()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

image = np.load("../datafiles/dc9_-0.08_1_img_det.npy", allow_pickle=True)
colors = cmap(norm(image))

num_slices = 3

x_axis = 0
y_axis = 1
z_axis = 2

x_shape = image.shape[x_axis]
y_shape = image.shape[y_axis]
z_shape = image.shape[z_axis]

z_slice_indices = np.linspace(0, z_shape-1, num_slices, dtype=int)
# breakpoint()

slice_centering = 0.5

# Outlines

frame_size = 0
num_pixels = 50

x, y = np.mgrid[0:x_shape:50j, 0:y_shape:50j]

for i in z_slice_indices:
    z = np.full_like(
        x, i + slice_centering - 0.3
    )
    ax.plot_surface(
        x, y, z, 
        rstride=1, cstride=1, 
        shade=False,
        color="#f2f2f2",
        edgecolor="#f2f2f2", 
    )


# Slices

x, y = np.indices(
    (x_shape + 1, y_shape + 1)
)

for i in z_slice_indices:
    z = np.full(
        (x_shape + 1, y_shape + 1), i + slice_centering
    )
    ax.plot_surface(
        x, y, z, 
        rstride=1, cstride=1, 
        facecolors=colors[:,:,i], 
        shade=False
    )



fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="vertical", label="Count", alpha=1)

plt.savefig(f"../plots/image_slices.png", bbox_inches='tight')
plt.close(fig)