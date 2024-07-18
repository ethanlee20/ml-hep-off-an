
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ml_hep_off_an_lib.plot import setup_mpl_params

setup_mpl_params()

image = np.load("../datafiles/dc9_-0.08_1_img_det.npy", allow_pickle=True)
# edges = np.load("../datafiles/dc9_-0.08_1_edges_det.npy", allow_pickle=True)

cmap = plt.cm.magma
norm = Normalize()

alpha = 1

colors = cmap(norm(image), alpha=alpha)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.voxels(image, facecolors=colors)

fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="vertical", label="Count", alpha=1)

plt.savefig(f"../plots/image.png", bbox_inches='tight')
plt.close(fig)

# ax.set_title(r"Average $q^2$ per Angular Bin")
# ax.set_xlabel(r"$\cos\theta_\mu$ Bin", labelpad=10)
# ax.set_ylabel(r"$\cos\theta_K$ Bin", labelpad=10)
# ax.set_zlabel(r"$\chi$ Bin", labelpad=10)
# ax.tick_params(pad=1)

# fig.colorbar(
#     sc,
#     ax=ax,
#     pad=0.025,
#     shrink=0.6,
#     location="left",
# )
