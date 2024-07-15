import matplotlib.pyplot as plt
import matplotlib as mpl


cmap=plt.cm.magma
norm=mpl.colors.Normalize(vmin=-1, vmax=1)

print(sorted([1,-5,9,5]))
print(norm(1.0))