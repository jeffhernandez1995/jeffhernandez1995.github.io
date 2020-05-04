import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x = np.linspace(0,1,50)
y = np.linspace(0,1,50)
z = np.linspace(0,1,50)

Z = np.outer(z.T, z)        # 50x50
X, Y = np.meshgrid(x, y)    # 50x50

color_dimension = 0.162360 + 0.059561*X + 0.3884*Y + 0.227154*Z + \
    0.069101*X*Y + 0.102860*X*Z - 0.021391*Y*Z -0.05879*X*Y*Z
color_dimension /= 2
minn, maxx = color_dimension.min(), color_dimension.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
m.set_array([])
fcolors = m.to_rgba(color_dimension)

# plot
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
ax.set_xlabel('LN')
ax.set_ylabel('DE')
ax.set_zlabel('HS')
plt.show()