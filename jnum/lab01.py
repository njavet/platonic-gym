
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def throw(v0, alpha, g=9.81):
    return (v0**2) * np.sin(2*alpha) / g


[v0, alpha] = np.meshgrid(np.linspace(0, 100, endpoint=True),
                          np.linspace(0, np.pi / 2))

t = throw(v0, alpha)

fig = plt.figure(figsize=(20, 20))
colors = plt.cm.Spectral(np.linspace(0, 1, len(t)))
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.set_title('Wireframe plot')
ax.set_xlabel('v0')
ax.set_ylabel('Alpha')
ax.set_zlabel('Throw Distance')
ax.plot_wireframe(v0, alpha, t)

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.set_title('Surface plot')
ax.set_xlabel('v0')
ax.set_ylabel('Alpha')
ax.set_zlabel('Throw Distance')
ax.plot_surface(v0, alpha, t, cmap=mpl.cm.coolwarm)

ax = fig.add_subplot(2, 2, 3)
ax.contour(v0, alpha, t)


def wave0(x, t):
    return np.sin(x + t)


def wave1(x, t):
    return np.sin(x + t) + np.cos(2*x + 2*t)


[xx, tt] = np.meshgrid(np.linspace(0, 32), np.linspace(0, 16))
w0 = wave0(xx, tt)
ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot_wireframe(xx, tt, w0)

plt.show()

