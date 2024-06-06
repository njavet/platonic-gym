import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl



def throw(v0, alpha, g=9.81):
    return v0**2 * np.sin(2*alpha) / g


def main():
    # 3D plotting setup
    fig = plt.figure(figsize=(20, 10))
    ax0 = fig.add_subplot(1, 4, 1, projection='3d')
    ax1 = fig.add_subplot(1, 4, 2, projection='3d')
    ax2 = fig.add_subplot(1, 4, 3, projection='3d')
    ax3 = fig.add_subplot(1, 4, 4)
    v0 = np.linspace(0, 100)
    alpha = np.linspace(0, np.pi / 2)
    # coordinate grid
    xv, yv = np.meshgrid(v0, alpha)
    w = throw(xv, yv)

    ax0.plot_wireframe(xv, yv, w)
    ax1.plot_surface(xv, yv, w, vmin=w.min(), cmap=cm.Blues)
    ax2.contour(xv, yv, w)
    ax3.contour(xv, yv, w)

    plt.show()


if __name__ == '__main__':
    main()



