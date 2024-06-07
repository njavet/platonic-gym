"""
    kmeans++ training

"""
import numpy as np
from sklearn.datasets import make_blobs
import collections
import matplotlib.pyplot as plt


X, _ = make_blobs(n_samples=16,
                  n_features=2,
                  cluster_std=0.8,
                  centers=2,
                  random_state=0x101)

K = 2
fig = plt.figure(figsize=(32, 16))

colors = ['cyan', 'purple', 'orange', 'green']
# centroids initialization
centroids = []
index = np.random.choice(len(X))
centroids = [X[index]]
for centroid_ind in range(K-1):
    prev_centroid = centroids[-1]
    distances = []
    for i, data_point in enumerate(X):
        dist = np.linalg.norm(data_point - prev_centroid, 2)
        distances.append((dist, i))
    _, p_ind = sorted(distances, reverse=True)[0]
    centroids.append(X[p_ind])
centroids = np.array(centroids)


for n_iter in range(8):
    ax = fig.add_subplot(2, 4, n_iter + 1)
    ax.grid()
    ax.scatter(centroids[:, 0], 
               centroids[:, 1], 
               c='red', 
               label='centroids',
               zorder=5)

    c2p = collections.defaultdict(list)
    for i, p in enumerate(X):

        dists = []
        for c_ind, c in enumerate(centroids):
            d = np.linalg.norm(p - c, 2)
            dists.append((d, c_ind))
        _, ci = sorted(dists)[0]
        c2p[ci].append(i)

    for ci, pis in c2p.items():
        data_points = X[pis]
        ax.scatter(data_points[:, 0], 
                   data_points[:, 1], 
                   color=colors[ci],
                   label=str(ci))
        centroids[ci] = np.mean(data_points, axis=0)


plt.legend()
plt.show()
