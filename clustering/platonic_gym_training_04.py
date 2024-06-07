"""
dbscan training
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import collections
from enum import Enum
import numpy as np

X, _ = make_blobs(n_samples=16,
                  n_features=2,
                  centers=2,
                  cluster_std=0.3,
                  random_state=0x101)

fig = plt.figure(figsize=(16, 16))

# hyperparameters
eps = 0.5
samples = 4

processed_points = []
noise_points = []
border_points = []
core_points = []
clusters = collections.defaultdict(list)


def compute_neighbors(p, X, eps):
    ns = []
    for i, q in enumerate(X):
        d = np.linalg.norm(p - q, 2)
        if d < eps:
            ns.append(i)
    return ns


def assign_to_cluster(ni, cluster):
    if ni in processed_points:
        return
    if ni in cluster:
        return
    processed_points.append(ni)
    cluster.append(ni)

    nis = compute_neighbors(X[ni], X, eps)
    if samples <= len(nis):
        core_points.append(ni)
        for i in nis:
            if i not in processed_points:
                assign_to_cluster(i, cluster)
    else:
        border_points.append(ni)


for i, p in enumerate(X):
    # p was already processed
    if i in processed_points:
        continue

    processed_points.append(i)
    p_neighbors = compute_neighbors(p, X, eps)
    # p is a core point, form new cluster
    if samples <= len(p_neighbors):
        core_points.append(i)
        cluster_ind = len(clusters)
        clusters[cluster_ind].append(i)
        # new all points in neighbors are either core points or border points
        # and all of them have to be assigned to the new cluster
        real_nis = [pi for pi in p_neighbors if pi != i]
        for ni in real_nis:
            assign_to_cluster(ni, clusters[cluster_ind])

    else:
        noise_points.append(i)

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(clusters))]
# colors = ['cyan', 'purple', 'pink', 'green']
ax = fig.add_subplot(111)
ax.scatter(X[noise_points, 0], X[noise_points, 1], color='black')
for point in X[noise_points]:
    circle = plt.Circle(point, eps, color='black', fill=False, linestyle='--')
    plt.gca().add_patch(circle)
for point in X[core_points]:
    circle = plt.Circle(point, eps, color='red', fill=False)
    plt.gca().add_patch(circle)
for point in X[border_points]:
    circle = plt.Circle(point, eps, color='green', fill=False)
    plt.gca().add_patch(circle)

for i, lst in clusters.items():
    print(f'cluster = {i}, points: {lst}')
    ax.scatter(X[lst, 0], X[lst, 1], color=colors[i])

plt.show()


