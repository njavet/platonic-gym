"""
dbscan training
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import collections
import numpy as np

X, _ = make_blobs(n_samples=16,
                  n_features=2,
                  centers=2,
                  cluster_std=1,
                  random_state=0x101)

fig = plt.figure(figsize=(32, 16))

# hyperparameters
eps = 2
samples = 4

processed_points = []
noise_points = []
border_points = []
core_points = []


def compute_neighbors(p, X, eps):
    p_neighbors = []
    for j, q in enumerate(X):
        d = np.linalg.norm(p - q, 2)
        if d < eps:
            p_neighbors.append(j)
    return p_neighbors


dists = collections.defaultdict(list)
clusters = collections.defaultdict(list)

for i, p in enumerate(X):
    # p was already processed
    if i in processed_points:
        continue

    processed_points.append(i)
    p_neighbors = compute_neighbors(p, X, eps)
    # p is a core point, form new cluster
    print('n', p_neighbors)
    if samples <= len(p_neighbors):
        core_points.append(i)
        cluster_ind = len(clusters)
        clusters[cluster_ind].append(i)
        
        # to_proc consists of all neighbors of p without p
        # they all have to be assigned to the new cluster and be
        # processed
        to_proc = [pi for pi in p_neighbors if pi != i]
        while to_proc:
            pind = to_proc.pop()
            clusters[cluster_ind].append(pind)
            processed_points.append(pind)
            neighbors = compute_neighbors(X[pind], X, eps)
            if samples <= len(neighbors):
                core_points.append(pind)
                for ni in neighbors:
                    if ni != pind and ni not in to_proc and ni not in
                    clusters[cluster_ind]:
                        to_proc.append(ni)
            else:
                border_points.append(pind)
    
    else:
        noise_points.append(i)
    
print('noise:', noise_points)
print('border:', border_points)
print('core:', core_points)


for i, lst in clusters.items():
    ax = plt.add_subplot(2, 4, i+1)
    ax.scatter(X[lst, 0], X[lst, 1])
plt.show()


