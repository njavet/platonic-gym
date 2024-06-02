import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def task1(random_state=0x101):
    X, _ = make_blobs(n_samples=300,
                      centers=4,
                      cluster_std=0.6,
                      random_state=random_state)

    km = KMeans(n_clusters=4, random_state=random_state)
    km.fit(X)
    cluster_labels = km.predict(X)

    def silhouette_score_method():
        scores = []
        for k in range(2, 10):
            _km = KMeans(n_clusters=k)
            labels = _km.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append(score)
        plt.plot(np.array(range(2, 10)), scores, marker='x')

    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, s=50)
    centers = km.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='black', s=200)
    plt.show()


def kmeans_moons(random_state=0x101):
    X, _ = make_moons(n_samples=202,
                      noise=0.05,
                      random_state=random_state)
    km = KMeans(n_clusters=2)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='black', s=200)
    plt.show()


def dbscan_moons(random_state=0x101):
    X, _ = make_moons(n_samples=202,
                      noise=0.05,
                      random_state=random_state)
    ds = DBSCAN(eps=0.25, min_samples=5)
    labels = ds.fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50)
    plt.show()


if __name__ == '__main__':
    dbscan_moons()
