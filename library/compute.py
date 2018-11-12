from copy import copy

import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples


def pca(embeddings):
    p = PCA(n_components=2)
    return p.fit_transform(embeddings)


def mds(embeddings, init=None):
    c = pdist(embeddings, metric='cosine')
    m = MDS(n_components=2, n_init=(1 if init is not None else 5), dissimilarity='precomputed')
    return m.fit_transform(squareform(c), init=init)


def umap(embeddings):
    return UMAP().fit_transform(embeddings)


# TODO distance - cosine?
def k_means(em_2d, k):
    km = KMeans(n_clusters=k).fit(em_2d)
    return km.cluster_centers_, km.labels_


def silhouette(em_2d, labels):
    return silhouette_samples(em_2d, labels)


# normalize by span
def normalize(arr):
    min_arr = np.min(arr, axis=0)
    span = np.max(arr, axis=0) - min_arr
    return (arr - min_arr) / span


# TODO use representative images?
def get_sizes(s, image_size):
    n = normalize(s)
    return np.clip(n * image_size, image_size / 5, None)


def get_positions(em_2d, canvas_size):
    return normalize(em_2d) * canvas_size


# choose either the closest to the center or largest silhouettes
def get_representative(em_2d, centers, labels, silhouettes):
    # closest to the centers
    # rep = [np.linalg.norm(em_2d - center, axis=1).argmin() for center in centers]

    # largest silhouette
    rep = [int(i[0]) for i in[np.where(silhouettes == np.max(silhouettes[np.where(labels == label)])) for label in range(len(centers))]]
    return rep


def corners(p, s):
    x, y = p
    return [(x, y), (x + s, y), (x, y + s), (x + s, y + s)]


def check_corners(c2, x1, y1, s1):
    for x2, y2 in c2:
        if x1 <= x2 <= x1 + s1 and y1 <= y2 <= y1 + s1:
            return True


def overlap(positions, sizes):
    for i, ps in enumerate(zip(positions, sizes)):
        p1, s1 = ps
        corners1 = corners(p1, s1)
        for j in range(i):
            p2, s2 = positions[j], sizes[j]
            corners2 = corners(p2, s2)
            if check_corners(corners1, *p2, s2) or check_corners(corners2, *p1, s1):
                return True

    return False


# Intra-cluster shrinking
# For each image, set it as close to the representative for that cluster as possible without overlapping
def shrink_intra(positions, sizes, representative, labels, image_size):
    for i, pos in enumerate(positions):
        rep_idx = representative[labels[i]]
        rep_pos = positions[rep_idx]

        if i == rep_idx:
            continue

        cluster_indices = np.where(labels == labels[i])
        new_positions = copy(positions)
        new_pos = pos

        for alpha in np.linspace(0.95, 0.0, 20):
            new_pos = pos - alpha * (pos - rep_pos)
            new_positions[i] = new_pos

            if not overlap(new_positions[cluster_indices], sizes[cluster_indices]):
                print(i, alpha)
                break

        positions[i] = new_pos

    return positions


# Inter-cluster shrinking
# For each cluster, move it to closer to the center of all images
def shrink_inter(positions, sizes, representative, labels, image_size):
    mean = np.mean(positions, axis=0)

    for i, rep_idx in enumerate(representative):
        rep_pos = positions[rep_idx]
        cluster_indices = np.where(labels == i)
        pos = positions[cluster_indices]
        pos -= 0.65 * (rep_pos - mean)
        positions[cluster_indices] = pos

    positions -= np.min(positions, axis=0)
    canvas_size = np.max(positions) + image_size

    return positions, canvas_size
