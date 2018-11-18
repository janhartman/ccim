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


def k_means(em_2d, k):
    km = KMeans(n_clusters=k).fit(em_2d)
    return km.cluster_centers_, km.labels_


def silhouette(em_2d, labels):
    return silhouette_samples(em_2d, labels)


# normalize by span to [0, 1]
def normalize(arr, axis=None):
    minimum = np.min(arr, axis=axis)
    span = np.max(arr, axis=axis) - minimum
    return (arr - minimum) / span


# choose either the closest to the center or largest silhouettes
def get_representative(em_2d, centers, labels, silhouettes):
    # closest to the centers
    rep = [np.linalg.norm(em_2d - center, axis=1).argmin() for center in centers]

    # largest silhouette in cluster
    # rep = [int(i[0]) for i in [np.where(silhouettes == np.max(silhouettes[np.where(labels == label)]))
    #                            for label in range(len(centers))]]
    return rep


# choose either centroid proximity or silhoettes
def get_sizes(image_size, em_2d, centers, labels, representative, silhouettes):
    # cluster center proximity - this only works with representative images being the closest to centers
    sizes = np.zeros(labels.shape)

    for label, center in enumerate(centers):
        cluster_indices = np.where(labels == label)
        cluster = em_2d[cluster_indices]
        norm = 1 / np.linalg.norm(cluster - center, axis=1)
        sizes[cluster_indices] = norm / np.max(norm)

    # decrease sizes above a limit to make sure we only have k big images
    sizes[np.where(sizes > 0.65)] /= 1.25
    sizes[representative] = 1

    # use silhouettes in a straightforward way
    # not the best option because there can be more "big" images per cluster
    # sizes = silhouettes

    sizes **= 2
    sizes = np.clip(sizes * image_size, image_size / 5, None)

    return sizes


def get_positions(em_2d, canvas_size):
    return normalize(em_2d, axis=0) * canvas_size


def get_distances(mat):
    return normalize(squareform(pdist(mat)))


def compare_distances(dists1, dists2):
    d = np.abs(dists1 - dists2)
    print('\n mean, max:', np.mean(d), np.max(d))
    return np.sum(d)


def corners(p, s):
    x, y = p
    return [(x, y), (x + s, y), (x, y + s), (x + s, y + s)]


def check_corners(c2, x1, y1, s1):
    for x2, y2 in c2:
        if x1 <= x2 <= x1 + s1 and y1 <= y2 <= y1 + s1:
            return True
    return False


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
    # make sure we start with biggest
    sort_indices = np.argsort(sizes)[::-1]
    reverse_sort_indices = np.argsort(sort_indices)

    # temporarily sort everything descending by size, unsort positions when returning
    positions = positions[sort_indices]
    sizes = sizes[sort_indices]
    labels = labels[sort_indices]
    representative = reverse_sort_indices[representative]

    for i, pos in enumerate(positions):
        rep_idx = representative[labels[i]]
        rep_pos = positions[rep_idx]

        if i == rep_idx:
            continue

        cluster_indices = np.where(labels == labels[i])
        new_positions = copy(positions)
        new_pos = pos

        for alpha in np.linspace(0.99, 0.0, 100):
            new_pos = pos - alpha * (pos - rep_pos)
            new_positions[i] = new_pos

            if not overlap(new_positions[cluster_indices], sizes[cluster_indices]):
                print('intra', i, alpha)
                break

        positions[i] = new_pos

    return positions[reverse_sort_indices]


# Inter-cluster shrinking
# For each cluster, move it to closer to the center of all images
def shrink_inter(positions, sizes, representative, labels, image_size):
    mean = np.mean(positions, axis=0)
    new_positions = positions

    # Try to move all clusters by the same factor
    for alpha in np.linspace(0.99, 0.0, 100):
        new_positions = copy(positions)

        for label, rep_idx in enumerate(representative):
            rep_pos = positions[rep_idx]
            cluster_indices = np.where(labels == label)
            pos = positions[cluster_indices]
            new_pos = pos - alpha * (rep_pos - mean)
            new_positions[cluster_indices] = new_pos

        if not overlap(new_positions, sizes):
            print('inter #1', alpha)
            break

    positions = new_positions

    # Try to move clusters closer separately
    for label, rep_idx in enumerate(representative):
        rep_pos = positions[rep_idx]
        cluster_indices = np.where(labels == label)
        pos = positions[cluster_indices]

        new_pos = pos
        new_positions = copy(positions)
        for alpha in np.linspace(0.99, 0.0, 100):
            new_pos = pos - alpha * (rep_pos - mean)
            new_positions[cluster_indices] = new_pos

            if not overlap(new_positions, sizes):
                print('inter #2', label, alpha)
                break

        positions[cluster_indices] = new_pos

    positions -= np.min(positions, axis=0) - 10
    canvas_size = np.max(positions) + image_size + 10

    return positions, canvas_size
