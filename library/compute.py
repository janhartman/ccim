from copy import copy

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_samples, silhouette_score


def pca(embeddings):
    p = PCA(n_components=2)
    return p.fit_transform(embeddings)


def mds(embeddings, init=None):
    c = pdist(embeddings, metric='cosine')
    m = MDS(n_components=2, n_init=(1 if init is not None else 5), dissimilarity='precomputed')
    return m.fit_transform(squareform(c), init=init)


def k_means(em_2d, k_default=None):
    if k_default:
        km_best = KMeans(n_clusters=k_default).fit(em_2d)
    else:
        km_all = [KMeans(n_clusters=k).fit(em_2d) for k in range(2, 11)]
        ss = [silhouette_score(em_2d, km.labels_) for km in km_all]
        km_best = km_all[np.argmax(ss)]
    return km_best.cluster_centers_, km_best.labels_


def get_silhouettes(em_2d, labels):
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
    # rep = [int(i[0]) for i in [np.where(silhouettes == np.max(silhouettes[np.where(labels == label)])) for label in range(len(centers))]]

    return rep


# choose either centroid proximity or silhoettes
def get_sizes(image_size, em_2d, ratios, centers, labels, representative):
    # cluster center proximity - this only works with representative images being the closest to centers
    sizes = np.zeros(labels.shape)

    for label, center in enumerate(centers):
        cluster_indices = np.where(labels == label)
        cluster = em_2d[cluster_indices]
        norm = np.linalg.norm(cluster - center, axis=1)
        sizes[cluster_indices] = norm / np.max(norm)

    # decrease sizes above a limit to make sure we only have k big images
    sizes = 1 / sizes
    sizes /= np.max(sizes)
    sizes **= 0.75
    sizes[representative] = 1

    # TODO: do we want the minimum or maximum dimension to be the specified size?
    sizes = np.stack((sizes, sizes / ratios), 1)
    sizes = np.clip(sizes * image_size, image_size / 5, None)

    return sizes


# assume sqrt(n) * sqrt(n) placement - doesn't really matter since it will be resolved with expanding
def get_positions(em_2d, image_size):
    return normalize(em_2d, axis=0) * np.sqrt(len(em_2d)) * image_size


def get_distances(mat):
    return normalize(squareform(pdist(mat)))


def compare_distances(dists1, dists2):
    return np.linalg.norm(dists1 - dists2) / len(dists1)


# indices is an array of indices which have changed - we only need to check these
def overlap(positions, sizes, padding, indices=None):
    other_side_pos = positions + sizes
    pos = np.hstack((positions, other_side_pos))
    for i in range(len(pos)):
        for j in (indices if indices is not None else range(i)):
            if i == j:
                continue

            p1 = pos[i]
            p2 = pos[j]

            if p1[0] > p2[2] + padding or p2[0] > p1[2] + padding:
                continue
            elif p1[1] > p2[3] + padding or p2[1] > p1[3] + padding:
                continue

            return True

    return False


# Intra-cluster shrinking
# For each image, set it as close to the representative for that cluster as possible without overlapping
def shrink_intra(positions, sizes, representative, labels, padding):
    # make sure we start with biggest
    sort_indices = np.argsort(np.linalg.norm(sizes, axis=1))[::-1]
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

            if not overlap(new_positions[cluster_indices], sizes[cluster_indices], padding):
                # print('intra', i, alpha)
                break

        positions[i] = new_pos

    return positions[reverse_sort_indices]


# Inter-cluster shrinking
# For each cluster, move it to closer to the center of all images
def shrink_inter1(positions, sizes, representative, labels, padding):
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

        if not overlap(new_positions, sizes, padding):
            print('inter #1 {:.2f}'.format(alpha))
            break

    return new_positions


def shrink_inter2(positions, sizes, representative, labels, padding):
    mean = np.mean(positions, axis=0)

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

            if not overlap(new_positions, sizes, padding, cluster_indices[0]):
                print('inter #2 {} {:.2f}'.format(label, alpha))
                break

        positions[cluster_indices] = new_pos

    return positions


# Move separately by x and y coordinate
def shrink_xy(positions, sizes, representative, labels, padding, smaller=False):
    for label, rep in enumerate(representative):
        pos_rep = positions[rep]
        sort_indices = np.argsort(np.linalg.norm(positions + sizes / 2 - pos_rep, axis=1))

        # go from nearer to farther images
        for i in sort_indices:
            if labels[i] != label:
                continue

            pos = copy(positions[i])

            for alpha in np.linspace(1.0, 0.0, 101):
                vec = - alpha * (pos - pos_rep)
                positions[i] = pos + vec

                if not overlap(positions, sizes, padding, [i]):
                    break
                else:
                    # check x and y separately
                    # if the smaller coordinate is causing problems, try setting it to 0
                    smaller_idx = np.argmin(np.abs(vec))
                    vec_c = copy(vec)
                    vec_c[smaller_idx] = 0
                    positions[i] = pos + vec_c

                    if not overlap(positions, sizes, padding, [i]):
                        break
                    else:
                        # try setting the bigger coordinate to 0 if it is allowed
                        if smaller:
                            vec_c = copy(vec)
                            vec_c[1 - smaller_idx] = 0
                            positions[i] = pos + vec_c

                            if not overlap(positions, sizes, padding, [i]):
                                break
                            else:
                                positions[i] = pos
                        else:
                            positions[i] = pos

    return positions


def shrink_with_shaking(positions, sizes, padding):
    for i, pos in enumerate(positions):
        pos = copy(pos)

        # vector coordinates are in [-1, 1], multiplied by third size of the image
        move = (np.random.random(2) * 2 - 1) * sizes[i] / 3
        positions[i] = pos + move

        if overlap(positions, sizes, padding, [i]):
            positions[i] = pos

    return positions
