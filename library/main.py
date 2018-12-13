import helper
import compute


def main(dataset_number=4, sparseness=0.0, image_size=500, padding=10, mode=0):
    # Load images and get embeddings from NN
    imgs = helper.get_images(dataset_number)
    embeddings = helper.get_embeddings(dataset_number, imgs)
    print('loaded {} images'.format(len(imgs)))

    # Compute 2D embeddings with MDS / UMAP
    if mode:
        em_2d = compute.mds(embeddings, init=compute.pca(embeddings))
    else:
        em_2d = compute.umap(embeddings)

    # Perform clustering
    if mode:
        cluster_centers, cluster_labels = compute.k_means(em_2d, k_default=None)
        cluster_labels_orig = cluster_labels
    else:
        cluster_centers, cluster_labels, cluster_labels_orig = compute.hdbscan(em_2d)

    # Representative images
    silhouettes = compute.get_silhouettes(em_2d, cluster_labels)
    representative = compute.get_representative(em_2d, cluster_centers, cluster_labels, silhouettes, mode)

    # Sizes and positions of the images
    ratios = helper.get_image_size_ratios(imgs)
    sizes = compute.get_sizes(image_size, em_2d, ratios, cluster_centers, cluster_labels, representative)
    positions = compute.get_positions(em_2d, image_size)

    helper.plot_clusters(em_2d, cluster_centers, cluster_labels_orig, representative)

    # helper.plot(imgs, positions, sizes)  # overlap

    # Expand as long as overlaps occur - gradually increase space between images
    iters = 0
    while compute.overlap(positions, sizes, padding):
        positions *= 1.05
        iters += 1
    print('overlap resolved in {} iterations'.format(iters))

    # helper.plot(imgs, positions, sizes)
    dists1 = compute.get_distances(positions)

    # Overlapping resolved, now "shrink" towards representative images
    positions = compute.shrink_intra(positions, sizes, representative, cluster_labels, padding)
    dists2 = compute.get_distances(positions)
    # helper.plot(imgs, positions, sizes)

    # Move clusters closer together by same factor
    positions = compute.shrink_inter1(positions, sizes, representative, cluster_labels, padding)
    dists3 = compute.get_distances(positions)
    # helper.plot(imgs, positions, sizes)

    # Move clusters closer together separately by different factors
    positions = compute.shrink_inter2(positions, sizes, representative, cluster_labels, padding)
    dists3 = compute.get_distances(positions)
    # helper.plot(imgs, positions, sizes)

    # Further shrink (move images that are closer to center first)
    positions = compute.shrink_with_sparseness(positions, sizes, sparseness, padding)
    dists4 = compute.get_distances(positions)
    helper.plot(imgs, positions, sizes)

    print()
    print('dists 2 score:', compute.compare_distances(dists2, dists1))
    print('dists 3 score:', compute.compare_distances(dists3, dists1))
    print('dists 4 score:', compute.compare_distances(dists4, dists1))


if __name__ == '__main__':
    main()
