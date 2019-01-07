import helper
import compute


def main(dataset_number=4, sparseness=0.0, image_size=500, padding=5, mode=1):
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
        cluster_centers, labels = compute.k_means(em_2d, k_default=None)
        labels_orig = labels
    else:
        cluster_centers, labels, labels_orig = compute.hdbscan(em_2d)

    # Representative images
    silhouettes = compute.get_silhouettes(em_2d, labels)
    representative = compute.get_representative(em_2d, cluster_centers, labels, silhouettes, mode)

    # Sizes and positions of the images
    ratios = helper.get_image_size_ratios(imgs)
    sizes = compute.get_sizes(image_size, em_2d, ratios, cluster_centers, labels, representative)
    positions = compute.get_positions(em_2d, image_size)

    # helper.plot_clusters(em_2d, cluster_centers, cluster_labels_orig, representative)

    # Expand as long as overlaps occur - gradually increase space between images
    iters = 0
    while compute.overlap(positions, sizes, padding):
        positions *= 1.05
        iters += 1
    print('overlap resolved in {} iterations'.format(iters))

    dists1 = compute.get_distances(positions)

    # Overlapping resolved, now "shrink" towards representative images
    positions = compute.shrink_intra(positions, sizes, representative, labels, padding)
    dists2 = compute.get_distances(positions)

    # Move clusters closer together by same factor
    positions = compute.shrink_inter1(positions, sizes, representative, labels, padding)
    dists3 = compute.get_distances(positions)

    # Move clusters closer together separately by different factors
    positions = compute.shrink_inter2(positions, sizes, representative, labels, padding)
    dists3 = compute.get_distances(positions)

    # Further shrink (move images that are closer to center first)
    # positions = compute.shrink_with_sparseness_center(positions, sizes, sparseness, representative, labels, padding)

    # Shrink by x and y separately
    positions = compute.shrink_xy(positions, sizes, representative, labels, padding)

    # "Shake" images with small offsets
    for _ in range(10):
        positions = compute.shrink_with_shaking(positions, sizes, padding)

    # Shrink to finalize positions
    positions = compute.shrink_xy(positions, sizes, representative, labels, padding)
    positions = compute.shrink_xy(positions, sizes, representative, labels, padding, smaller=True)
    positions = compute.shrink_inter2(positions, sizes, representative, labels, padding)

    helper.plot(imgs, positions, sizes)

    dists4 = compute.get_distances(positions)

    print()
    print('dists 2 score:', compute.compare_distances(dists2, dists1))
    print('dists 3 score:', compute.compare_distances(dists3, dists1))
    print('dists 4 score:', compute.compare_distances(dists4, dists1))


if __name__ == '__main__':
    main()
