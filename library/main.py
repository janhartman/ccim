import helper
import compute


def main(k=3, sparseness=0.2, dataset_number=2, padding=2):
    # Load images and get embeddings from NN
    imgs = helper.get_images(dataset_number)
    embeddings = helper.get_embeddings(dataset_number, imgs)
    print('loaded {} images'.format(len(imgs)))

    # Compute 2D embeddings with MDS / UMAP
    # em_2d = compute.mds(embeddings)
    em_2d = compute.umap(embeddings)

    # Perform clustering
    cluster_centers, cluster_labels = compute.k_means(em_2d, k)
    cluster_labels_orig = cluster_labels
    # cluster_centers, cluster_labels, cluster_labels_orig = compute.hdbscan(em_2d)

    # Representative images
    representative = compute.get_representative(em_2d, cluster_centers, cluster_labels, None)

    # Sizes and positions of the images
    image_size, ratios = helper.get_image_size_ratios(imgs)
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
