import helper
import compute


def main(dataset_number=5, image_size=500, padding=5, n_clusters=None):
    # Load images and get embeddings from NN
    imgs = helper.get_images(dataset_number)
    embeddings = helper.get_embeddings(dataset_number, imgs)
    print('loaded {} images'.format(len(imgs)))

    # Compute 2D embeddings with MDS
    em_2d = compute.mds(embeddings, init=compute.pca(embeddings))

    # Perform clustering
    cluster_centers, labels = compute.k_means(em_2d, k_default=n_clusters)
    print('clusters:', len(cluster_centers))
    print('sizes of clusters: ', end='')
    for l in range(max(labels) + 1):
        print(sum(labels == l), end=', ')
    print()

    # Representative images
    silhouettes = compute.get_silhouettes(em_2d, labels)
    representative = compute.get_representative(em_2d, cluster_centers, labels, silhouettes)

    # Sizes and positions of the images
    ratios = helper.get_image_size_ratios(imgs)
    sizes = compute.get_sizes(image_size, em_2d, ratios, cluster_centers, labels, representative)
    positions = compute.get_positions(em_2d, image_size)

    # Expand as long as overlaps occur - gradually increase space between images
    iters = 0
    while compute.overlap(positions, sizes, padding):
        positions *= 1.05
        iters += 1
    print('overlap resolved in {} iterations'.format(iters))

    dists = [compute.get_distances(positions)]

    # Overlapping resolved, now "shrink" towards representative images
    positions = compute.shrink_intra(positions, sizes, representative, labels, padding)
    dists.append(compute.get_distances(positions))

    # Move clusters closer together by same factor
    positions = compute.shrink_inter1(positions, sizes, representative, labels, padding)
    dists.append(compute.get_distances(positions))

    # Move clusters closer together separately by different factors
    positions = compute.shrink_inter2(positions, sizes, representative, labels, padding)
    dists.append(compute.get_distances(positions))

    # Shrink by x and y separately
    positions = compute.shrink_xy(positions, sizes, representative, labels, padding)
    dists.append(compute.get_distances(positions))

    # "Shake" images with small offsets
    for _ in range(10):
        positions = compute.shrink_with_shaking(positions, sizes, padding)
    dists.append(compute.get_distances(positions))

    # Shrink to finalize positions
    positions = compute.shrink_xy(positions, sizes, representative, labels, padding)
    dists.append(compute.get_distances(positions))
    positions = compute.shrink_xy(positions, sizes, representative, labels, padding, smaller=True)
    dists.append(compute.get_distances(positions))
    positions = compute.shrink_inter2(positions, sizes, representative, labels, padding)
    dists.append(compute.get_distances(positions))

    im = helper.plot(imgs, positions, sizes)
    im.save('../img.png')
    # helper.plot_clusters(em_2d, cluster_centers, labels, representative)

    scores = list(map(lambda d: compute.compare_distances(dists[0], d), dists))

    print('\nscores:')
    for i, s in enumerate(scores[1:]):
        print('{:.3f},'.format(s), end=' ')


if __name__ == '__main__':
    main()
