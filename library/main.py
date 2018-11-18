import helper
import compute


# TODO sparseness
def main(k=4, image_size=200, canvas_size=500):
    # Load images and get embeddings from NN
    imgs = helper.get_images(1)
    embeddings = helper.get_embeddings(imgs)

    # Compute 2D embeddings with MDS / UMAP
    em_2d = compute.mds(embeddings)
    # em_2d = compute.umap(embeddings)

    # Perform K-means clustering, compute silhouettes
    cluster_centers, cluster_labels = compute.k_means(em_2d, k)
    silhouettes = compute.silhouette(em_2d, cluster_labels)

    # K representative images
    representative = compute.get_representative(em_2d, cluster_centers, cluster_labels, silhouettes)

    # Sizes and positions of the images
    sizes = compute.get_sizes(image_size, em_2d, cluster_centers, cluster_labels, representative, silhouettes)
    positions = compute.get_positions(em_2d, canvas_size - image_size)

    # helper.plot(imgs, positions, sizes, representative, canvas_size)  # overlap
    dists1 = compute.get_distances(positions)

    # Expand as long as overlaps occur
    iters = 0
    while compute.overlap(positions, sizes):
        positions *= 1.05  # gradually increase space between images
        canvas_size *= 1.05
        iters += 1

    print('overlap', iters)

    # helper.plot(imgs, positions, sizes, representative, canvas_size)
    dists2 = compute.get_distances(positions)

    # Overlapping resolved, now "shrink" towards representative images
    positions = compute.shrink_intra(positions, sizes, representative, cluster_labels, image_size)

    dists3 = compute.get_distances(positions)

    # helper.plot(imgs, positions, sizes, representative, canvas_size)

    # Move clusters closer together
    positions, canvas_size = compute.shrink_inter(positions, sizes, representative, cluster_labels, image_size)

    dists4 = compute.get_distances(positions)
    print('dists 2 to 1:', compute.compare_distances(dists2, dists1))
    print('dists 3 to 2:', compute.compare_distances(dists3, dists2))
    print('dists 4 to 2:', compute.compare_distances(dists4, dists2))
    print('dists 4 to 3:', compute.compare_distances(dists4, dists3))

    # helper.plot(imgs, positions, sizes, representative, canvas_size)
    # helper.plot_clusters(em_2d, cluster_centers, cluster_labels, representative)


if __name__ == '__main__':
    main()
