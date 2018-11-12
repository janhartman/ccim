import helper
import compute


def main(k=4, image_size=100, canvas_size=500):
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

    # TODO: is this right? the top K biggest silhouettes won't necessary be in different clusters
    # Sizes and positions of the images
    sizes = compute.get_sizes(silhouettes, image_size)
    positions = compute.get_positions(em_2d, canvas_size - image_size)

    # helper.plot_clusters(em_2d, cluster_centers, cluster_labels, representative)

    # helper.plot(imgs, positions, sizes, canvas_size)  # overlap

    # Expand as long as overlaps occur
    iters = 0
    while compute.overlap(positions, sizes):
        positions *= 1.05  # gradually increase space between images
        canvas_size *= 1.05
        iters += 1

    print('overlap', iters)

    # helper.plot(imgs, positions, sizes, canvas_size)

    # Overlapping resolved, now "shrink" towards representative images
    positions = compute.shrink_intra(positions, sizes, representative, cluster_labels, image_size)

    # helper.plot(imgs, positions, sizes, canvas_size)

    # Move clusters closer together
    positions, canvas_size = compute.shrink_inter(positions, sizes, representative, cluster_labels, image_size)

    helper.plot(imgs, positions, sizes, canvas_size)


if __name__ == '__main__':
    main()
