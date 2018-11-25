import helper
import compute

import numpy as np

def main(k=4, image_size=200, sparseness=0.5, dataset_number=4):
    # Load images and get embeddings from NN
    imgs = helper.get_images(dataset_number)
    embeddings = helper.get_embeddings(dataset_number, imgs)

    # Compute 2D embeddings with MDS / UMAP
    em_2d = compute.mds(embeddings)
    #em_2d = compute.umap(embeddings)

    # Perform K-means clustering, compute silhouettes
    cluster_centers, cluster_labels = compute.k_means(em_2d, k)
    silhouettes = compute.silhouette(em_2d, cluster_labels)

    # K representative images
    representative = compute.get_representative(em_2d, cluster_centers, cluster_labels, silhouettes)

    # Sizes and positions of the images
    sizes = compute.get_sizes(image_size, em_2d, cluster_centers, cluster_labels, representative)
    positions = compute.get_positions(em_2d, image_size)

    # helper.plot_clusters(em_2d, cluster_centers, cluster_labels, representative)
    # return

    #helper.plot(imgs, positions, sizes)  # overlap

    # Expand as long as overlaps occur - gradually increase space between images
    iters = 0
    while compute.overlap(positions, sizes):
        positions *= 1.05
        iters += 1
    print('overlap resolved in {} iterations'.format(iters))

    #helper.plot(imgs, positions, sizes)
    dists1 = compute.get_distances(positions)

    # Overlapping resolved, now "shrink" towards representative images
    positions = compute.shrink_intra(positions, sizes, representative, cluster_labels)
    dists2 = compute.get_distances(positions)
    #helper.plot(imgs, positions, sizes)

    # Move clusters closer together by same factor
    positions = compute.shrink_inter1(positions, sizes, representative, cluster_labels)
    dists3 = compute.get_distances(positions)
    #helper.plot(imgs, positions, sizes)

    # Move clusters closer together separately by different factors
    positions = compute.shrink_inter2(positions, sizes, representative, cluster_labels)
    dists3 = compute.get_distances(positions)
    #helper.plot(imgs, positions, sizes)

    # Further shrink (move images that are closer to center first)
    positions = compute.shrink_with_sparseness(positions, sizes, sparseness)
    dists4 = compute.get_distances(positions)
    helper.plot(imgs, positions, sizes)

    print()
    print('dists 2 score:', compute.compare_distances(dists2, dists1))
    print('dists 3 score:', compute.compare_distances(dists3, dists1))
    print('dists 4 score:', compute.compare_distances(dists4, dists1))



if __name__ == '__main__':
    main()
