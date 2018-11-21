# Library

This directory contains the library of implemented algorithms.

The aims of the algorithm:
- use MDS/UMAP for the initial 2D placement
- images do not overlap
- the size matches the weight (typicality?) of the image
- the distances between images match the distances between their vector representations (cosine distance?)

- basic idea: use silhouette for image "typicality" and make sure that only a few images are large (cutoff for a few images with the maximal silhouette)


## Idea 1
- place images with MDS
- expand until there are no more overlaps
- shrink towards cluster centers with "gravity" to remove blank space, but do not cause overlaps

- intra-cluster shrinking: start with biggest
- inter-cluster shrinking: go over all alphas to make sure clusters are moved by same factor

- sizing: use ranking (bring down higher sizes with 1/x to some power)

- scoring: MDS has perfect score, so does scaled MDS
- when shrinking, try to get as close as possible (first, just print them out)
- score using distance matrices (try more different functions)

- sparseness parameter determining how much empty space we will have
- randomly move images closer to the mean by that parameter (e.g. 10% - move by that percentage)

- local or global scoring (depends on starting method: MDS/tSNE)
- absolute or square?

## Idea 2
- iteratively place images (representative first)
- for each image, find already placed images that should be close to it and place it randomly while taking those into account
- option: use iterative deepening (increase canvas size if we cannot place images)
- option: calculate the image sizes so that a placement will be possible
- use power function to down/upscale image sizes on [0, 1] interval

