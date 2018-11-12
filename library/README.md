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


## Idea 2
- iteratively place images (representative first)
- for each image, find already placed images that should be close to it and place it randomly while taking those into account
- option: use iterative deepening (increase canvas size if we cannot place images)
- option: calculate the image sizes so that a placement will be possible
- use power function to down/upscale image sizes on [0, 1] interval

