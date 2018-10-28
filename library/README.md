# Library

This directory contains the library of implemented algorithms.

The aims of the algorithm:
- use MDS/UMAP for the initial 2D placement
- images do not overlap
- the size matches the weight (typicality?) of the image
- the distances between images match the distances between their vector representations (cosine distance?)

- basic idea: use silhouette for image "typicality" and make sure that only a few images are large (cutoff for a few images with the maximal silhouette)
