import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageDraw
from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

datasets = {
    1: '1',
}


def get_images(dataset_number):
    image_dir = '../data/' + datasets[dataset_number]
    return [image_dir + '/' + file for file in os.listdir(image_dir)]


def get_embeddings(image_file_paths):
    file_name = '../data/saved_embeddings/' + image_file_paths[0].split('/')[2] + '.npy'

    # read from file if it exists to save time
    if os.path.isfile(file_name):
        return np.load(file_name)
    else:
        with ImageEmbedder(model='inception-v3', layer='penultimate') as embedder:
            embeddings = embedder(image_file_paths)
        np.save(file_name, embeddings)
        return embeddings


def plot_clusters(em_2d, cluster_centers, cluster_labels, rep):
    plt.scatter(em_2d[:, 0], em_2d[:, 1], c=cluster_labels, s=50, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.3)
    plt.scatter(em_2d[rep, 0], em_2d[rep, 1], c='red', s=100, alpha=0.3)
    plt.show()


def plot(image_file_paths, positions, sizes, representative, canvas_size, border=10):
    canvas_size = int(canvas_size) + 2 * border
    vis = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(vis)

    for image_file_name, pos, size, i in zip(image_file_paths, positions + border, sizes, range(len(sizes))):
        size = int(size)
        image = Image.open(image_file_name)

        # if i in representative:
        #     image = ImageOps.expand(image, border=5, fill=100)

        image.thumbnail((size, size))

        r = [pos[0], pos[1], pos[0] + size, pos[1] + size]
        draw.rectangle(r, fill=(255, 255, 255, 255), outline=(0, 0, 0))

        vis.paste(image, (int(pos[0]), int(pos[1])))

    # x, y = np.mean(positions, axis=0)
    # r = [int(x), int(y), int(x) + 5, int(y)+5]
    # draw.rectangle(r, fill=(255, 0, 0))

    vis.show()
