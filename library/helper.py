import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageDraw
from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

datasets = {
    1: '1',
    2: 'traffic-signs',
    3: 'bone-healing',
    4: 'dicty-development',
    5: 'yplp'
}


def get_images(dataset_number):
    image_file_paths = glob.glob('../data/' + datasets[dataset_number] + '/*/*')
    return list(map(lambda x: x.replace('\\', '/'), image_file_paths))


def get_embeddings(dataset_number, image_file_paths):
    file_name = '../data/saved_embeddings/' + datasets[dataset_number] + '.npy'

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


def plot(image_file_paths, positions, sizes, border=10):
    tmp_positions = positions - np.min(positions, axis=0) + border

    canvas_size = np.max(tmp_positions + sizes, axis=0) + border
    canvas_size = tuple(canvas_size.astype(np.int32))

    vis = Image.new('RGB', canvas_size, (255, 255, 255))
    draw = ImageDraw.Draw(vis)

    for image_file_name, pos, size, i in zip(image_file_paths, tmp_positions, sizes, range(len(sizes))):
        image = Image.open(image_file_name)
        image.thumbnail(tuple(size.astype(np.int32)))
        vis.paste(image, (int(pos[0]), int(pos[1])))

        r = [pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]]
        draw.rectangle(r, outline=(0, 0, 0))

    vis.show()