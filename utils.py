import math
import numpy as np


def make_grid(batch_images, nrow=8):
    if len(batch_images.shape) == 4 and batch_images.shape[1] == 1:  # single-channel images
        batch_images = np.concatenate((batch_images, batch_images, batch_images), axis=1)

    # make the mini-batch of images into a grid
    nmaps = batch_images.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(batch_images.shape[2]), int(batch_images.shape[3])
    grid = np.zeros((3, height * ymaps, width * xmaps))
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            i = k // xmaps
            j = k % ymaps
            grid[:, i*height:(i+1)*height, j*width:(j+1)*width] = batch_images[k]
            k = k + 1
    return grid


def save_image(batch_images, filename, nrow=8):
    from PIL import Image
    grid = make_grid(batch_images, nrow=nrow)
    ndarr = np.transpose(np.clip(grid * 255, 0, 255), (1, 2, 0))
    im = Image.fromarray(ndarr.astype('uint8'))
    im.save(filename)

def denorm(x):
    out = (x + 1) / 2
    return np.clip(out, 0, 1)


