import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from src.data.preprocessing import mask_from_compact_notation
from src.utils import cloud_colors


def visualize_labeled_image(
        img_path: str,
        data: dict[str, list[int]],
        downscale: int = 4,
        img_height: int = 1400,
        img_width: int = 2100
):
    """ Create an image with the regions of different cloud types

    Parameters
    ----------
      img_path:
      data: cloud type (key) to compact mask definition (value)
      downscale: how many times to downscale the image
      img_height:
      img_width:
    """

    colors = cloud_colors()

    masks = {k: mask_from_compact_notation(v, img_height, img_width)
             for k, v in data.items()}

    img = Image.open(img_path).convert('RGBA')
    img.putalpha(256)

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(overlay)

    for k, mask in masks.items():
        new_mask = Image.fromarray((mask * 110).astype(np.uint8), mode='L')
        drawing.bitmap((0, 0), new_mask, fill=colors[k])

    img = Image.alpha_composite(img, overlay)
    img = img.resize((img.width // downscale, img.height // downscale))

    return img


def visualize_multiple_predictions(y, y_hat, n: int = 5):
    """ Visualize predicted segments for multiple images

    Parameters
    ----------
      y: tensor or np.array with shape(None, widht, height, 4)
      y_hat: tensor or np.array with shape(None, widht, height, 4)
      n: number of images
    """

    assert y.ndim == 4
    assert y_hat.ndim == 4

    figsize = (12, 4 * n)
    fig = plt.figure(figsize=figsize)

    idx = 1
    for r in range(n):
        for c in range(4):
            plt.subplot(2 * n, 4, idx)
            plt.title('segments')
            plt.imshow(y[r, ..., c])
            plt.axis('off')

            plt.subplot(2 * n, 4, idx + 4)
            plt.title('predictions')
            plt.imshow(y_hat[r, ..., c])
            plt.axis('off')

            idx += 1
        idx += 4

    plt.tight_layout()
    return fig


# TOOD: include masks
def visualize_image_augmentations(img_orig, img_augmented, figsize=(20, 8)):
    """ Visualize image or mask augmentations """

    fig = plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(img_orig)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(img_augmented)

    return fig
