import numpy as np

from PIL import Image, ImageDraw
from src.data.generate_tr_va_te import mask_from_compact_notation
from src.utils import cloud_colors


def create_labeled_image(
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
