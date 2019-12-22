from PIL import Image, ImageDraw
from .preprocessing import get_masks_bboxes_of_disconnected_regions_from_encoded_pixels
from typing import List, Dict, Tuple, Union
import numpy as np
from torchvision.transforms import functional as F


def create_labeled_image(img_path: str, data: Dict, scale: int = 4):
    """ Create an image with the regions of different cloud types

    :param img_path:
    :param data:
        {
            'Sugar': List[int]  # encoded_pixels,
            'Gravel': List[int]  # encoded_pixels
            'Flower': List[int]  # encoded_pixels
            'Fish': List[int]  # encoded_pixels
        }
    :param scale:
        variable for rescaling the dimensions of the image
    :return:
    """

    colors = {
        'Sugar': (128, 0, 0, 128),
        'Gravel': (0, 128, 0, 128),
        'Flower': (0, 0, 128, 128),
        'Fish': (128, 0, 128, 128)
    }

    all_masks_and_bboxes = get_masks_bboxes_of_disconnected_regions_from_encoded_pixels(data)

    img = Image.open(img_path).convert('RGBA')
    img.putalpha(256)

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(overlay)

    for k, masks_and_bboxes in all_masks_and_bboxes.items():

        for mask_and_bbox in masks_and_bboxes:
            mask = mask_and_bbox['mask']
            bbox = mask_and_bbox['bbox']

            new_mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            drawing.bitmap((0, 0), new_mask, fill=colors[k])
            drawing.rectangle(bbox, outline=(0, 0, 0, 256))

    img = Image.alpha_composite(img, overlay)
    img = img.resize((img.width // scale, img.height // scale))

    return img


def create_labled_image_from_dataloader_batch(images, targets):
    """ Plot the output of the dataloader.

    Check if all custom data transformations and augmentation steps are
    implemented properly. Plot images bounding boxes and masks.

    :param images: batch of torch.tensors
    :param targets: batch of dictionaries, each of them having the same content as a CloudsDataset element
    :return:
        PIL.image

    Example:
        import math
        import matplotlib.pyplot as plt

        dataiter = iter(data_loader)
        images, targets = dataiter.next()

        all_images = create_labled_image_from_dataloader_batch(images, targets)

        n = len(all_images)
        columns = 5
        rows = math.ceil(n / float(columns))

        fig = plt.figure(figsize=(20, 2.5 * columns))

        for idx, image in enumerate(all_images, start=1):
            ax = fig.add_subplot(rows, columns, idx, xticks=[], yticks=[])
            plt.imshow(image)
            ax.set_title("title {0:d}".format(idx), color="green")

        plt.pause(0.001)
        plt.show()
    """
    colors = {
        'Sugar': (128, 0, 0, 128),
        'Gravel': (0, 128, 0, 128),
        'Flower': (0, 0, 128, 128),
        'Fish': (128, 0, 128, 128)
    }

    inv_map = {
        1: 'Sugar',
        2: 'Gravel',
        3: 'Flower',
        4: 'Fish'
    }

    all_images = []

    for image, target in zip(images, targets):

        masks = target['masks'].numpy()
        labels = target['labels'].numpy()
        bboxes = target['boxes'].numpy().astype(int)

        img = F.to_pil_image(pic=image, mode='RGB').convert('RGBA')
        img.putalpha(256)

        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        drawing = ImageDraw.Draw(overlay)

        for label, mask, bbox in zip(labels, masks, bboxes):

            color = colors[inv_map[label]]
            new_mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            drawing.bitmap((0, 0), new_mask, fill=color)
            drawing.rectangle(list(bbox), outline=(0, 0, 0, 256))

        img = Image.alpha_composite(img, overlay)

        all_images.append(img)

    return all_images


# # TODO: this is wrong !!
# def create_labeled_image(img_path: str, data: Dict):
#     """ create an image with the regions of different cloud types
#
#     :param img_path:
#     :param data:
#         {
#             'Sugar': List[int],  # encoded pixels
#             'Gravel': List[int],  # encoded pixels
#             'Flower': List[int],  # encoded pixels
#             'Fish': List[int]  # encoded pixels
#         }
#     :return:
#     """
#
#     img_1 = Image.open(img_path).convert('RGBA')
#     img_2 = Image.new(mode='RGBA', size=img_1.size, color=(0, 0, 0, 255))
#     draw = ImageDraw.Draw(img_2)
#
#     for k, v in data.items():
#         color = colors[k]
#         lines = get_lines_from_encoded_pixels(v)
#         for line in lines:
#             draw.line(line, fill=color)
#
#     img = Image.blend(img_1, img_2, alpha=0.5)
#     img = img.resize((img.width // 4, img.height // 4))
#
#     return img


# # TODO: this is wrong
# def create_labeled_image_v2(img_path, data):
#     """ create an image with the regions of different cloud types
#
#     In this case we draw every pixel separately (this is a check that
#     we have not fucked up the process of extraction of individual pixels)
#
#     :param img_path:
#     :param data:
#     :return:
#     """
#     img_1 = Image.open(img_path).convert('RGBA')
#     w_max, h_max = img_1.width, img_1.height
#
#     img_2 = Image.new(mode='RGBA', size=img_1.size, color=(0, 0, 0, 255))
#     draw = ImageDraw.Draw(img_2)
#
#     for k, v in data.items():
#         color = colors[k]
#         pixels = get_all_pixels_from_encoded_pixels(v)
#         pixels = list(filter(lambda x: 0 <= x[0] < w_max and 0 <= x[1] < h_max, pixels))
#         for px in pixels:
#             draw.point(px, fill=color)
#
#     img = Image.blend(img_1, img_2, alpha=0.5)
#     img = img.resize((img.width // 4, img.height // 4))
#
#     return img
