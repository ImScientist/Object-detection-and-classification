from PIL import Image, ImageDraw
from .preprocessing import get_masks_bboxes_of_disconnected_regions_from_encoded_pixels
from typing import List, Dict, Tuple, Union
import numpy as np
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


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

            new_mask = Image.fromarray((mask * 210).astype(np.uint8), mode='L')
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


def visualize(image, mask, original_image=None, original_mask=None):
    """ Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {'Sugar': 0, 'Gravel': 1, 'Flower': 2, 'Fish': 3}

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(20, 10))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f'Mask {class_dict[i]}', fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 5, figsize=(20, 10))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        for k, v in class_dict.items():
            ax[0, v + 1].imshow(original_mask[:, :, v])
            ax[0, v + 1].set_title(f'Original mask {k}', fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title('Transformed image', fontsize=fontsize)

        for k, v in class_dict.items():
            ax[1, v + 1].imshow(mask[:, :, v])
            ax[1, v + 1].set_title(f'Transformed mask {k}', fontsize=fontsize)
