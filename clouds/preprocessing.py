import cv2
import pandas as pd
import numpy as np
from collections import ChainMap
from typing import List, Dict, Tuple, Union, Any


def split_str(x: str) -> List[int]:
    if type(x) == str:
        return [int(el) for el in x.split(' ')]
    else:
        return []


def preprocess_bboxes(data_path: str, n_el: int = 100):

    if n_el:
        df = pd.read_csv(data_path, nrows=4*n_el)
    else:
        df = pd.read_csv(data_path)

    df['pixels'] = df['EncodedPixels'].apply(lambda x: split_str(x))

    df['type'] = df['Image_Label'].apply(lambda x: x.split('_')[-1])

    df['image'] = df['Image_Label'].apply(lambda x: x.split('_')[0])

    df['pixels'] = df[['type', 'pixels']].apply(lambda x: {x[0]: x[1]}, 1)

    df.drop(labels=['Image_Label', 'EncodedPixels', 'type'], axis=1, inplace=True)

    df = df \
        .groupby(by=['image'], as_index=False) \
        .agg(list)

    df['pixels'] = df['pixels'].apply(lambda x: dict(ChainMap(*x)))

    df = df.sort_values(by=['image'])

    return df


def grouped(iterable, n):
    """
    s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ...
    """
    return zip(*[iter(iterable)] * n)


def get_bbox(mask):
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]


def resize_bbox_position(bbox, scale=4):
    """

    :param bbox: = [x1, y1, x2, y2]
    :param scale: scaling factor
    """
    bbox = (bbox/scale).round().astype(int)
    return bbox


def encoded_pixels_to_mask(encoded_pixels, img_height: int = 1400, img_width: int = 2100):
    """

    :param encoded_pixels:
        the zeroth, second, fourth .. element correspond to pixel position of flattened 2d array
        the first, third, fifth .. element correspond to the length of the vertical line that starts
            from one of the elements from the previous group
    :param img_height:
    :param img_width:
    :return:
    """

    mask = np.zeros(shape=(img_height * img_width,), dtype=int)

    for px_pos, n_px in grouped(encoded_pixels, 2):
        mask[px_pos - 1: px_pos - 1 + n_px] = 1

    # first fill in first column then the next one and so on
    mask = mask.reshape((img_height, img_width), order='F')

    return mask


def get_masks_bboxes_of_disconnected_regions_from_encoded_pixels(data: Dict, img_height: int = 1400, img_width: int = 2100):
    """

    :param data:
        {
            'Sugar': List[int]  # encoded_pixels,
            'Gravel': List[int]  # encoded_pixels
            'Flower': List[int]  # encoded_pixels
            'Fish': List[int]  # encoded_pixels
        }
    :param img_height:
    :param img_width:
    :return:
        {
            'Sugar': [
                {
                    'mask': ..,
                    'bbox': ..
                },
                ...
            ],
            'Gravel': [
                {
                    'mask': ..,
                    'bbox': ..
                },
                ...
            ],
            ...
        }
    """
    masks_and_bboxes = {}

    for k, v in data.items():

        mask = encoded_pixels_to_mask(v, img_height, img_width)

        # separate the binary mask in a list of non-connected regions
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

        # labels == 0 is the background. ignore it
        masks = [(labels == el).astype(int) for el in range(1, n_labels)]
        masks = [el for el in masks if el.sum() > 10000]

        bboxes = [get_bbox(el) for el in masks]

        masks_and_bboxes[k] = [{'mask': mask, 'bbox': bbox} for mask, bbox in zip(masks, bboxes)]

    return masks_and_bboxes


