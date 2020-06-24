import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from clouds.preprocessing import preprocess_bboxes, get_masks_bboxes_of_disconnected_regions_from_encoded_pixels


class CloudsDataset(Dataset):

    def __init__(
            self, img_dir, labels_path, transforms=None, n_el=None, bad_images=()
    ):
        self.img_dir = img_dir
        self.labels_path = labels_path
        self.transforms = transforms

        labels = preprocess_bboxes(labels_path, n_el)
        self.labels = labels[~labels['image'].isin(bad_images)]

        self.imgs = self.labels['image'].to_list()
        self.classmap = {
            'Sugar': 1,
            'Gravel': 2,
            'Flower': 3,
            'Fish': 4
        }

    def __getitem__(self, idx):
        record = self.labels.iloc[idx]
        img_name, data = record['image'], record['pixels']

        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_height, img_width = img.shape[:2]
        masks_and_bboxes = get_masks_bboxes_of_disconnected_regions_from_encoded_pixels(
            data, img_height, img_width
        )

        masks = np.stack([
            el['mask'] for k, v in masks_and_bboxes.items() for el in v
        ], axis=0)  # shape = (#masks, h, w)
        boxes = np.array([
            el['bbox'] for k, v in masks_and_bboxes.items() for el in v
        ])
        labels = np.array([
            self.classmap[k] for k, v in masks_and_bboxes.items() for _ in v
        ])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        image_id = idx
        iscrowd = np.zeros(shape=(len(labels),))

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
