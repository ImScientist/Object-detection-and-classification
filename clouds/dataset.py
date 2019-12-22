import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from .preprocessing import preprocess_bboxes, get_masks_bboxes_of_disconnected_regions_from_encoded_pixels


class CloudsDataset(Dataset):

    def __init__(self, img_dir, labels_path, transforms=None, n_el=None, bad_images=()):
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
        img = Image.open(img_path).convert("RGB")

        masks_and_bboxes = get_masks_bboxes_of_disconnected_regions_from_encoded_pixels(data, img.height, img.width)

        masks = torch.as_tensor([el['mask'] for k, v in masks_and_bboxes.items() for el in v], dtype=torch.uint8)
        boxes = torch.as_tensor([el['bbox'] for k, v in masks_and_bboxes.items() for el in v], dtype=torch.float32)
        labels = torch.as_tensor([self.classmap[k] for k, v in masks_and_bboxes.items() for el in v], dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        image_id = torch.tensor([idx])
        num_objs = masks.size()[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

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
