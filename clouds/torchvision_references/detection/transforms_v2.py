import torch
import numpy as np
from torchvision.transforms import functional as F


class ImgAugmentation(object):
    """ Apply a sequence of augmentations to the image and targets
    """

    def __init__(self, comp_aug):
        self.comp_aug = comp_aug

    def __call__(self, image, target):
        """
        
        :param image: 
        :param target:
            {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,         # shape = (#masks,h,w)
                "image_id": image_id,
                "area": area,
                "iscrowd": iscrowd
            }
        :return:
            image ,
            targets = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,         # shape = (#masks,h,w) !!!
                "image_id": image_id,
                "area": area,
                "iscrowd": iscrowd
            }

        """

        n_boxes = 0
        augmented = {}
        while n_boxes == 0:
            augmented = self.comp_aug(image=image,
                                      mask=target['masks'].transpose(1, 2, 0),  # (#masks,h,w) -> (h,w,#masks)
                                      bboxes=target['boxes'],
                                      labels=target['labels'])
            n_boxes = len(augmented['bboxes'])

        boxes = np.array(augmented['bboxes']).astype(int)

        new_target = {
            "boxes": boxes,
            "labels": augmented['labels'],
            "masks": augmented['mask'].transpose(2, 0, 1),  # (h,w,#masks) -> (#masks,h,w)
            "image_id": target['image_id'],
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": [0] * len(augmented['labels'])
        }

        new_image = augmented['image']

        return new_image, new_target


class ToTensor(object):
    def __call__(self, image, target):

        image = F.to_tensor(image)

        if 'boxes' in target:
            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)

        if 'labels' in target:
            target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)

        if 'masks' in target:
            target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)

        if 'image_id' in target:
            target['image_id'] = torch.tensor([target['image_id']], dtype=torch.int64)

        if 'area' in target:
            target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)

        if 'iscrowd' in target:
            target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.float32)

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_transform(training_mode, composite_aug):

    transforms = list()
    if training_mode:
        transforms.append(ImgAugmentation(composite_aug))

    transforms.append(ToTensor())

    return Compose(transforms)
