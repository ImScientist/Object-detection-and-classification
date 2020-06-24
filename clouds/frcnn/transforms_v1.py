import torch
import random
import torch.nn.functional as nnf
from torchvision.transforms import functional as ttf


class RandomHorizontalFlip(object):
    # bboxes (x,y) values start from (0,0) and not from (1,1)
    # width -> width - 1

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - 1 - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomVerticalFlip(object):
    # bboxes (x,y) values start from (0,0) and not from (1,1)
    # height -> height - 1

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - 1 - bbox[:, [3, 1]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-2)
        return image, target


class Resize(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image, target):
        height, width = image.shape[-2:]
        new_height, new_width = round(height * self.scale), round(width * self.scale)

        # TODO: remove the if condition
        bbox = target["boxes"]
        if bbox.size() != torch.Size([0]):
            bbox[:, [0, 2]] *= new_width / width
            bbox[:, [1, 3]] *= new_height / height
        target["boxes"] = bbox

        # image should be a 4D tensor (1, channels, height, width)
        # mode 'nearest' for masks and `bilinear` for images
        image = nnf.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=True)[0]

        if "masks" in target:
            target["masks"] = nnf.interpolate(
                target["masks"].unsqueeze(0).to(dtype=torch.float32),
                size=(new_height, new_width),
                mode='nearest')[0] \
                .to(dtype=torch.uint8)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = ttf.to_tensor(image)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_transform(training_mode):
    transforms = list()
    transforms.append(ToTensor())

    if training_mode:
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))

    return Compose(transforms)
