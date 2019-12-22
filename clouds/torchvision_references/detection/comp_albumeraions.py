import albumentations as A
from albumentations import Compose as AlbCompose
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    BboxParams,
    RandomRotate90,
    RandomSizedBBoxSafeCrop,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    Resize,
    RGBShift,
    Transpose
)

augs = [
    OneOf([
        RandomSizedBBoxSafeCrop(width=1280, height=960, erosion_rate=0.2, p=1.0),
        Resize(width=1280, height=960, p=1.0)
        # RandomSizedBBoxSafeCrop(width=1280, height=960, erosion_rate=0.2, p=1.0),
        # RandomSizedBBoxSafeCrop(width=1280, height=1280, erosion_rate=0.2, p=1.0),
        # Resize(width=1200, height=800, p=1.0)
    ], p=1),
    A.augmentations.transforms.ToGray(p=0.25),
    A.augmentations.transforms.ChannelShuffle(p=0.25),
    RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5),
    RandomGamma(p=0.5),
    HorizontalFlip(p=0.25),
    VerticalFlip(p=0.25)
    # RandomRotate90(p=0.25),
    # Transpose(p=0.25)
]

comp_aug = AlbCompose(augs,
                      bbox_params=BboxParams(format='pascal_voc',
                                             min_visibility=0.2,
                                             label_fields=['labels']))
