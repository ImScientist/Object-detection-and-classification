import albumentations as A
from albumentations import Compose as AlbCompose

augs = [
    A.OneOf([
        A.RandomSizedBBoxSafeCrop(width=1280, height=960, erosion_rate=0.2, p=1.0),
        A.Resize(width=1280, height=960, p=1.0)
        # RandomSizedBBoxSafeCrop(width=1280, height=960, erosion_rate=0.2, p=1.0),
        # RandomSizedBBoxSafeCrop(width=1280, height=1280, erosion_rate=0.2, p=1.0),
        # Resize(width=1200, height=800, p=1.0)
    ], p=1),
    A.ToGray(p=0.25),
    A.ChannelShuffle(p=0.25),
    A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=0.5),
    A.RandomGamma(p=0.5),
    A.HorizontalFlip(p=0.25),
    A.VerticalFlip(p=0.25)
    # RandomRotate90(p=0.25),
    # Transpose(p=0.25)
]

comp_aug = AlbCompose(
    augs,
    bbox_params=A.BboxParams(
        format='pascal_voc', min_visibility=0.2, label_fields=['labels']
    )
)
