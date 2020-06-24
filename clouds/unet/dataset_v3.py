import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from clouds.preprocessing import preprocess_bboxes, encoded_pixels_to_mask


class CloudsDataset(Dataset):

    def __init__(
            self,
            img_dir,
            labels_path,
            transforms=None,
            preprocessing=None,
            n_el=None,
            bad_images=()
    ):
        self.img_dir = img_dir
        self.labels_path = labels_path
        self.transforms = transforms
        self.preprocessing = preprocessing

        labels = preprocess_bboxes(labels_path, n_el)
        self.labels = labels[~labels['image'].isin(bad_images)]

        self.imgs = self.labels['image'].to_list()
        self.classes = ['Sugar', 'Gravel', 'Flower', 'Fish']
        self.classmap = {
            'Sugar': 0,
            'Gravel': 1,
            'Flower': 2,
            'Fish': 3
        }

    def __getitem__(self, idx):
        record = self.labels.iloc[idx]
        img_name, data = record['image'], record['pixels']

        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.stack(
            [encoded_pixels_to_mask(data[k]) for k in self.classes]
        ).transpose((1, 2, 0))  # shape = (h, w, #masks)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']

        return img, mask

    def __len__(self):
        return len(self.imgs)
