""" An example from
https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools#Importing-libraries
that makes use of the catalyst library.
"""
import os
import torch
import pandas as pd
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback
from clouds.others.utils import CloudDataset
from clouds.others.utils import \
    get_preprocessing, \
    get_training_augmentation, \
    get_validation_augmentation

ENCODER = 'resnet50'  # 'resnet18'  #
ENCODER_WEIGHTS = 'imagenet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTIVATION = None


def train_model(
        data_dir: str,
        logdir: str = "./logs/segmentation",
        num_workers: int = 0,
        num_epochs: int = 19,
        bs: int = 2
):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    id_mask_count = train.loc[train['EncodedPixels'].isnull() is False, 'Image_Label'] \
        .apply(lambda x: x.split('_')[0]) \
        .value_counts() \
        .reset_index() \
        .rename(columns={'index': 'img_id', 'Image_Label': 'count'})

    train_ids, valid_ids = train_test_split(
        id_mask_count['img_id'].values,
        random_state=42,
        stratify=id_mask_count['count'],
        test_size=0.1)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)

    train_dataset = CloudDataset(
        df=train,
        datatype='train',
        data_dir=data_dir,
        img_ids=train_ids,
        transforms=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn))

    valid_dataset = CloudDataset(
        df=train,
        datatype='valid',
        data_dir=data_dir,
        img_ids=valid_ids,
        transforms=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    valid_loader = DataLoader(
        valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    # model, criterion, optimizer
    #
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = smp.utils.losses.BCEWithLogitsLoss()
    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback(),
                   EarlyStoppingCallback(patience=5, min_delta=0.001)],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )

    return runner


if __name__ == '__main__':
    train_model()
