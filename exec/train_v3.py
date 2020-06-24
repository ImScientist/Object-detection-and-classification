import os
import argparse
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from clouds.unet.dataset_v3 import CloudsDataset
from clouds.unet.engine import train_one_epoch, evaluate_one_epoch
from clouds.utils import \
    get_preprocessing, \
    get_training_augmentation, \
    get_validation_augmentation

ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = None
DATA_DIR = os.environ.get('DATA_DIR', 'data')


def train(
        img_dir_train: str,
        labels_path_train: str,
        model_dir: str,
        log_dir: str = None,
        size_tr_val: int = None,
        size_val: int = 500,
        num_epochs: int = 19,
        batch_size: int = 2,
        print_freq: int = 10,
        load_epoch: int = None,
        seed: int = 1
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )
    model.to(device)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    dataset_train = CloudsDataset(
        img_dir=img_dir_train,
        labels_path=labels_path_train,
        transforms=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        n_el=size_tr_val
    )

    dataset_test = CloudsDataset(
        img_dir=img_dir_train,
        labels_path=labels_path_train,
        transforms=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        n_el=size_tr_val
    )

    indices = torch.randperm(len(dataset_train)).tolist()

    assert len(dataset_train) > size_val, "validation set >= data set"

    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-size_val])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-size_val:])

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
        {'params': model.segmentation_head.parameters(), 'lr': 1e-3}
    ])

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)

    criterion = nn.BCEWithLogitsLoss()

    if load_epoch is not None:
        epoch_range = range(load_epoch + 1, num_epochs)
    else:
        epoch_range = range(num_epochs)

    for epoch in epoch_range:

        logger_train = train_one_epoch(
            model, optimizer, criterion, data_loader_train, device, epoch, print_freq
        )

        lr_scheduler.step(metrics=logger_train.meters['loss'].value)

        logger_test = evaluate_one_epoch(
            model, criterion, data_loader_test, device, epoch, print_freq
        )

        torch.save(
            model.state_dict(),
            os.path.join(model_dir, f'state_dict_epoch_{epoch}.pth')
        )

        with SummaryWriter(log_dir) as w:
            for k, meter in logger_train.meters.items():
                w.add_scalars(k, {'train': meter.global_avg}, epoch)
            for k, meter in logger_test.meters.items():
                w.add_scalars(k, {'test': meter.global_avg}, epoch)

    print("That's it!")


if __name__ == "__main__":
    """
    setup $DATA_DIR
     
    python exec/train_v3.py \
        --size_tr_val 20 \
        --size_val 8 \
        --batch_size 2 \
        --print_freq 2 \
        --num_epochs 3 \
        --seed 1

        # --load_epoch 2    
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--img_dir_train',
        type=str,
        dest='img_dir_train',
        default=os.path.join(DATA_DIR, 'train_images'),
        help='Directory with training images.')

    parser.add_argument(
        '--labels_path_train',
        type=str,
        dest='labels_path_train',
        default=os.path.join(DATA_DIR, 'train.csv'),
        help='Path of the csv file with training labels.')

    parser.add_argument(
        '--model_dir',
        type=str,
        dest='model_dir',
        default=os.path.join(DATA_DIR, 'saved_models'),
        help='Directory where trained models will be saved.')

    parser.add_argument(
        '--log_dir',
        type=str,
        dest='log_dir',
        default=os.path.join(DATA_DIR, 'logs'),
        help='Directory where training logs will be saved.')

    parser.add_argument(
        '--size_tr_val',
        type=int,
        dest='size_tr_val',
        default=None,
        help='Numbers of different images from `labels_path_train` that will be read.\n'
             'Used only to test the algorithm with a small amount of data.')

    parser.add_argument(
        '--size_val',
        type=int,
        dest='size_val',
        default=300,
        help='Size of the validation set.')

    parser.add_argument(
        '--batch_size',
        type=int,
        dest='batch_size',
        default=4,
        help='Batch size')

    parser.add_argument(
        '--print_freq',
        type=int,
        dest='print_freq',
        default=10,
        help='Print training info every `print_freq` epochs.')

    parser.add_argument(
        '--load_epoch',
        type=int,
        dest='load_epoch',
        default=None,
        help='Continue training by loading a model from a given epoch.')

    parser.add_argument(
        '--num_epochs',
        type=int,
        dest='num_epochs',
        default=10,
        help='Number of training epochs.')

    parser.add_argument(
        '--seed',
        type=int,
        dest='seed',
        default=1,
        help='seed')

    args = parser.parse_args()

    train(**vars(args))
