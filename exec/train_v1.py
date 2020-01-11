import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from clouds.dataset import CloudsDataset
from clouds.torchvision_references.detection.engine import train_one_epoch, evaluate_one_epoch
from clouds.torchvision_references.detection import utils
from clouds.torchvision_references.detection.transforms import get_transform

from clouds.utils import get_model_instance_segmentation


def train_v1(img_dir_train: str,
             labels_path_train: str,
             model_dir: str = None,
             log_dir: str = None,
             num_classes: int = 5,
             size_tr_val: int = None,
             size_val: int = 500,
             batch_size: int = 4,
             print_freq: int = 10,
             num_epochs: int = 10,
             load_epoch: int = None,
             seed: int = 1):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset_train = CloudsDataset(img_dir_train, labels_path_train, get_transform(True), size_tr_val)
    dataset_test = CloudsDataset(img_dir_train, labels_path_train, get_transform(False), size_tr_val)

    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset_train)).tolist()

    assert len(dataset_train) > size_val, "validation set > data set"

    dataset_train = Subset(dataset_train, indices[:-size_val])
    dataset_test = Subset(dataset_test, indices[-size_val:])

    data_loader_train = DataLoader(dataset_train,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=1,
                                   collate_fn=utils.collate_fn)

    data_loader_test = DataLoader(dataset_test,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=1,
                                  collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model_instance_segmentation(num_classes, use_my_model=True)

    if load_epoch is not None:
        print('load saved model')
        model.load_state_dict(torch.load(os.path.join(model_dir,
                                                      f'state_dict_epoch_{load_epoch}.pth')))

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    if load_epoch is not None:
        epoch_range = range(load_epoch + 1, num_epochs)
    else:
        epoch_range = range(num_epochs)

    for epoch in epoch_range:

        logger_train = train_one_epoch(model, optimizer, data_loader_train, device, epoch,
                                       print_freq=print_freq)

        lr_scheduler.step(epoch=epoch)

        logger_test = evaluate_one_epoch(model, data_loader_test, device, epoch,
                                         print_freq=print_freq)

        torch.save(model.state_dict(), os.path.join(model_dir, f'state_dict_epoch_{epoch}.pth'))

        with SummaryWriter(log_dir) as w:
            for k, meter in logger_train.meters.items():
                w.add_scalars(k, {'train': meter.global_avg}, epoch)
            for k, meter in logger_test.meters.items():
                w.add_scalars(k, {'test': meter.global_avg}, epoch)

    print("That's it!")


if __name__ == "__main__":
    """ 
    python train_v1.py \
        --img_dir_train /content/drive/My Drive/data/source/clouds/train_images \
        --labels_path_train /content/drive/My Drive/data/source/clouds/train.csv \
        --model_dir /content/drive/My Drive/data/saved_models/clouds/delete \
        --log_dir /content/drive/My Drive/data/saved_models/clouds/delete/log \
        --size_tr_val 20 \
        --size_val 8 \
        --batch_size 4 \
        --print_freq 10 \
        --num_epochs 10 \
        --seed 1
        
        # --load_epoch 2    
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dir_train',
                        type=str,
                        dest='img_dir_train',
                        default=None,
                        help='Directory with training images.')

    parser.add_argument('--labels_path_train',
                        type=str,
                        dest='labels_path_train',
                        default=None,
                        help='Path of the csv file with training labels.')

    parser.add_argument('--model_dir',
                        type=str,
                        dest='model_dir',
                        default=None,
                        help='Directory where trained models will be saved.')

    parser.add_argument('--log_dir',
                        type=str,
                        dest='log_dir',
                        default=None,
                        help='Directory where training logs will be saved.')

    parser.add_argument('--size_tr_val',
                        type=int,
                        dest='size_tr_val',
                        default=None,
                        help='Numbers of different images from `labels_path_train` that will be read.\n'
                             'Used only to test the algorithm with a small amount of data.')

    parser.add_argument('--size_val',
                        type=int,
                        dest='size_val',
                        default=300,
                        help='Size of the validation set.')

    parser.add_argument('--batch_size',
                        type=int,
                        dest='batch_size',
                        default=4,
                        help='Batch size')

    parser.add_argument('--print_freq',
                        type=int,
                        dest='print_freq',
                        default=10,
                        help='Print training info every `print_freq` epochs.')

    parser.add_argument('--load_epoch',
                        type=int,
                        dest='load_epoch',
                        default=None,
                        help='Continue training by loading a model from a given epoch.')

    parser.add_argument('--num_epochs',
                        type=int,
                        dest='num_epochs',
                        default=10,
                        help='Number of training epochs.')

    parser.add_argument('--seed',
                        type=int,
                        dest='seed',
                        default=1,
                        help='seed')

    args = parser.parse_args()

    train_v1(img_dir_train=args.img_dir_train,
             labels_path_train=args.labels_path_train,
             model_dir=args.model_dir,
             log_dir=args.log_dir,
             num_classes=5,
             size_tr_val=args.size_tr_val,
             size_val=args.size_val,
             batch_size=args.batch_size,
             print_freq=args.print_freq,
             num_epochs=args.num_epochs,
             load_epoch=args.load_epoch,
             seed=args.seed)
