import os
import argparse
import torch
import segmentation_models_pytorch as smp
from clouds.unet.dataset_v3 import CloudsDataset
from clouds.postprocessing import get_encoded_predictions_single_image
from clouds.others.utils import \
    get_preprocessing, \
    get_validation_augmentation

ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = None
DATA_DIR = os.environ.get('DATA_DIR', 'data')


def predict(
        predictions_dir: str,
        predictions_file: str,
        img_dir_test: str,
        labels_path_test: str,
        model_dir: str = None,
        load_epoch: int = 0,
        nrows: int = None,
):
    os.makedirs(predictions_dir, exist_ok=True)
    save_file = os.path.join(predictions_dir, predictions_file)

    if os.path.isfile(save_file):
        os.remove(save_file)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )

    model.load_state_dict(
        torch.load(
            f=os.path.join(model_dir, f'state_dict_epoch_{load_epoch}.pth'),
            map_location=device
        )
    )

    model.to(device)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    dataset_test = CloudsDataset(
        img_dir=img_dir_test,
        labels_path=labels_path_test,
        transforms=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        n_el=nrows
    )

    for idx, (img, _) in enumerate(dataset_test):
        img_torch = torch.as_tensor(img, dtype=torch.float32).to(device)
        mask_pred = model(img_torch.unsqueeze(0)).squeeze(0)
        img_name = dataset_test.labels.iloc[idx]['image']

        enc_predictions = get_encoded_predictions_single_image(
            img_name=img_name, masks=mask_pred, threshold_mask=0.
        )

        with open(save_file, 'a') as f:
            f.writelines("{0:s}\n".format(item) for item in enc_predictions)

    print("That's it!")


if __name__ == "__main__":
    """
    setup $DATA_DIR (for local testing)

    python exec/predict_v3.py \
        --nrows 10 \
        --load_epoch 0
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--predictions_dir',
        type=str,
        dest='predictions_dir',
        default=os.path.join(DATA_DIR, 'predictions', 'v3'),
        help='Directory where predictions will be saved.')

    parser.add_argument(
        '--predictions_file',
        type=str,
        dest='predictions_file',
        default='submission.csv',
        help='Saved predictions file name.')

    parser.add_argument(
        '--img_dir_test',
        type=str,
        dest='img_dir_test',
        default=os.path.join(DATA_DIR, 'test_images'),
        help='Directory with test images.')

    parser.add_argument(
        '--labels_path_test',
        type=str,
        dest='labels_path_test',
        default=os.path.join(DATA_DIR, 'test.csv'),
        help='Path of the csv file with test labels.\n'
             'Actually they do not exist.\n'
             'We use empty lists.')

    parser.add_argument(
        '--model_dir',
        type=str,
        dest='model_dir',
        default=os.path.join(DATA_DIR, 'saved_models'),
        help='Directory where trained models will be saved.')

    parser.add_argument(
        '--nrows',
        type=int,
        dest='nrows',
        default=None,
        help='Number of images to predict.\n'
             'Used only to test the algorithm with a small amount of data.')

    parser.add_argument(
        '--load_epoch',
        type=int,
        dest='load_epoch',
        default=0,
        help='Continue training by loading a model from a given epoch.')

    args = parser.parse_args()

    predict(**vars(args))
