import os
import argparse
import torch
from torch.utils.data import Dataset

from clouds.postprocessing import torch_reduce_predictions, \
    torch_get_encoded_predictions_all_classes_single_image
from clouds.dataset import CloudsDataset
from clouds.torchvision_references.detection import utils
from clouds.torchvision_references.detection.transforms import get_transform

from clouds.utils import get_model_instance_segmentation


def predict_v1(predictions_dir: str,
               predictions_file: str,
               img_dir_test: str,
               labels_path_test: str,
               model_dir: str = None,
               num_classes: int = 5,
               size_test: int = None,
               batch_size: int = 4,
               load_epoch: int = 0):
    save_file = os.path.join(predictions_dir, predictions_file)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model_instance_segmentation(num_classes, use_my_model=True)
    model.to(device)

    model.load_state_dict(torch.load(f=os.path.join(model_dir, f'state_dict_epoch_{load_epoch}.pth'),
                                     map_location=device))

    model.eval()

    dataset_test = CloudsDataset(img_dir_test, labels_path_test, get_transform(False), size_test)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=2,
                                                   collate_fn=utils.collate_fn)

    os.makedirs(predictions_dir, exist_ok=True)

    dataiter = iter(data_loader_test)

    if os.path.isfile(save_file):
        os.remove(save_file)

    while True:
        try:
            images, targets = dataiter.next()  # in the test set the targets are empty lists

            images = [img.to(device) for img in images]

            print(targets[0]['image_id'][0])

            with torch.no_grad():
                predictions_batch = model(images)

            for prediction, target in zip(predictions_batch, targets):
                res = torch_reduce_predictions(prediction=prediction,
                                               top_n=5,
                                               threshold_score=0.7)

                idx = target['image_id'].item()

                enc_predictions = torch_get_encoded_predictions_all_classes_single_image(
                    img_name=dataset_test.labels.iloc[idx]['image'],
                    masks=res['masks'],
                    labels=res['labels'],
                    threshold_mask=0.5)

                with open(save_file, 'a') as f:
                    f.writelines("{0:s}\n".format(item) for item in enc_predictions)

        except StopIteration as e:
            print('DONE')
            break


if __name__ == "__main__":
    """ 
    python predict_v1.py \
        --predictions_dir /content/drive/My Drive/data/source/clouds/predictions/raw_predictions_temp \
        --predictions_file result_augmentation_ep_3.txt \
        --img_dir_test /content/drive/My Drive/data/source/clouds/test_images \
        --labels_path_test /content/drive/My Drive/data/source/clouds/test.csv \
        --model_dir /content/drive/My Drive/data/saved_models/clouds/delete \
        --size_test 20 \
        --batch_size 4 \
        --load_epoch 3
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_dir',
                        type=str,
                        dest='predictions_dir',
                        default=None,
                        help='Directory where predictions will be saved.')

    parser.add_argument('--predictions_file',
                        type=str,
                        dest='predictions_file',
                        default=None,
                        help='Saved predictions file name.')

    parser.add_argument('--img_dir_test',
                        type=str,
                        dest='img_dir_test',
                        default=None,
                        help='Directory with test images.')

    parser.add_argument('--labels_path_test',
                        type=str,
                        dest='labels_path_test',
                        default=None,
                        help='Path of the csv file with test labels.\n'
                             'Actually they do not exist.\n'
                             'We use empty lists.')

    parser.add_argument('--model_dir',
                        type=str,
                        dest='model_dir',
                        default=None,
                        help='Directory where trained models will be saved.')

    parser.add_argument('--size_test',
                        type=int,
                        dest='size_test',
                        default=None,
                        help='Number of images to predict.\n'
                             'Used only to test the algorithm with a small amount of data.')

    parser.add_argument('--batch_size',
                        type=int,
                        dest='batch_size',
                        default=4,
                        help='Batch size')

    parser.add_argument('--load_epoch',
                        type=int,
                        dest='load_epoch',
                        default=None,
                        help='Continue training by loading a model from a given epoch.')

    args = parser.parse_args()

    predict_v1(predictions_dir=args.predictions_dir,
               predictions_file=args.predictions_file,
               img_dir_test=args.img_dir_test,
               labels_path_test=args.labels_path_test,
               model_dir=args.model_dir,
               num_classes=5,
               size_test=args.size_test,
               batch_size=args.batch_size,
               load_epoch=args.load_epoch)
