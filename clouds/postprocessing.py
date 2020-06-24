import torch
import torch.nn.functional as nnf
import numpy as np
from typing import List, Dict, Tuple, Union, Any


def torch_reduce_predictions(prediction: Dict,
                             top_n: int = 5,
                             threshold_score: float = 0.7) -> Dict:
    """ Reduce the number of the model predictions.

    By default the model predicts the location of ~100 objects (bboxes, masks, etc)
    ordered by the confidence of the prediction. We reduce this number to at most `top_n`.

    :param prediction: prediction of the model; contains 100 objects
    :param top_n: maximum number of objects that we are going to keep
    :param threshold_score: minimum confidence that is required for the prediction
    :return:
    """

    scores = prediction['scores']

    # keep only those scores that are above a certain threshold
    # keep at most `top_n` scores
    # if no score is above `threshold` keep only the best score
    if any(scores > threshold_score):
        scores = scores[scores > threshold_score][:top_n]
    else:
        scores = torch.tensor([0], dtype=scores.dtype)

    max_n = len(scores)

    labels = prediction['labels'][:max_n]

    # reduce the size of the masks to (n, 1, 350, 525)
    masks = nnf.interpolate(prediction["masks"][:max_n],
                            size=(350, 525),
                            mode='nearest')

    # reshape (n, 1, 350, 525) -> (n, 350, 525)
    masks = masks.sum(axis=1)

    return {'scores': scores, 'labels': labels, 'masks': masks}


def torch_mask_to_encoded_pixels(mask: torch.Tensor) -> torch.Tensor:
    """ Map 2D mask to 1D list of encoded pixels

    :param mask: 2D torch.Tensor
    :return:
        1D torch.Tensor
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # reshape by starting with the elements of the first column, second column, etc.
    mask = torch.transpose(mask, -2, -1).reshape(-1)
    mask = torch.cat((torch.tensor([0], dtype=mask.dtype, device=device),
                      mask,
                      torch.tensor([0], dtype=mask.dtype, device=device)))

    hits = mask[1:] != mask[:-1]
    enc_px = torch.arange(hits.size()[0])[hits]

    enc_px[1::2] -= enc_px[::2]
    enc_px[::2] += 1

    return enc_px


def np_mask_to_encoded_pixels(mask: np.ndarray):
    """ Map 2D mask to 1D list of encoded pixels

    :param mask: 2D numpy array
    :return:
    """
    # reshape by starting with the elements of the first column, second column, etc.
    mask = mask.transpose(1, 0).reshape(-1)
    mask = np.concatenate(([0], mask, [0]))

    hits = mask[1:] != mask[:-1]

    enc_px = np.arange(len(hits))[hits]
    enc_px[1::2] -= enc_px[::2]
    enc_px[::2] += 1

    return enc_px


def torch_get_encoded_predictions_all_classes_single_image(img_name: str,
                                                           masks: torch.Tensor,
                                                           labels: torch.Tensor,
                                                           threshold_mask: float = 0.5):
    """ Encode the prediction for the presence of a single clouds class in a single image
     to the expected kaggle format.

    :param masks:
        torch.Tensor (probabilities that an object is present) of size (# predictions, 350, 525)
    :param labels:
        torch.Tensor of size (# predictions)
        the prediction labels that can take the following values: {1,2,3,4}
    :param threshold_mask:
        threshold used to convert the masks into binary masks
    :param img_name:
        the name of the image used to make a prediction
    :return:
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    classmap = {
        'Sugar': torch.tensor(1).to(device),
        'Gravel': torch.tensor(2).to(device),
        'Flower': torch.tensor(3).to(device),
        'Fish': torch.tensor(4).to(device)
    }

    # covert mask to binary mask
    masks = (masks >= torch.tensor(threshold_mask)).type(torch.int8)

    enc_predictions = []

    for k, v in classmap.items():
        mask_label = masks[labels == v].sum(axis=0).clamp(0, 1)

        # map the mask to a tensor of encoded pixels
        encoded_pixels = torch_mask_to_encoded_pixels(mask_label)

        encoded_pixels = ' '.join(encoded_pixels.numpy().astype(str))

        enc_prediction = f"{img_name}_{k}, {encoded_pixels}"

        enc_predictions.append(enc_prediction)

    return enc_predictions


def get_encoded_predictions_single_image(
        img_name: str,
        masks,
        threshold_mask: float = 0.):
    enc_predictions = []
    classes = ['Sugar', 'Gravel', 'Flower', 'Fish']

    # covert mask to binary mask
    masks = (masks >= torch.tensor(threshold_mask)).type(torch.int8)

    for i, class_name in enumerate(classes):
        encoded_px = torch_mask_to_encoded_pixels(masks[i])
        encoded_px = ' '.join(encoded_px.numpy().astype(str))
        enc_prediction = f"{img_name}_{class_name}, {encoded_px}"
        enc_predictions.append(enc_prediction)

    return enc_predictions
