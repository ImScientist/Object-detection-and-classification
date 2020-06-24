# Object detection and classification


We modify the architecture from the object detection example in a
[pytorch tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
by adding an evaluation function for the test data and by using a better
data augmentation layer. The model was used in the
[Understanding Clouds from Satellite Images](
https://www.kaggle.com/c/understanding_cloud_organization) Kaggle
competition.

## Quick start (WIP)
Set environment variables in `.env`: 
```bash
source .env
```
Setup the environment: 
```bash
pip install -r requirements.txt
python setup.py install
```
Generate a dummy `test.csv` that is required by the dataloader. 
```bash
python genera_test_csv.py \
    --data_dir /content/drive/My Drive/data/source/clouds
```
Train one of the three training models (`exec/train_v1.py`,
`exec/train_v2.py`, `exec/train_v3.py`):
```bash
python exec/train_v3.py \
        --size_tr_val 20 \
        --size_val 8 \
        --batch_size 2 \
        --print_freq 2 \
        --num_epochs 3 \
        --seed 1
```
To make a prediction we have created a dummy `test.csv` file that has
the same structure as the `train.csv` file. It is created in order to
use the same dataloader for training and for making predictions. To make
a prediction use `exec/predict_v1.py`, `exec/predict_v2.py` or  `exec/predict_v3.py`:
```bash 
python exec/predict_v3.py \
        --nrows 10 \
        --load_epoch 2
```

Docker image (WIP)


<details>
    <summary>Faster RCNN Architecture </summary>

The examples (`train_v1.py`, `train_v2.py`) use a Faster RCNN with a pretrained Resnet50 as
a backbone. The model detects objects, masks and bounding boxes. Since
the training data provided in the Kaggle competition contains only masks
of four different object types and no bounding boxes we have used an
algorithm that detects non-connected regions from the masks and assigns
to each of them a bounding box (this just seemed easier than modifying
the loss function of the model).

The original model has several weaknesses:
- The loss function depends on the bounding box loss which is irrelevant
  for the current task (WIP).
- We use only random horizontal and vertical image flips to augment the 
  data.
- The default `evaluate()` function can not run on a GPU. We do not know
  if the model is overfitting.


#### 1 Add `evaluate()` function for the test data

The default `forward` function of the used classes has different output 
that depends on whether the model is in `train` or `eval` mode. In 
`eval` mode the losses are not calculated. In the current implementation
we have derived new classes from:
 - `torchvision.models.detection.rpn.RegionProposalNetwork`
 - `torchvision.models.detection.roi_heads.RoIHeads`
 - `torchvision.models.detection.generalized_rcnn.GeneralizedRCNN`    
 
which have a modified `forward()` method with an additional argument
`return_loss=False` that allows to return the losses in `eval` mode.
Look at [/clouds/myclasses.py](/clouds/frcnn/myclasses.py) for the new class
definitions.

This function is used in both `train.py` and `train_v2.py`.


#### 2 Better data augmentation

We have used the [albumeration](https://github.com/albu/albumentations) 
library for data augmentation. The library takes care of all 
transformations of the masks and bounding boxes. The `Dataset` 
defined in `/clouds/dataset_v2.py` was modified to take into account
data transformations from this library.

Bounding box formats:
 - `coco`: [x_min, y_min, width, height]
 - `pascal_voc`: [x_min, y_min, x_max, y_max]   

The data augmentation is used only in `train_v2.py`.
</details>

<details>
    <summary>Segmentation model Architecture (preferred) </summary>
    
The example `train_v3.py` makes use of the segmentation models library. 
In this case we only work with different the U-shaped convolutional neural 
networks which previously were representing only the backbone of the RCNN,
i.e. no region proposal network, no RoI heads for bounding boxes and masks.

</details>
