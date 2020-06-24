import torch
import torchvision
import pretrainedmodels
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from clouds.frcnn.myclasses import my_maskrcnn_resnet50_fpn


def get_colors():
    colors = {
        'Sugar': (128, 0, 0, 255),
        'Gravel': (0, 128, 0, 255),
        'Flower': (0, 0, 128, 255),
        'Fish': (128, 0, 128, 255)
    }
    return colors


def see_colors_map():
    colors = get_colors()

    img = Image.new(mode='RGBA', size=(240, 30), color=(0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    for idx, (k, v) in enumerate(colors.items()):
        draw.text(xy=(10 + 60 * idx, 10), text=k, fill=v)

    plt.figure(figsize=(8, 3))
    plt.imshow(img)
    plt.pause(0.001)
    plt.show()


class MyBackbone(object):
    """ A wrapper of a `.features` method of a neural network.

    It adds an .out_channels property that is required for all backbones used in
    MaskRCNN / FastRCNN.
    """

    def __init__(self, backbone, out_channels):
        self.backbone = backbone
        self.out_channels = out_channels

    def __call__(self, input):
        return self.backbone(input)


def get_model_instance_segmentation(num_classes, use_my_model=False):
    """ Load pre-trained model.
    Replace the box and mask predictors with new predictors with
    whose outputs are adequate to the `num_classes`.
    """
    # load an instance segmentation model pre-trained pre-trained on COCO
    if use_my_model is True:
        model = my_maskrcnn_resnet50_fpn(pretrained=True)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_model_instance_segmentation_v2(num_classes, architecture: str = 'resnet18'):
    """ By modifying this function we will be able to use a large variety of
    pretrained backbones but besides the backbones nothing else will be trained.

    A better solution seems to be to load a pre-trained model and then to
    change the mask and box predictors.
    """

    # Pretrained model for num_classes=1000, but we will not use the final layers anyway.
    model = pretrainedmodels.__dict__[architecture](num_classes=1000, pretrained='imagenet')

    my_backbone = MyBackbone(model.features, 512)

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                         output_size=14,
                                                         sampling_ratio=2)

    model = MaskRCNN(my_backbone,
                     num_classes=num_classes,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler,
                     mask_roi_pool=mask_roi_pooler)

    return model


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def collate_fn(batch):
    return tuple(zip(*batch))
