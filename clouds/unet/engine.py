import sys
import math
import torch
from ..logger.logger import MetricLogger, SmoothedValue
from ..utils import warmup_lr_scheduler


def train_one_epoch(
        model,
        optimizer,
        criterion,
        data_loader,
        device,
        epoch,
        print_freq=10
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    for idx, param_gr in enumerate(optimizer.param_groups):
        metric_logger.add_meter(
            f'lr_{idx}', SmoothedValue(window_size=1, fmt='{value:.6f}')
        )
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = images.to(device)
        targets = targets.to(device)

        output = model(images)
        loss = criterion(output, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss)
        metric_logger.update(
            **dict((f'lr_{idx}', el['lr']) for idx, el in enumerate(optimizer.param_groups))
        )

    return metric_logger


@torch.no_grad()
def evaluate_one_epoch(model, criterion, data_loader, device, epoch, print_freq=10):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device)
        targets = targets.to(device)

        output = model(images)
        loss = criterion(output, targets)

        metric_logger.update(loss=loss)

    return metric_logger
