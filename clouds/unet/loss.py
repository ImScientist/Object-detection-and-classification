import torch
import torch.nn.functional as nnf


def dice_loss(input_, target):
    smooth = 1.0
    input_ = torch.sigmoid(input_)
    numerator = (input_ * target).sum()
    denominator = input_.sum() + target.sum()

    return 1 - (2.0 * numerator + smooth) / (denominator + smooth)


class FocalLoss(torch.nn.Module):
    """ Implement the following loss function
    y = +/- (+ positive class; - negative class)

    loss_binary = - ln ( sigmoid( y * x ) )

    sigmoid( x ) = 1 / (1 + exp( - x ) )

    loss_focal = sigmoid^gamma( - y * x ) * loss_binary
                = exp( gamma * logsigmoid( - y * x) ) * loss_binary

    """

    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input_, target):
        if not (target.size() == input_.size()):
            raise ValueError("Target size ({}) and input size ({}) must be the same"
                             .format(target.size(), input_.size()))

        yx = (target * 2.0 - 1.0) * input_
        loss = - nnf.logsigmoid(yx)
        loss_focal = (self.gamma * nnf.logsigmoid(-yx)).exp() * loss

        return loss_focal.mean()


class MixedLoss(torch.nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input_, target):
        loss = self.alpha * self.focal(input_, target) \
               - dice_loss(input_, target)
        return loss
