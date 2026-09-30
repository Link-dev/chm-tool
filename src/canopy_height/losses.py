"""Training losses. Label cells equal to -999 are missing and ignored by the Charbonnier loss."""
import torch
import torch.nn.functional as F


def charbonnier(x, y, eps=1e-9):
    x = x.flatten()
    y = y.flatten()
    mask = y != -999
    diff = x[mask] - y[mask]
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def gradient_x(img):
    return img[:, :, :, 1:] - img[:, :, :, :-1]


def gradient_y(img):
    return img[:, :, 1:, :] - img[:, :, :-1, :]


def edge_aware_gradient_loss(prediction, target, weight=1.0):
    """L1 distance between horizontal and vertical image gradients (structure from the ALS teacher)."""
    loss_x = F.l1_loss(gradient_x(prediction), gradient_x(target))
    loss_y = F.l1_loss(gradient_y(prediction), gradient_y(target))
    return weight * (loss_x + loss_y)


def avgpool_loss(pred, target, k):
    """Charbonnier loss after k x k average pooling (coarse heights from the GEDI teacher)."""
    return charbonnier(F.avg_pool2d(pred, kernel_size=k, stride=k),
                       F.avg_pool2d(target, kernel_size=k, stride=k))

