import torch
from torch import nn


def l2_norm_loss(output: torch.tensor, target: torch.tensor, epoch: int):
    mse_loss = nn.MSELoss(reduction='mean')
    rmse_loss_value = torch.sqrt(mse_loss(output, target))
    return rmse_loss_value
