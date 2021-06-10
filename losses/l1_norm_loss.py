from typing import Dict

import torch
from torch import nn


def l1_norm_loss(output: torch.tensor, target: torch.tensor, epoch: int, config: Dict, changing_epochs_num: int, epsilon: float = 1e-10):
    start_alpha = config['start_alpha']
    end_alpha = config['end_alpha']
    if epoch >= changing_epochs_num:
        alpha_value = end_alpha
    else:
        alpha_value = start_alpha - ((start_alpha-end_alpha) * (epoch / changing_epochs_num))

    l1_loss = nn.L1Loss(reduction='none')
    l1_loss_value = torch.pow(l1_loss(output, target) + epsilon, alpha_value).mean()
    return l1_loss_value
