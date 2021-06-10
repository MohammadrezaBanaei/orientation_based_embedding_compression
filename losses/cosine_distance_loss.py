import torch
from torch import nn


def cosine_distance_loss(output: torch.tensor, target: torch.tensor, epoch: int):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_loss = 1.0 - cos(output, target)
    cos_loss = cos_loss.mean()
    return cos_loss
