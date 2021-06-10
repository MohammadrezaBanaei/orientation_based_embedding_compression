import os
from typing import List

import torch
from torch.utils.data import Dataset


class TokenEmbeddingDataset(Dataset):
    def __init__(self, matrix: torch.tensor, token_ids: List):
        self.matrix = matrix
        self.token_ids = token_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return self.matrix[idx], self.token_ids[idx]
