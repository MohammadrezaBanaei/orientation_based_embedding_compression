from typing import Tuple

import torch
import torch.nn as nn

from utils_funcs import get_layer_params_num


class Decoder(nn.Module):
    def __init__(self, out_dim: int, latent_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.latent_dim = latent_dim

    def forward(self, input: torch.tensor):
        raise NotImplementedError

    def get_params_num(self) -> int:
        raise NotImplementedError


class LinearDecoder(Decoder):
    def __init__(self, out_dim: int, latent_dim: int):
        super(LinearDecoder, self).__init__(out_dim=out_dim, latent_dim=latent_dim)
        self.dec = nn.Linear(in_features=latent_dim, out_features=out_dim)

    def forward(self, input: torch.tensor):
        return self.dec(input)

    def get_params_num(self) -> int:
        return get_layer_params_num(self.dec)


class NonLinearDecoder(Decoder):
    def __init__(self, out_dim: int, latent_dim: int):
        super(NonLinearDecoder, self).__init__(out_dim=out_dim, latent_dim=latent_dim)
        self.dec = nn.Linear(in_features=latent_dim, out_features=out_dim)
        self.activation = nn.ELU()

    def forward(self, input: torch.tensor):
        return self.activation(self.dec(input))

    def get_params_num(self) -> int:
        return get_layer_params_num(self.dec)
