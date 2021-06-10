import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(self, input: torch.tensor):
        raise NotImplementedError


class LinearEncoder(Encoder):
    def __init__(self, input_dim: int, latent_dim: int):
        super(LinearEncoder, self).__init__(input_dim=input_dim, latent_dim=latent_dim)
        self.enc = nn.Linear(in_features=input_dim, out_features=latent_dim)

    def forward(self, input: torch.tensor):
        return self.enc(input)


class NonLinearEncoder(Encoder):
    def __init__(self, input_dim: int, latent_dim: int):
        super(NonLinearEncoder, self).__init__(input_dim=input_dim, latent_dim=latent_dim)
        self.enc = nn.Linear(in_features=input_dim, out_features=latent_dim)
        self.activation = nn.ELU()

    def forward(self, input: torch.tensor):
        return self.activation(self.enc(input))
