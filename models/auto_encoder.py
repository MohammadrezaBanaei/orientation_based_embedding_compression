import torch
import torch.nn as nn

from models.decoder import Decoder
from models.encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, weights_path: str = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))

    def forward(self, input: torch.tensor) -> torch.tensor:
        latents = self.encoder(input)
        cloned_latents = latents.clone()
        out = self.decoder(latents)
        return out, cloned_latents
