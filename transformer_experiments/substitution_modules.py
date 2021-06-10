from typing import Tuple

import torch
from torch import nn

from utils_funcs import (get_ae_sub_compression_stats,
                         get_svd_compression_stats,
                         get_lin_rec_latent_dim,
                         run_svd)


class SubstitutionModule(nn.Module):

    def __init__(self, pad_token_id: int, original_params_num: int):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.original_params_num = original_params_num

    def forward(self, input: torch.tensor) -> torch.tensor:
        raise NotImplementedError

    def get_compression_stats(self) -> Tuple[float, float]:
        raise NotImplementedError


class WordEmbeddingModule(SubstitutionModule):

    def __init__(self,
                 decoder_module: nn.Module,
                 emb_weights: torch.Tensor,
                 pad_token_id: int,
                 original_params_num: int):
        super().__init__(pad_token_id=pad_token_id, original_params_num=original_params_num)

        self.decoder_module = decoder_module
        self.embedding_module = nn.Embedding.from_pretrained(emb_weights, padding_idx=self.pad_token_id)

    def forward(self, input: torch.tensor) -> torch.tensor:
        inputs_embeds = self.embedding_module(input)

        out = self.decoder_module(inputs_embeds)

        return out

    def get_compression_stats(self) -> float:
        cr = get_ae_sub_compression_stats(original_params_num=self.original_params_num,
                                          embedding=self.embedding_module.weight,
                                          decoder=self.decoder_module)
        return cr


class MFWordEmbeddingModule(SubstitutionModule):

    def __init__(self,
                 reduced_matrix: torch.tensor,
                 second_matrix: torch.tensor,
                 pad_token_id: int,
                 original_params_num: int):
        super().__init__(pad_token_id=pad_token_id, original_params_num=original_params_num)

        self.embedding_module = nn.Embedding.from_pretrained(reduced_matrix, padding_idx=self.pad_token_id)
        self.second_matrix = second_matrix

    def forward(self, input: torch.tensor) -> torch.tensor:
        inputs_embeds = self.embedding_module(input)
        svd_w = self.second_matrix

        result = torch.mm(inputs_embeds.view(-1, inputs_embeds.shape[2]), svd_w)
        result = result.view(-1, inputs_embeds.shape[1], self.second_matrix.shape[1])

        return result

    def get_compression_stats(self) -> float:
        return get_svd_compression_stats(orginal_params_num=self.original_params_num,
                                         reduced_matrix_numel=self.embedding_module.weight.numel(),
                                         second_matrix_numel=self.second_matrix.numel())

    @staticmethod
    def get_svd_emb_module(input_matrix: torch.tensor,
                           compression_ratio: float,
                           n_iter: int,
                           random_state: int,
                           device: torch.device,
                           pad_token_id: int) -> SubstitutionModule:
        original_params_num = input_matrix.numel()
        latent_dim = get_lin_rec_latent_dim(compression_ratio, input_matrix)

        reduced_matrix, svd = run_svd(input_matrix=input_matrix.cpu().numpy(),
                                      latent_dim=latent_dim,
                                      n_iter=n_iter,
                                      random_state=random_state)

        reduced_matrix = torch.tensor(reduced_matrix, dtype=input_matrix.dtype, device=device)
        second_matrix = torch.tensor(svd.components_, dtype=input_matrix.dtype, device=device)

        svd_word_emb_module = MFWordEmbeddingModule(reduced_matrix=reduced_matrix,
                                                    second_matrix=second_matrix,
                                                    pad_token_id=pad_token_id,
                                                    original_params_num=original_params_num)
        return svd_word_emb_module
