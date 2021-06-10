import math
import os
from typing import Tuple, Iterable, List, Dict

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from collections import Counter
from datasets import load_dataset

from evaluation import eval_substitution_emb_module
from zeroshot_mlm_utils import get_transformer_mlm_trainer


def get_layer_params_num(layer: nn.Linear) -> int:
    params_num = layer.weight.numel() + layer.bias.numel()
    return params_num


def get_mlm_trainer(config: Dict, seed: int) -> transformers.trainer.Trainer:
    text_download_folder = config["text_download_folder"]
    model_name = config["model_name"]
    text_dataset_path = config["LM_text_dataset_path"]

    if not (os.path.isfile(text_dataset_path)):
        os.makedirs(text_download_folder, exist_ok=True)
        assert os.system("wget %s -P %s" % (config['wiki_text_103_url'], text_download_folder)) == 0, \
            "Downloading text dataset  failed"
        assert os.system("unzip %s -d %s" % (os.path.join(text_download_folder, "wikitext-103-raw-v1.zip"),
                                             text_download_folder)) == 0, "unzip of text dataset failed"
        text_dataset_path = os.path.join(text_download_folder, "wikitext-103-raw", "wiki.test.raw")

    mlm_trainer = get_transformer_mlm_trainer(eval_data_path=text_dataset_path, seed=seed, model_name=model_name)
    return mlm_trainer


def run_svd(input_matrix: np.ndarray, latent_dim: int, n_iter: int, random_state: int) -> Tuple[
    np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=latent_dim, n_iter=n_iter, random_state=random_state)
    svd.fit(input_matrix)
    reduced_matrix = svd.transform(input_matrix)
    return reduced_matrix, svd


def get_linear_rec_svd(input_matrix: np.ndarray, latent_dim: int, n_iter: int, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced_matrix, svd = run_svd(input_matrix, latent_dim, n_iter, random_state)

    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    return reconstructed_matrix, reduced_matrix, svd.components_


def get_lin_rec_latent_dim(compression_ratio: float, input_matrix: np.ndarray) -> int:
    original_params_num = input_matrix.shape[0] * input_matrix.shape[1]
    return int(original_params_num / (compression_ratio * (input_matrix.shape[0] + input_matrix.shape[1])))


def save_lin_rec_stats(writer: SummaryWriter,
                       original_matrix: torch.tensor,
                       compression_ratio: float,
                       iters: Iterable,
                       mlm_trainer: transformers.trainer.Trainer,
                       seed: int,
                       device: torch.device):
    input_matrix = original_matrix.cpu().numpy()
    latent_dim = get_lin_rec_latent_dim(compression_ratio, input_matrix)

    for i in iters:
        lin_rec, reduced_matrix, svd_comp = get_linear_rec_svd(input_matrix=input_matrix,
                                                               latent_dim=latent_dim,
                                                               n_iter=i,
                                                               random_state=42)
        lin_rec_tensor = torch.tensor(lin_rec, dtype=original_matrix.dtype, device=original_matrix.device)

        _ = eval_substitution_emb_module(original_embedding=original_matrix, reconstructed_embedding=lin_rec_tensor,
                                         writer=writer, mlm_trainer=mlm_trainer, global_step=i, seed=seed, device=device)

        svd_cr = get_svd_compression_stats(orginal_params_num=input_matrix.size,
                                           reduced_matrix_numel=reduced_matrix.size,
                                           second_matrix_numel=svd_comp.size)

        writer.add_scalar('Training_info/Compression_Ratio', svd_cr, i)


def get_ae_sub_compression_stats(original_params_num: int,
                                 embedding: torch.Tensor,
                                 decoder: nn.Module) -> float:
    embedding_params_num = embedding.numel()
    dec_params_num = decoder.get_params_num()
    current_params_num = embedding_params_num + dec_params_num
    cr = original_params_num / current_params_num
    return cr


def get_svd_compression_stats(orginal_params_num: int, reduced_matrix_numel: int, second_matrix_numel: int) -> float:
    cr = orginal_params_num / (reduced_matrix_numel + second_matrix_numel)
    return cr


def init_model_with_svd(input_matrix: np.ndarray, model: nn.Module, num_iters: int, device: torch.device) -> nn.Module:
    _, svd = run_svd(input_matrix=input_matrix,
                     latent_dim=model.latent_size,
                     n_iter=num_iters,
                     random_state=42)

    dtype = model.encoder.enc.weight.dtype

    model.encoder.enc.weight.data = torch.tensor(svd.components_, device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.zeros_(model.encoder.enc.bias.data)
    model.decoder.dec.weight.data = torch.tensor(svd.components_.T, device=device, dtype=dtype, requires_grad=True)
    torch.nn.init.zeros_(model.decoder.dec.bias.data)

    return model
