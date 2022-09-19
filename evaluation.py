import math

import torch
import transformers
from tensorboardX import SummaryWriter
from transformers import set_seed


def compute_perplexity(mlm_trainer: transformers.trainer.Trainer, reconstructed_embedding: torch.tensor, seed: int):
    mlm_trainer.model.bert.embeddings.word_embeddings.weight.copy_(reconstructed_embedding)
    set_seed(seed)
    eval_output = mlm_trainer.evaluate()
    perplexity = math.exp(eval_output["eval_loss"])
    return perplexity


def eval_substitution_emb_module(original_embedding: torch.tensor,
                                 reconstructed_embedding: torch.tensor,
                                 writer: SummaryWriter,
                                 global_step: int,
                                 mlm_trainer: transformers.trainer.Trainer,
                                 seed: int,
                                 device: torch.device) -> float:
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    cosine_distance = (1.0 - cos(reconstructed_embedding, original_embedding)).mean()
    rmse = torch.sqrt(torch.pow((original_embedding - reconstructed_embedding), 2).mean()).cpu()
    mae = (original_embedding - reconstructed_embedding).abs().mean().cpu()
    powered_mae = torch.pow(torch.abs(original_embedding - reconstructed_embedding) + 1e-10, 0.6).mean()
    perplexity = compute_perplexity(mlm_trainer, reconstructed_embedding.to(device), seed)

    writer.add_scalar(tag='Metrics/cosine_distance', scalar_value=cosine_distance, global_step=global_step)
    writer.add_scalar(tag='Metrics/rmse', scalar_value=rmse, global_step=global_step)
    writer.add_scalar(tag='Metrics/mae', scalar_value=mae, global_step=global_step)
    writer.add_scalar(tag='Metrics/powered_mae', scalar_value=powered_mae, global_step=global_step)
    writer.add_scalar(tag='Metrics/perplexity', scalar_value=perplexity, global_step=global_step)

    return perplexity


