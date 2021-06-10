import os
from typing import Dict, List

import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from tqdm import tqdm

from evaluation import eval_substitution_emb_module


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def log_epoch_info(writer: SummaryWriter, optimizer: _LRScheduler, epoch: int):
    writer.add_scalar('Training_info/lr', get_lr(optimizer), epoch)


def log_iteration(writer: SummaryWriter, multi_obj_losses: Dict, iter: int):
    for loss_name, (coeff, loss_value) in multi_obj_losses.items():
        writer.add_scalar(f'Raw_losses/{loss_name}', loss_value, iter)
        writer.add_scalar(f'Losses_coeffs/{loss_name}', coeff, iter)
        writer.add_scalar(f'Loss_times_coeff/{loss_name}', loss_value * coeff, iter)


def save_state(checkpoints_dir: str, model: nn.Module, optimizer: Optimizer, embedding_latents: torch.tensor):
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model.pt'))
    torch.save(optimizer.state_dict(), os.path.join(checkpoints_dir, 'optimizer.pt'))
    torch.save(embedding_latents, os.path.join(checkpoints_dir, 'latents.pt'))


def train_model(checkpoints_dir: str, epochs: int, additional_epochs: int, model: nn.Module, data_loader: DataLoader,
                validation_dataset: Dataset, writer: SummaryWriter, optimizer: Optimizer,
                scheduler: _LRScheduler, losses: List, mlm_trainer: transformers.trainer.Trainer, seed: int,
                device: torch.device):
    best_early_stopping_obj_score = None
    iter = 0
    model.train()
    for epoch in range(0, epochs + additional_epochs):

        log_epoch_info(writer, optimizer, epoch)

        with tqdm(data_loader, unit="batch") as tepoch:
            for batch, ids in tepoch:
                batch = batch.to(device)
                optimizer.zero_grad()

                out, out_latents = model(batch)

                multi_obj_losses = {loss_name: (coeff, loss_func(output=out, target=batch, epoch=epoch)) for loss_name, (coeff, loss_func) in losses.items()}

                log_iteration(writer, multi_obj_losses, iter)

                final_loss = sum([coeff * loss_value for loss_name, (coeff, loss_value) in multi_obj_losses.items()])

                final_loss.backward()
                optimizer.step()

                iter += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            reconstructed_embedding, embedding_latents = model(validation_dataset.matrix.to(device))
            early_stopping_obj_score = eval_substitution_emb_module(original_embedding=validation_dataset.matrix,
                                                                    reconstructed_embedding=reconstructed_embedding.detach().cpu(),
                                                                    writer=writer, global_step=epoch,
                                                                    mlm_trainer=mlm_trainer, seed=seed, device=device)
        model.train()

        if best_early_stopping_obj_score is None or early_stopping_obj_score < best_early_stopping_obj_score:
            best_early_stopping_obj_score = early_stopping_obj_score
            save_state(checkpoints_dir=checkpoints_dir, model=model, optimizer=optimizer, embedding_latents=embedding_latents)
