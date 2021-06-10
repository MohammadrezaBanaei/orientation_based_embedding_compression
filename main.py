import os
from functools import partial
from typing import List, Dict

import yaml
import torch
import random
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import transformers_data_utils
from ae_train import train_model
from custom_dataset import TokenEmbeddingDataset
from losses.cosine_distance_loss import cosine_distance_loss
from losses.l1_norm_loss import l1_norm_loss
from losses.l2_norm_loss import l2_norm_loss
from models.auto_encoder import AutoEncoder
from models.decoder import LinearDecoder, NonLinearDecoder
from models.encoder import LinearEncoder, NonLinearEncoder
from utils_funcs import init_model_with_svd, get_mlm_trainer, save_lin_rec_stats, get_ae_sub_compression_stats


def set_global_seed(seed_value: int):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def create_dirs_if_needed(dirs: List):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)


def init_ae_model(config: Dict, emb_dim: int, device: torch.device):
    latent_dim = config['model']['latent_dim']
    if config['encoder']['is_linear']:
        encoder = LinearEncoder(input_dim=emb_dim, latent_dim=latent_dim)
    else:
        encoder = NonLinearEncoder(input_dim=emb_dim, latent_dim=latent_dim)

    if config['decoder']['is_linear']:
        decoder = LinearDecoder(out_dim=emb_dim, latent_dim=latent_dim)
    else:
        decoder = NonLinearDecoder(out_dim=emb_dim, latent_dim=latent_dim)

    model = AutoEncoder(encoder=encoder, decoder=decoder, weights_path=config['model']['weights_path'])
    model.to(device)
    return model


if __name__ == '__main__':
    stream = open("configs/config.yaml", 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)

    main_dir = config["paths"]["main_dir"]
    exp_dir = os.path.join(main_dir, config["paths"]["experiment_name"])
    log_dir = os.path.join(exp_dir, 'logs')
    checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
    create_dirs_if_needed([main_dir, exp_dir, log_dir, checkpoints_dir])

    seed_value = config["global"]["seed"]
    set_global_seed(seed_value=seed_value)

    transformer_data = transformers_data_utils.get_model_weight_dict()
    training_data = transformer_data[config["dataset"]["input_matrix_name"]][config["dataset"]["input_matrix_subname"]]
    token_emb_dataset = TokenEmbeddingDataset(matrix=training_data,
                                              token_ids=[i for i in range(training_data.shape[0])])

    loader = DataLoader(token_emb_dataset, batch_size=config['training']['batch_size'], shuffle=True,
                        num_workers=0, pin_memory=False)

    emb_dim = token_emb_dataset.matrix.size(1)
    vocab_size = token_emb_dataset.matrix.size(0)
    latent_dim = config['model']['latent_dim']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = init_ae_model(config, emb_dim=emb_dim, device=device)

    if config['model']['svd']['init_ae_with_svd']:
        model = init_model_with_svd(input_matrix=training_data.cpu().detach().numpy(), model=model,
                                    num_iters=config['model']['svd']['svd_iters'], device=device)

    # Training
    optimizer = Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = StepLR(optimizer,
                       step_size=config['training']['step_lr_scheduler']['step_size'],
                       gamma=config['training']['step_lr_scheduler']['gamma'])

    mlm_trainer = get_mlm_trainer(config['dataset']['lm_dataset'], seed=config["global"]["seed"])

    cr = token_emb_dataset.matrix.numel() / (vocab_size * latent_dim + latent_dim * emb_dim)

    if config['svd_run']['activated']:
        svd_writer = SummaryWriter(os.path.join(log_dir, 'svd'))
        save_lin_rec_stats(writer=svd_writer, original_matrix=token_emb_dataset.matrix, compression_ratio=cr,
                           iters=range(1, config['svd_run']['max_iters']), mlm_trainer=mlm_trainer,
                           seed=config["global"]["seed"], device=device)

    if config['ae_run']['activated']:
        ae_writer = SummaryWriter(os.path.join(log_dir, 'ae'))

        losses_cfg = config['ae_run']['loss']
        losses = {}
        if losses_cfg['cos_dist']['coeff'] > 0.0:
            losses['cos_dist'] = (
                losses_cfg['cos_dist']['coeff'] / losses_cfg['cos_dist']['scaler_div'], cosine_distance_loss)
        if losses_cfg['l2_norm']['coeff'] > 0.0:
            losses['l2_norm'] = (losses_cfg['l2_norm']['coeff'], l2_norm_loss)
        if losses_cfg['l1_norm']['coeff'] > 0.0:
            losses['l1_norm'] = (losses_cfg['l1_norm']['coeff'], partial(l1_norm_loss,
                                                                         config=losses_cfg['l1_norm'],
                                                                         changing_epochs_num=config['training']['epochs']))
    train_model(checkpoints_dir=checkpoints_dir,
                epochs=config['training']['epochs'],
                additional_epochs=config['training']['additional_epochs'],
                model=model,
                data_loader=loader,
                validation_dataset=token_emb_dataset,
                writer=ae_writer,
                optimizer=optimizer,
                scheduler=scheduler,
                losses=losses,
                mlm_trainer=mlm_trainer,
                seed=config["global"]["seed"],
                device=device)

    with open(os.path.join(exp_dir, 'config.yml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
