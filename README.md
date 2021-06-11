# Orientation-based Token Embedding Compression

Official PyTorch implementation of the paper 
[Direction is what you need: Improving Word Embedding Compression in Large Language Models]().

## Requirements
Please install needed packages by running `pip install -r requirements.txt`. Please note that for SQuAD experiment,
you need a different version of huggingface Transformers (3.5.0).

## Abstract

The adoption of Transformer-based models in natural language processing (NLP) has led to great success using a massive number of parameters.
However, due to deployment constraints in edge devices, there has been a rising interest in the compression of these models to improve their inference time and memory footprint.
This paper presents a novel loss objective to compress token embeddings in the Transformer-based models by leveraging an AutoEncoder architecture.
More specifically, we emphasize the importance of the direction of compressed embeddings with respect to original uncompressed embeddings.
The proposed method is task-agnostic and does not require further language modeling pre-training.
Our method significantly outperforms the commonly used SVD-based matrix-factorization approach in terms of initial language model Perplexity.
Moreover, we evaluate our proposed approach over SQuAD v1.1 dataset and several downstream tasks from the GLUE benchmark, where we also outperform the baseline in most scenarios.

## Experiments
This repository provides a code for compressing token embedding matrix of transformer-based architectures. There are two methods
for compression studies here:

1) Compression by a multi-objective loss (which enforces orientation similarity to input embeddings) with an AutoEncoder architecture
2) Singular Value Decomposition (SVD) baseline for low-rank matrix factorzation of the token embeddings


We also provide a script to use these compressed token embeddings in different down-stream tasks
(GLUE benchmark and SQuAD datasets)

### Compressed Embeddings Generation
Please use the following command in order to generate compressed token embeddings.
```
python main.py
```
The aforementioned command will read the config file located at `configs/config.yaml`. Please update the output paths and
hyper-parameters that you would like to use by updating this configuration file.


### SQuAD experiments
To run the SQuAD training, run from `transformer_experiments/Question_Answering_experiment` directory with the following commands:

```
export PYTHONPATH=.:repo_root_directory
python run_squad_script.py  --config qa_cfg.json
```
Please update the correct paths and your intended hyper-parameters in `qa_cfg.json` in the same directory.


### GLUE experiments
To run different fine-tuning experiments from GLUE benchmark, 
run from `transformer_experiments/GLUE_experiments` directory with the following commands:

```
export PYTHONPATH=.:repo_root_directory
python ablation_study_glue.py  --config glue_cfg.json
```
Please update the paths, `task_name` and your intended hyper-parameters in `glue_cfg.json` in the same directory.


## Reference

If you found the provided code useful, please consider citing our work.
