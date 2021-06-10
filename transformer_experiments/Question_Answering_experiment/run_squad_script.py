import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    dataset_dir = config['dataset_dir']
    os.makedirs(dataset_dir, exist_ok=True)

    train_url = config['train_url']
    dev_url = config['dev_url']
    test_url = config['test_url']

    for download_link in [train_url, dev_url, test_url]:
        os.system("wget %s -P %s" % (download_link, dataset_dir))

    train_file = os.path.join(dataset_dir, train_url.split("/")[-1])
    dev_file = os.path.join(dataset_dir, dev_url.split("/")[-1])
    test_file = os.path.join(dataset_dir, test_url.split("/")[-1])

    save_dir = config['save_dir']
    run_name = config['run_name']

    additional_args = ""
    if config['run_type'] == 'ae':
        embeddings_ae_dir = config['pre_trained_ae']['dir']
        additional_args = f"--embeddings_ae_dir {embeddings_ae_dir}"
    elif config['run_type'] == 'svd':
        cr = config['svd_args']['compression_ratio']
        svd_num_iters = config['svd_args']['num_iters']
        run_name = f'{run_name}_svd_{cr}'
        additional_args = f"--cr {cr} --svd --svd_num_iters {svd_num_iters}"

    seed = config['seed']
    output_dir = os.path.join(save_dir, f'{run_name}_{seed}')

    os.system("python run_squad.py --output_dir %s \
                                   --model_type=bert \
                                   --model_name_or_path=bert-base-uncased \
                                   --do_train \
                                   --do_eval \
                                   --evaluate_during_training\
                                   --logging_steps 500 \
                                   --threads 4 \
                                   --train_file %s \
                                   --predict_file %s \
                                   --per_gpu_train_batch_size 2 \
                                   --learning_rate 3e-5 \
                                   --gradient_accumulation_steps 8 \
                                   --num_train_epochs 3.0 \
                                   --max_seq_length 384 \
                                   --doc_stride 128 \
                                   --seed=%s %s" % (output_dir, train_file,
                                                    dev_file, seed, additional_args))

    with open(os.path.join(output_dir, 'exp_cfg.json'), 'w') as json_file:
        json.dump(config, json_file)
