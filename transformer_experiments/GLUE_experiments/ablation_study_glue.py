import os
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    task_name = config['task_name']
    save_dir = "ablation_studies"
    run_name = config['run_name']
    run_type = config['run_type']
    for cr in config['crs']:
        additional_args = ""
        if run_type == 'ae':
            embeddings_ae_dir = config['pre_trained_ae'][cr]
            run_name = f"{config['run_name']}_linear_ae_{cr}"
            additional_args = f"--embeddings_ae_dir {embeddings_ae_dir} "
        elif run_type == 'svd':
            run_name = f"{config['run_name']}_svd_{cr}"
            additional_args = f"--cr {cr} --svd --svd_iter {config['svd_iters'][cr]}"

        seed = config['seed']
        output_dir = os.path.join(save_dir, task_name, f'{run_name}_{seed}')

        os.system(f"python -m transformer_experiments.GLUE_experiments.run_glue \
                           --output_dir {output_dir} \
                           --task_name {task_name} \
                           --model_name_or_path=bert-base-uncased \
                           --evaluation_strategy=steps \
                           --logging_dir runs/{task_name}_{run_type}_{cr} \
                           --max_seq_length 128 \
                           --do_train \
                           --do_predict \
                           --save_strategy epoch \
                           --fp16 \
                           --do_eval \
                           --eval_steps {config['eval_steps']} \
                           --logging_steps 10 \
                           --per_device_train_batch_size 16 \
                           --per_device_eval_batch_size 4 \
                           --learning_rate 2e-5 \
                           --gradient_accumulation_steps 2 \
                           --num_train_epochs {config['epochs']} \
                           --seed={seed} {additional_args}")

        with open(os.path.join(output_dir, 'exp_cfg.json'), 'w') as json_file:
            json.dump(config, json_file)
