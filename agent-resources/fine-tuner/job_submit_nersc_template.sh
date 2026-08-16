#!/bin/bash
# EveNet — NERSC batch job (fine-tuning + prediction)
#
# Placeholders to replace:
#   <project_name>      user's project name
#   <account>           NERSC account without _g suffix, e.g. m2616
#   <wall_time>         e.g. 04:00:00
#   <total_gpus>        number_of_workers * resources_per_worker["GPU"] from finetune YAML
#   <container_image>   e.g. registry.nersc.gov/<project>/avencast/evenet:1.5
#   <run_dir>           absolute path to EveNet-Full/run
#   <evenet_full>       absolute path to EveNet-Full repo
#   <wandb_api_key>     W&B API key
#   <wandb_project>     W&B project name

#SBATCH --job-name=evenet_<project_name>
#SBATCH --account=<account>_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=<wall_time>
#SBATCH --nodes=1
#SBATCH --gpus=<total_gpus>
#SBATCH --image=<container_image>
#SBATCH --output=<run_dir>/logs/run_%j.out
#SBATCH --error=<run_dir>/logs/run_%j.err

cd <evenet_full>
export PYTHONPATH=<evenet_full>:$PYTHONPATH
export WANDB_API_KEY=<wandb_api_key>
export WANDB_PROJECT=<wandb_project>

echo "=== Fine-tuning ==="
shifter python3 scripts/train.py share/<project_name>.yaml --load_all

echo "=== Prediction ==="
shifter python3 scripts/predict.py share/predict_<project_name>.yaml
