#!/bin/bash
# EveNet — Docker job (fine-tuning + prediction)
# Run in background via the Bash tool with run_in_background=True.
#
# Placeholders to replace:
#   <project_name>      user's project name
#   <evenet_full>       absolute path to EveNet-Full repo on host
#   <wandb_api_key>     W&B API key
#   <wandb_project>     W&B project name
#   <run_dir>           absolute path to EveNet-Full/run (for log output)

EVENET_FULL=<evenet_full>
CONTAINER_IMAGE=docker.io/avencast1994/evenet:1.5

docker run --rm --gpus all \
  -v ${EVENET_FULL}:/workspace/EveNet_Full \
  -e WANDB_API_KEY=<wandb_api_key> \
  -e WANDB_PROJECT=<wandb_project> \
  ${CONTAINER_IMAGE} bash -c \
  "cd /workspace/EveNet_Full && \
   export PYTHONPATH=/workspace/EveNet_Full:\$PYTHONPATH && \
   echo '=== Fine-tuning ===' && \
   python scripts/train.py share/<project_name>.yaml --ray_dir ~/ray_results && \
   echo '=== Prediction ===' && \
   python scripts/predict.py share/predict_<project_name>.yaml" \
  > <run_dir>/logs/run.out 2>&1
