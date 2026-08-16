# EveNet environment setup

One-time setup so the environment is ready for `physics-planner` and the rest of the agent pipeline. This does not require GPU or launching a container. Read and follow this directly (it's a plain instructions file, not a discovered skill or agent) whenever the environment looks unset up.

## Step 1: Ask the user's computing environment

Ask the user whether they are on:
- **NERSC / HPC with Shifter**
- **A server with Docker available**

Record the environment type for use by the rest of the pipeline.

## Step 2: Clone the EveNet repository

```bash
git clone --recursive https://github.com/EveNet-HEP/EveNet-Full.git
```

If the repo is already cloned, skip this step.

## Step 3: Pull the container image

**Docker:**
```bash
docker pull docker.io/avencast1994/evenet:1.5
```

**NERSC (Shifter):**
```bash
shifterimg pull docker.io/avencast1994/evenet:1.5
```

## Step 4: Configure W&B credentials

Ask the user for:
- `WANDB_API_KEY`: Users register at https://wandb.ai/ to get their API key.
- `WANDB_PROJECT`: A descriptive project name chosen by the user.

Record these for use by the rest of the pipeline (do not store API keys in any committed file).

## Step 5: Download pretrained weights

Download checkpoint files from HuggingFace into a `Weights` folder in the user's project directory (the parent of EveNet-Full, or wherever they prefer):

```bash
mkdir -p Weights
wget -O Weights/SSL.20M.last.ckpt \
  "https://huggingface.co/Avencast/EveNet/resolve/main/SSL.20M.last.ckpt"
wget -O Weights/checkpoints.20M.a4.last.ckpt \
  "https://huggingface.co/Avencast/EveNet/resolve/main/checkpoints.20M.a4.last.ckpt"
```

If checkpoints already exist, verify their presence and skip download.

Both are downloaded because `physics-planner`/`fine-tuner` need to pick one as `pretrain_ckpt_path`:
- `checkpoints.20M.a4.last.ckpt` is **EveNet-Full** (Stage 2: backbone + jointly pretrained Classification + TruthGeneration + ReconGeneration heads). Recommended default — the EveNet paper shows it consistently outperforms SSL-only as a fine-tuning starting point, including on decay topologies unseen during pretraining.
- `SSL.20M.last.ckpt` is **EveNet-SSL** (Stage 1: backbone + ReconGeneration only, self-supervised, no Classification/TruthGeneration heads). Weaker as a starting point per the paper, but a valid choice for a from-scratch-head ablation/baseline.

`physics-planner` states this default in its plan rather than asking proactively — see the top-level `CLAUDE.md` for how the pipeline is structured.

## Output

After completing all steps, report:
- Environment type (Docker / NERSC)
- EveNet-Full repo path
- Weights directory path with available checkpoints
- Confirmation that the environment is ready for `physics-planner`
