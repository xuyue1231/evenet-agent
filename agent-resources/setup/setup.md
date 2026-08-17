# EveNet environment setup

One-time setup so the environment is ready for `physics-planner` and the rest of the agent pipeline. This does not require GPU or launching a container. Read and follow this directly (it's a plain instructions file, not a discovered skill or agent) whenever the environment looks unset up.

## Step 1: Ask the user's computing environment

Ask the user whether they are on:
- **NERSC / HPC with Shifter**
- **A server with Docker available**

Record the environment type for use by the rest of the pipeline.

## Step 2: Clone the EveNet repository, and pin the `evenet` submodule

```bash
git clone --recursive https://github.com/EveNet-HEP/EveNet-Full.git
cd EveNet-Full
```

**Then pin the `evenet` submodule to the last verified-working commit — don't leave it on whatever HEAD the clone happened to pull:**

```bash
cd evenet
git checkout b6518dc
cd ..
```

**Why**: `EveNet-HEP/Core`'s default branch has shipped a regression before. Commit `d2aa1fa` (2026-04-20, "update analysis.yaml and train.yaml to change pt normalization method and adjust neutrino binning parameters...") silently swapped the invisible-particle (`TruthGeneration`) feature-padding scheme for a learned `InvisibleInputProjector`, but never updated `Normalizer.denormalize()`'s padding-removal logic to match — the result is a `padding_size=0` case that Python's `x[:-0]` slicing (which means "keep nothing," not "remove nothing") turns into a 0-length tensor, crashing any `TruthGeneration` analysis's validation step and `predict.py` with a broadcast `RuntimeError`. None of this is mentioned in the commit message, so it's not something a changelog skim would catch. Discovered 2026-08-17 by diffing a freshly-cloned, broken install against `EveNet-photon` (an older install pinned at `b6518dc`, which predates the regression and works).

If the repo is already cloned, don't assume it's safe — check the submodule's actual commit before trusting it:

```bash
cd EveNet-Full/evenet && git log -1 --oneline
```

If it's not at `b6518dc` (or a later commit you've separately verified doesn't have this bug — e.g. by confirming EveNet-HEP/Core fixed the padding regression upstream, or by running a `TruthGeneration` analysis through a full validation epoch without the `Normalizer.denormalize` crash), check out `b6518dc` as above before proceeding. Skip the clone step itself if already cloned, but don't skip the pin check.

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
