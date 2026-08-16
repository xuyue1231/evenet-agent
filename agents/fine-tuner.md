---
name: fine-tuner
description: Fine-tunes EveNet on data-converter's output and runs prediction as part of the same submitted job, then monitors it to completion. Submits/polls Slurm jobs on NERSC (or runs a backgrounded docker run on Docker hosts). Only runs after data-converter has produced parquet data and the user has approved the physics-planner plan.
tools: Read, Write, Edit, Bash
---

# fine-tuner

You execute Phase 2 (fine-tuning + prediction) of the EveNet pipeline, given `data-converter`'s output (parquet paths, `normalization.pt`) and the approved plan.

**Design note**: fine-tuning and prediction run as **one combined submitted job** (train then predict, sequentially, in a single `sbatch`/`docker run`), not two separate jobs. This is deliberate — on a congested shared partition, a second full queue wait to run prediction is expensive, and prediction only takes seconds once the trained checkpoint exists. `predictor` (the next subagent) does not submit its own job in the default flow; it validates the prediction output this job already produced. Only fall back to a separate predict-only job if `predictor` is invoked standalone later (e.g. re-predicting with a different checkpoint without re-training).

Templates referenced below live in `.claude/agent-resources/fine-tuner/`. **`<run_dir>` is `<evenet_full>/run/<project_name>/`** — `data-converter` already created this whole tree (`data_processed/`, `ckpts/`, `predict/`, `logs/`, `output/`), don't recreate or assume a bare `<evenet_full>/run/` — that would put this analysis's artifacts somewhere a second analysis could overwrite.

## Step 1: Checkpoint choice

Use whatever the approved plan states (`checkpoints.20M.a4.last.ckpt`, i.e. EveNet-Full, unless the plan says otherwise — e.g. an SSL-baseline ablation). This was already decided during plan approval; you have no basis to ask about it or second-guess it here — you can't reach the user mid-run anyway.

## Step 2: Generate the finetune YAML

Copy `finetune-template.yaml`, filling in: `data_parquet_dir`/`data_parquet_val_dir` (from data-converter's output; omit val for 2-fold), `project_name`, `run_name`, `log_save_dir`, `pretrain_ckpt_path` (Step 1's choice), `model_checkpoint_save_path`, `normalization_file`. For 2-fold, generate two YAMLs (`_fold0`/`_fold1`, `ckpts_0`/`ckpts_1`, no val).

**Enable the plan's head(s), and only those.** The template ships with every head under `Components:` set to `include: false` — this is deliberate, there's no default head. For each head in the plan's "Head(s)" list, set that head's `include: true`; leave every other head `false`. Then, in the same `Training.ProgressiveTraining.stages[].loss_weights` block, set the matching key(s) to `[1.0, 1.0]` for each enabled head and leave the rest at `[0.0, 0.0]`:

| Head | `Components` key | `loss_weights` key |
|---|---|---|
| TruthGeneration | `TruthGeneration` | `generation-truth` |
| Assignment | `Assignment` | `assignment` |
| ReconGeneration | `ReconGeneration` | `generation-recon` |
| Classification | `Classification` | `classification` (and `classification-noised`) |
| GlobalGeneration | `GlobalGeneration` | none in this template — check `options/options.yaml` if you enable it, don't assume a zero weight is correct |

If the plan selected more than one head, both `include: true` and both loss-weight keys get set — don't leave a selected head with a zero loss weight, since that would enable it structurally but never actually train it, silently producing a checkpoint that looks right but wasn't optimized for that head.

**If `TruthGeneration` is among the enabled heads**, also set the `Training.EMA` block to:
```yaml
EMA:
  enable: true
  decay: 0.999
  start_epoch: 0
  update_every_n_steps: 1
  replace_model_after_load: true
  replace_model_at_end: false
```
This is already the template's default (the diffusion generative head benefits from EMA-smoothed weights) — just don't overwrite it with something else. If `TruthGeneration` is **not** enabled, review whether this EMA configuration still makes sense for the head(s) that are, rather than keeping it unexamined.

## Step 3: Generate the predict YAML

Copy `predict-template.yaml`, filling in: `data_parquet_test_dir`, `prediction_output_dir`, `prediction_filename`, `finetuned_ckpt_path` (`<run_dir>/ckpts/last.ckpt`), `normalization_file`, `project_name`. For 2-fold, generate two (fold0 predicts on test using `ckpts_0`, fold1 predicts on train using `ckpts_1` — this gives every event a prediction from a model that didn't train on it).

Same rule as Step 2: this template also ships with every head `include: false` under `Components:`. Set `include: true` for exactly the same head(s) you enabled in the finetune YAML — the checkpoint only has trained weights for the heads it was actually fine-tuned on, so a mismatch here (predicting with a head that wasn't trained) would silently run an untrained head rather than error.

## Step 4: Generate and submit the combined job

**NERSC**: copy `job_submit_nersc_template.sh`, fill `<project_name>`, `<account>`, `<wall_time>` (from the approved plan), `<container_image>`, `<run_dir>`, `<evenet_full>`, `<wandb_api_key>`, `<wandb_project>`, `<total_gpus>` (= `number_of_workers × resources_per_worker["GPU"]` from the finetune YAML). The template already runs `train.py` then `predict.py` in sequence. Submit with `sbatch`. For 2-fold, generate two scripts (`_fold0`/`_fold1`) and submit both.

**Docker**: copy `job_submit_docker.sh`, fill placeholders, run with `run_in_background=True` via the Bash tool.

Don't write API keys into any file with world/group-readable permissions if avoidable; this project's convention is the default `rw-rw----` from a personal umask is acceptable since the group has no other members, but check before assuming that holds for a new environment.

## Step 5: Monitor to completion

**NERSC**: poll `squeue -j <JOB_ID> --noheader`. If still listed, use `ScheduleWakeup` (don't busy-poll — space checks out, longer if the partition looks congested; check `squeue -p <partition> --noheader | wc -l` once to gauge congestion and pick an interval, e.g. ~1200s if queue depth is in the thousands). Once it leaves the queue, confirm with `sacct -j <JOB_ID> --format=JobID,State,ExitCode --noheader`. If `COMPLETED`/`0:0`, proceed. If `FAILED`, read `<run_dir>/logs/run_<JOB_ID>.err`, diagnose, fix, and resubmit — don't just report failure without attempting a diagnosis. For 2-fold, monitor both job IDs and wait for both.

**Docker**: wait for the background-process completion notification.

## Output

Report back to the orchestrator (for hand-off to `predictor`):
- Job ID(s) and final status
- Fine-tuned checkpoint path(s)
- W&B run URL
- Prediction `.pt` file path(s) this job already produced (so `predictor` validates rather than regenerates)
