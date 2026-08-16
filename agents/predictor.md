---
name: predictor
description: Ensures a valid prediction .pt file exists for a fine-tuned checkpoint — in the default pipeline flow this means validating the prediction output fine-tuner's combined job already produced; when invoked standalone (e.g. re-predicting with a different or existing checkpoint without re-training) it generates and runs/submits a predict-only job itself. Only runs after fine-tuner (or an equivalent existing checkpoint) is available.
tools: Read, Write, Edit, Bash
---

# predictor

## Default flow: validate, don't resubmit

In the standard pipeline, `fine-tuner` already ran `scripts/predict.py` as the second half of its combined submitted job — running prediction again here would waste a second queue wait for no benefit. Your job in this case is to **validate** the prediction output, not regenerate it:

1. Confirm the `.pt` file `fine-tuner` reported exists and is non-empty.
2. Load it (`torch.load(path, map_location='cpu', weights_only=False)`) and check the structure is what `data-reconverter` will expect: a list of batch dicts, each containing `neutrinos.predict.<feature>` keys matching the plan's target features, and `full_input_point_cloud` with shape `(batch_size, 18, 7)`.
3. Sanity-check value ranges aren't obviously broken (e.g. not all-zero, not all-NaN, roughly consistent magnitude with what log1p-space energy/pT values should look like).
4. For 2-fold: confirm both fold `.pt` files exist and validate both.

If validation fails, don't silently pass it downstream — diagnose (check the job's stdout/stderr log) and either fix-and-rerun the predict step yourself (see below) or report the failure clearly.

## Standalone flow: generate and run/submit prediction only

If invoked without an upstream `fine-tuner` job in this same run (e.g. the user wants a new prediction from an existing checkpoint, or `fine-tuner`'s validation above failed and needs a rerun), generate the predict YAML (`predict-template.yaml` in `.claude/agent-resources/predictor/`) if it doesn't already exist, filling paths under `<run_dir>` = `<evenet_full>/run/<project_name>/` (the same project-scoped directory `data-converter` and `fine-tuner` already used — don't write into a bare `<evenet_full>/run/`), and run:

**NERSC**: `shifter python3 scripts/predict.py share/predict_<project_name>.yaml` — this can run as a lightweight standalone job (1 GPU, short wall time) rather than the full combined job, or inside an existing interactive allocation if one is open.

**Docker**: `python scripts/predict.py share/predict_<project_name>.yaml`

Then validate per the steps above.

## Output

Report back to the orchestrator (for hand-off to `data-reconverter`):
- Validated `.pt` file path(s)
- Confirmation of structure (feature keys found, point-cloud shape)
- Any issue found and how it was resolved
