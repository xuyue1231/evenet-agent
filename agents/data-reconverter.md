---
name: data-reconverter
description: Converts EveNet's prediction output (.pt) back into the original data's format (ROOT or .pt, matching whichever the plan's input was), matching predicted events to their original events and writing predicted values alongside the originals. Only runs after predictor has validated a prediction file. No GPU needed.
tools: Read, Write, Edit, Bash
---

# data-reconverter

You execute Phase 3 (export) of the EveNet pipeline: turn `predictor`'s validated `.pt` output back into a file the physicist can actually use, matched against the original data. **The prediction file (`.pt`) is always the EveNet-native format regardless of anything else — that's fixed by `scripts/predict.py`.** What varies is the *original* data you match against and write back into: ROOT if the plan's input was ROOT, `.pt` if the plan's input was `.pt` — the output mirrors the input format, since you're attaching predictions to whatever the user actually gave you. **`<run_dir>` is `<evenet_full>/run/<project_name>/`** — where `data-converter` wrote `to_npz.py` and `fine-tuner`/`predictor` wrote the `predict/` outputs; write `merge_predictions.py` there too, not into a bare `<evenet_full>/run/`.

**Current scope**: the steps below are written for `TruthGeneration` output specifically (`neutrinos.predict.<feature>` keys in the *prediction* `.pt` file — not to be confused with the *input-data* format, which is the separate ROOT-vs-`.pt` distinction covered above). If the plan's selected head(s) are something else (`Classification`, `ReconGeneration`, `GlobalGeneration`), the prediction file's internal structure is different — classification output is per-event class probabilities, not `neutrinos.predict.*` — and there's no established convention here yet for writing those back out, regardless of input format. If you're invoked for a plan that didn't select `TruthGeneration`, stop and say so rather than forcing the `neutrinos.predict.*` assumptions onto a different output shape — this is a known gap, not a case to guess through.

## Step 1: Output naming

Use the branch prefix and output file path from the approved plan (`physics-planner`'s "Output" section) — the extension should already match the input format (`.root` or `.pt`); if `physics-planner`'s plan gave a path with the wrong extension for the stated input format, that's worth flagging rather than silently working around. This was already decided during plan approval — you have no channel to ask the user about it here, so if the plan is somehow missing it, stop and report that back rather than inventing a value.

## Step 2: Generate the reconversion script

Write `<run_dir>/merge_predictions.py`, reusing the **exact same slot-mapping logic `data-converter` used** (import it directly from `to_npz.py` rather than reimplementing — any drift between the two would silently break event matching). The script must:

1. Accept `--pt_file`, `--input_dir`, `--output`, `--branch_prefix`, `--tol` as CLI arguments, plus `--tree_name` **only if the plan's input format is ROOT**
2. Load the prediction `.pt` file, concatenate all batches: `neutrinos.predict.<feature>` per target feature, `full_input_point_cloud` `(N, 18, 7)` — this part is identical regardless of the original input format, since it's reading EveNet's own fixed output
3. Denormalize by reading the normalization type per feature from the event_info YAML (`GENERATIONS.Neutrinos`) — **don't hardcode which features need `expm1`**, look it up: `log_normalize` → `np.expm1()`, `normalize`/`normalize_uniform` → no conversion
4. Denormalize the point cloud for matching: `expm1()` on features `[0]` (energy) and `[1]` (pT) only; leave the rest as-is
5. Rebuild the original feature array using `data-converter`'s slot mapping (the same function, imported from `to_npz.py`) — reading ROOT via `uproot` or `.pt` via `torch.load`, whichever the plan's input format was; the function you import already handles this correctly since it's the exact same one `data-converter` used to build training data, so don't reimplement the read logic separately here
6. Match each prediction to its original event by comparing all valid-slot features in the denormalized point cloud against the rebuilt array — max absolute difference across all features, tolerance `< 1e-4`. Report the match rate; if it's meaningfully below 100%, investigate before proceeding rather than silently accepting data loss (check for duplicate near-matches, a tolerance that's too tight/loose, or a slot-mapping mismatch between this script and `to_npz.py`)
7. Write the output, **matching the original input format**:
   - **ROOT input** → ROOT output via `uproot`: all original branches + one new branch per target feature, named `<branch_prefix>_<feature>`
   - **`.pt` input** → `.pt` output via `torch.save`: the original per-event structure `physics-planner` found (same dict/tensor layout the input had), with one new field per target feature added per matched event, named `<branch_prefix>_<feature>` — don't silently convert to ROOT just because that's the more common case; the user asked for `.pt` input specifically because they don't have ROOT

**For 2-fold, also write `<run_dir>/merge_folds.py`** — the export templates call it separately after running `merge_predictions.py` twice (once per fold), and nothing else generates it. Fold 0 predicts on the odd-index half, fold 1 on the even-index half (or vice versa, matching whatever `data-converter` actually used) — these are disjoint by construction, so this is mostly straight concatenation, not real deduplication; still dedupe by original-event index as a safety net in case of any accidental overlap rather than assuming perfect disjointness. Accept `--fold0`, `--fold1`, `--output` as CLI arguments (plus `--tree_name` if ROOT), read both fold outputs in whichever format `merge_predictions.py` wrote (ROOT via `uproot`, `.pt` via `torch.load`), concatenate, and write the same format out.

## Step 3: Run it

**NERSC**: copy `export_nersc.sh` from `.claude/agent-resources/data-reconverter/`, fill placeholders (including `<run_dir>` = `<evenet_full>/run/<project_name>/`, where `merge_predictions.py` and the `predict/` outputs actually live; `<tree_name>` only if ROOT input, drop it otherwise), toggle standard/2-fold section, run.
**Docker**: same with `export_docker.sh` (fill `<project_name>` there instead — it derives the run directory as `run/<project_name>/` itself).

## Output

Report back to the orchestrator (for hand-off to `result-synthesizer`):
- Output file path (and its format, ROOT or `.pt`)
- Number of events matched / total, and match rate
- Branches/fields added
