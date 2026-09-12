# Hexmon/vyra-yolo-ppe-detection — raw checkpoint baseline

This is a **reproduced, raw third-party checkpoint's numbers**, not a model
trained or fine-tuned in this repo. See [ATTRIBUTION.md](../ATTRIBUTION.md)
and [`configs/models/registry.yaml`](../configs/models/registry.yaml) for the
vetting record. Do **not** present these numbers as this repo's fine-tuning
result — they are the starting point Phase 3 fine-tunes from, not the
outcome.

- **Checkpoint:** [Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection),
  YOLOv8m, pinned at commit `08895b33d95d2587423ebe4f7c1b9c41beebd642`.
- **Fetched via:** `scripts/fetch_checkpoint.py`; sha256 recorded in `models/pretrained/manifest.json`.
- **Eval command:** `python scripts/eval.py --weights models/pretrained/hexmon_vyra/best.pt --split test`
- **Eval data:** `data/processed/combined/data.yaml` (Combined v4 test split, 4,423 images, unified 14-class schema).

## Read this before the numbers below

**The raw `scripts/eval.py` mAP/precision/recall numbers in this doc are not
a valid measure of the checkpoint's real detection quality.** Ultralytics'
`model.val()` matches a prediction to a ground-truth box by **numeric class
ID**, not by class name. Hexmon's own output head orders its 14 classes
**alphabetically** (`Fall-Detected=0, Gloves=1, Goggles=2, Hardhat=3, ...`),
while `data/processed/combined`'s labels use this repo's **unified schema
order** (`helmet=0, no_helmet=1, vest=2, no_vest=3, ...`) — the same
class-order mismatch documented as a first-class risk in
[`scripts/finetune.py`](../scripts/finetune.py)'s schema guard. Running
`model.val()` directly against `combined.yaml` therefore scores each class
against the *wrong* ground-truth column almost everywhere.

This was empirically confirmed, not just theorized: on the same 15 (alphabetically-first)
test images, ground truth contains 52 `helmet` boxes and the model correctly
predicts 55 `helmet` boxes at conf≥0.25 (verified via `PPEDetector`, which
unifies by **name**, not ID) — clear evidence the checkpoint itself detects
helmets well. Yet the raw `model.val()` run below scores unified class id 0
(`helmet`) at precision=0, recall=0, because Hexmon's own id 0 means
`Fall-Detected`, not `helmet` — the helmet predictions land under Hexmon's
raw id 3 (`Hardhat`) and are compared against whatever unified id 3
(`no_vest`) means instead.

**Getting a scientifically valid raw mAP for this checkpoint requires an
id-permuted evaluation** (either remap the label files to Hexmon's own class
order for this one comparison, or post-process predicted class indices
before scoring) — this is real follow-up work, not done here. The raw
numbers below are recorded for completeness/reproducibility of the exact
`eval.py` command, not as a claim about the model's quality.

## Raw numbers (id-mismatched — see caveat above)

Global (test split, 4,423 images, conf=0.25, iou=0.5):

- mAP@0.50 = 0.057
- mAP@0.50:0.95 = 0.040
- Precision = 0.049
- Recall = 0.130

Only two classes score non-trivially — and even these are almost certainly
spatial-overlap coincidences (e.g. `NO-Hardhat`-labeled head-region boxes
overlapping `mask` ground-truth boxes on the same face), not genuine
per-class evidence:

| unified id | unified name | scored against Hexmon's raw class | P | R | mAP50 |
|---|---|---|---|---|---|
| 8 | `mask` | `NO-Hardhat` (id 8 in Hexmon's own order) | 0.192 | 0.906 | 0.179 |
| 9 | `no_mask` | `NO-Mask` (id 9 in Hexmon's own order) | 0.493 | 0.908 | 0.622 |

All other classes score at or near 0 for the reason explained above, not
because the checkpoint fails to detect them. Full per-class output:
`results/analysis/eval_hexmon_vyra_raw.json`.

## Qualitative spot-check (name-unified, via `PPEDetector`)

On `data/processed/combined/test/images/-1003-_png_jpg.rf.40a811e60f1b213fd091d11af90ad6e4.jpg`:

```
helmet 0.874  (368.8, 275.3, 511.2, 383.0)
helmet 0.867  (123.4, 263.9, 279.0, 386.9)
helmet 0.665  (310.7, 130.8, 338.6, 154.3)
```

Correctly unified, high-confidence helmet detections — consistent with the
count-matching evidence above. The compliance-summary pass (`scripts/eval.py`'s
`--skip-compliance`-off default) found 0 workers across the first 64 test
images; this is at least partly explained by those images being
disproportionately close-up/helmet-only shots (52 of 52 ground-truth boxes
in the first 15 images are `helmet`), not necessarily a `person`-detection
weakness — this needs a larger, more representative sample to confirm either
way before drawing conclusions.

## How this was generated

```powershell
python scripts/fetch_checkpoint.py --repo-id Hexmon/vyra-yolo-ppe-detection --filename best.pt --revision 08895b33d95d2587423ebe4f7c1b9c41beebd642 --name hexmon_vyra --license cc-by-4.0
python scripts/eval.py --weights models/pretrained/hexmon_vyra/best.pt --split test --out results/analysis/eval_hexmon_vyra_raw.json
```

This doc is hand-written from that eval run's output (unlike `docs/baseline.md`,
there is no dedicated regenerate script for this doc yet — the caveat above
is the reason: a naive re-run would need the same id-mismatch health warning
attached every time, which is worth writing once rather than templating
until an id-permuted evaluator exists).
