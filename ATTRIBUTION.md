# Attribution

Licence of this project: AGPL-3.0-only (see `LICENSE` and `NOTICE`). This file credits the third-party
work it builds on.

## Ultralytics YOLOv8

Training and export use [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). The
released vest models are derived from YOLOv8n (COCO-pretrained initialization) and are distributed
under AGPL-3.0.

## Inherited baseline (not redistributed)

- **Author / repo:** Snehil Sanyal,
  [snehilsanyal/Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)
- **Role:** the starting point this repository was rewritten from. It is **not** a model trained here.
- **Reported Construction v28 numbers (theirs, 100 epochs YOLOv8n):**
  mAP@0.50 = 0.809, mAP@0.50:0.95 = 0.507, precision 0.900, recall 0.731.
- **What is still here:** the dataset notes / YAML and the training log under
  `archive/baselines/snehilsanyal_yolov8n_css/`.
- **What was removed:** its weights, plots, sample photographs and video. They carried no stated
  licence. They remain in git history and are not covered by this project's licence.

## Datasets (Roboflow Universe, CC BY 4.0)

Use of these datasets requires attribution under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

| Dataset | Role here | Link |
|---|---|---|
| Construction Site Safety v28 | Inherited baseline train/eval (10 classes; `machinery` / `vehicle` stay Construction-only) | [universe.roboflow.com/.../construction-site-safety/dataset/28](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28) |
| Personal Protective Equipment Combined Model v4 | Unified train set (44,002 images, 14 classes) | [universe.roboflow.com/.../personal-protective-equipment-combined-model/dataset/4](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/dataset/4) |
| Hard Hat Universe | Held-out helmet-domain eval only (not mixed into training) | [universe.roboflow.com/.../hard-hat-universe-0dy7t](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t) |
| no-vest-detect (v1) | Extra `no_vest` data for the vest specialist (998 images, ~1,112 no-vest-labeled instances) | [universe.roboflow.com/novest/no-vest-detect](https://universe.roboflow.com/novest/no-vest-detect) |

**Licence verification status.** The CC BY 4.0 statement above is taken from each dataset's Roboflow
Universe page. It was not independently re-verified for `novest/no-vest-detect` when this release was
prepared; check that page before redistributing anything derived from it under different terms. No
dataset images are redistributed in this repository.

Construction v28 notes that some images were cloned from Combined PPE and other
Universe sets. Default protocol is **no merge** of Construction into Combined.

**On `no-vest-detect`:** [project-stixd/vest-5byyt](https://universe.roboflow.com/project-stixd/vest-5byyt)
was considered first but rejected — every one of its 4 downloadable versions
applies a Roboflow preprocessing step (`preprocessing.remap.labels.no-vest.omit:
true`, confirmed via the Roboflow API against all 4 versions) that strips the
`no-vest` class from the actual export, defeating the entire point of using
it. `novest/no-vest-detect` version 1 was verified via the same API check to
omit nothing before downloading.

## Adopted fine-tuning base checkpoint

[Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection)
(YOLOv8m, 14 classes, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
was vetted and adopted as the base checkpoint for gap fine-tuning on
vest/no_vest reliability (README priority #1; see `docs/experiments.md`). It
was trained on the same Roboflow "Personal Protective Equipment Combined
Model v4" export listed above, so its raw class-name strings match this
repo's `COMBINED_RAW_TO_UNIFIED` mapping exactly — no new label mapping was
needed. Pinned at commit `08895b33d95d2587423ebe4f7c1b9c41beebd642` via
`archive/scripts/fetch_checkpoint.py`; see `archive/models/pretrained/manifest.json` for the
verified sha256 and fetch record, and `archive/configs/models/registry.yaml` for the
vetting notes. Do not present the raw checkpoint's numbers
(`archive/docs/baseline_hf.md`) as a model trained in this repo — they are the
starting point for fine-tuning, not our result.

**Class-order correction:** Hexmon's own class order is alphabetical, not
this repo's `UNIFIED_CLASS_NAMES` order — since Ultralytics matches
predictions to ground truth by numeric id, not name, this silently corrupted
both the raw `archive/docs/baseline_hf.md` eval numbers and would have corrupted any
fine-tuning against a standard unified-order dataset. Fixed once via
`archive/scripts/reorder_checkpoint_classes.py`, which permutes only the detection
head's classification-conv output channels (box-regression weights and
everything else are bit-identical) — verified correct via
`archive/scripts/verify_reorder.py` on multiple real test images (identical boxes and
confidences, only class ids/names change). The corrected checkpoint is
`models/pretrained/hexmon_vyra/best_unified_order.pt` (see the
`hexmon_vyra_unified_order` manifest entry); use this one, not the raw fetch,
for both evaluation and fine-tuning.

## Vetted and rejected

[Tanishjain9/yolov8n-ppe-detection-6classes](https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes)
(MIT license) was evaluated as an alternative base checkpoint and rejected:
its 6 classes include no negative/violation classes (no `no_vest`,
`no_helmet`, etc.), which is structurally incompatible with this repo's
paired-class compliance design. See `archive/configs/models/registry.yaml` for the
full vetting notes.
