# Attribution

This repository is an original rewrite. The files under
`baselines/snehilsanyal_yolov8n_css/` are **inherited third-party artifacts**,
not work trained in this project. Do not present `models/best.pt` (now at
`baselines/snehilsanyal_yolov8n_css/models/best.pt`) as our training run.

## Inherited baseline

- **Author / repo:** Snehil Sanyal —
  [snehilsanyal/Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)
- **What we kept:** YOLOv8n weights (`best.pt`, pretrained `yolov8n.pt`),
  Ultralytics plots and `results.csv`, sample source media, inference outputs,
  and the original Roboflow dataset notes / YAML.
- **Reported Construction v28 numbers (theirs, 100 epochs YOLOv8n):**
  mAP@0.50 = 0.809, mAP@0.50:0.95 = 0.507, precision 0.900, recall 0.731.

## Datasets (Roboflow Universe, CC BY 4.0)

Use of these datasets requires attribution under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

| Dataset | Role here | Link |
|---|---|---|
| Construction Site Safety v28 | Inherited baseline train/eval (10 classes; `machinery` / `vehicle` stay Construction-only) | [universe.roboflow.com/.../construction-site-safety/dataset/28](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28) |
| Personal Protective Equipment Combined Model v4 | Unified train set (44,002 images, 14 classes) | [universe.roboflow.com/.../personal-protective-equipment-combined-model/dataset/4](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/dataset/4) |
| Hard Hat Universe | Held-out helmet-domain eval only (not mixed into training) | [universe.roboflow.com/.../hard-hat-universe-0dy7t](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t) |

Construction v28 notes that some images were cloned from Combined PPE and other
Universe sets. Default protocol is **no merge** of Construction into Combined.

## External reference (not our work)

[Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection)
(YOLOv8m on Combined v4) is an external published reference, not a checkpoint
produced here.
