# Attribution

Everything under `baselines/snehilsanyal_yolov8n_css/` is third-party work,
inherited rather than trained here. That includes `best.pt`, which lives at
`baselines/snehilsanyal_yolov8n_css/models/best.pt` and is not a result from
this repository.

## Inherited baseline

- Author and repo: Snehil Sanyal,
  [snehilsanyal/Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)
- Kept here: YOLOv8n weights (`best.pt` and pretrained `yolov8n.pt`), Ultralytics
  plots and `results.csv`, sample media, inference outputs, and the original
  Roboflow dataset notes and yaml.
- Their reported Construction v28 numbers, 100 epochs of YOLOv8n:
  mAP@0.50 = 0.809, mAP@0.50:0.95 = 0.507, precision 0.900, recall 0.731.

## Datasets (Roboflow Universe, CC BY 4.0)

Use of these datasets requires attribution under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

| Dataset | Role here | Link |
|---|---|---|
| Construction Site Safety v28 | Inherited baseline train/eval (10 classes; `machinery` / `vehicle` stay Construction-only) | [universe.roboflow.com/.../construction-site-safety/dataset/28](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28) |
| Personal Protective Equipment Combined Model v4 | Unified train set (44,002 images, 14 classes) | [universe.roboflow.com/.../personal-protective-equipment-combined-model/dataset/4](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model/dataset/4) |
| Hard Hat Universe | Held-out helmet-domain eval only (not mixed into training) | [universe.roboflow.com/.../hard-hat-universe-0dy7t](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t) |

Construction v28 notes that some of its images were cloned from Combined PPE
and other Universe sets, so Construction is never merged into Combined here.

## External reference (not our work)

[Hexmon/vyra-yolo-ppe-detection](https://huggingface.co/Hexmon/vyra-yolo-ppe-detection),
a YOLOv8m on Combined v4, is cited as a published reference. It was not produced
here.
