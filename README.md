# Worker Safety PPE Detection Model

A computer vision project for detecting Personal Protective Equipment (PPE) compliance on construction and industrial sites. The model is fine-tuned to identify whether workers are wearing required safety gear, flagging violations for site safety monitoring.

> **Status:** Early-stage / in development. Training and experimentation are currently being done in Roboflow; code, inference scripts, and evaluation results will be added to this repository as the project progresses.

## Overview

- **Task:** Object detection for PPE compliance
- **Base architecture:** YOLOv8, fine-tuned via [Roboflow](https://roboflow.com/)
- **Detected classes:** `hardhat`, `vest`, `no-hardhat`, `no-vest`

## Model & Data

| Resource | Link |
| --- | --- |
| Roboflow model project | [Personal Protective Equipment Combined Model](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model) |
| Training dataset | [Hard Hat Universe](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t) |

## Roadmap

- [ ] Add training / fine-tuning scripts
- [ ] Add an inference script and demo
- [ ] Publish evaluation metrics (mAP, precision, recall)
- [ ] Add sample detection outputs
- [ ] Add a deployment guide

## Acknowledgements

This project draws inspiration from the following open-source PPE detection projects:

- [snehilsanyal/Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)
- [Vinayakmane47/PPE_detection_YOLO](https://github.com/Vinayakmane47/PPE_detection_YOLO)

## License

No license has been specified yet.
