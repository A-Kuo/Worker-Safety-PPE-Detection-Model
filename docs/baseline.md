# Construction YOLOv8n baseline

This is a **reproduced third-party baseline**, not a model trained in this repo.
Weights, Ultralytics plots, and `results.csv` were inherited from
[snehilsanyal/Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)
by **Snehil Sanyal**. Credit Snehil Sanyal and the Roboflow Construction Site Safety v28 dataset (CC BY 4.0).

Do **not** compare this 10-class Construction mAP to a later 14-class Combined model as a like-for-like win.

## Headline numbers (epoch 99)

Cited values from the inherited training log (rounded):

- **mAP@0.50** = 0.809
- **mAP@0.50:0.95** = 0.507
- **Precision** = 0.900
- **Recall** = 0.731

Exact `results.csv` values at the final logged epoch (99): P=0.89994, R=0.73111, mAP50=0.80881, mAP50-95=0.50710.

Best logged mAP50 is epoch 87 (0.80917). Best mAP50-95 is epoch 99 (0.50710).

## Dataset and split caveat

- Dataset: Roboflow Universe [Construction Site Safety v28](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28).
- Classes (10): `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`, `machinery`, `vehicle`.
- Published split: **2605 / 114 / 82** (train / val / test).
- **Val-size caveat (n=114):** per-class precision, recall, F1, and mAP on 114 images are noisy.
  Do not treat a single-class swing as a real modeling result. Test (n=82) is even smaller.
- `machinery` and `vehicle` exist only on this Construction baseline. They are **not** in the unified Combined 14-class schema.

## Artifacts used

- Weights: `C:/GitHub/Worker Safety PPE Detection Model/baselines/snehilsanyal_yolov8n_css/models/best.pt`
- Training log: `C:/GitHub/Worker Safety PPE Detection Model/baselines/snehilsanyal_yolov8n_css/results/results.csv`
- Construction yaml: `C:/GitHub/Worker Safety PPE Detection Model/configs/data/construction.yaml`
- Image root: `not found` (0 images discovered)

## Ultralytics val (per-class)

Per-class P / R / F1 / mAP **require a val pass** with Construction images on disk.
Images were not available in this checkout (`data/raw/construction` and the original `train`/`valid`/`test` folders are empty or missing).

The inherited `results.csv` only stores **global** box metrics, so class-wise numbers cannot be recovered from the CSV alone.
Re-run:

```bash
python scripts/eval_baseline.py
```

after placing Construction v28 images (or setting `--data` / `--weights`).

## Confusion-matrix interpretation (inherited plot)

Read from the shipped `C:/GitHub/Worker Safety PPE Detection Model/baselines/snehilsanyal_yolov8n_css/results/confusion_matrix.png` (normalized; **not** a substitute for Ultralytics per-class P/R/F1/mAP):

- Strongest diagonal (recall-like): `machinery` ~0.93, `Safety Cone` ~0.91, `Mask` ~0.90.
- Mid: `Person` ~0.80, `Safety Vest` ~0.78, `Hardhat` ~0.76.
- Weakest: `NO-Safety Vest` ~0.70, `NO-Mask` ~0.66, `NO-Hardhat` ~0.62, `vehicle` ~0.57.
- Safety-critical `NO-*` classes leak into **background** (missed violations): `NO-Hardhat` ~0.36, `NO-Mask` ~0.34.
- `vehicle` is often missed entirely (~0.43 background FN). `Person` has a high background false-positive rate (~0.29).
- Some `NO-Safety Vest` boxes are predicted as `Safety Vest` (~0.07) — the costly polarity flip for compliance.

These patterns are why later work sweeps confidence on `no_*` and reports recall-first operating points.

## Per-epoch global metrics

Parsed from the inherited Ultralytics `results.csv` (one row per epoch). These are **global** box metrics, not per-class.

| epoch | P | R | mAP50 | mAP50-95 | val/box | val/cls | val/dfl |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.4523 | 0.2823 | 0.2523 | 0.1069 | 1.8647 | 2.8809 | 1.7853 |
| 1 | 0.5237 | 0.3971 | 0.4055 | 0.1869 | 1.8527 | 2.2875 | 1.8273 |
| 2 | 0.4721 | 0.3817 | 0.3578 | 0.1542 | 1.9889 | 2.3773 | 2.0292 |
| 3 | 0.4432 | 0.3978 | 0.3891 | 0.1825 | 1.9192 | 2.8834 | 1.9127 |
| 4 | 0.5746 | 0.3895 | 0.4133 | 0.1727 | 1.9914 | 2.3586 | 1.9910 |
| 5 | 0.6179 | 0.4693 | 0.4857 | 0.2316 | 1.8551 | 1.9631 | 1.8355 |
| 6 | 0.6232 | 0.4590 | 0.5114 | 0.2389 | 1.8754 | 1.8641 | 1.8966 |
| 7 | 0.6633 | 0.4732 | 0.5360 | 0.2377 | 1.8917 | 1.7950 | 1.8874 |
| 8 | 0.6679 | 0.5339 | 0.5869 | 0.2924 | 1.7357 | 1.6263 | 1.7274 |
| 9 | 0.6936 | 0.5094 | 0.5524 | 0.2445 | 1.9004 | 1.6762 | 1.8905 |
| 10 | 0.7110 | 0.5478 | 0.5857 | 0.2842 | 1.8020 | 1.5603 | 1.8099 |
| 11 | 0.7733 | 0.5316 | 0.6013 | 0.3128 | 1.7277 | 1.5025 | 1.7374 |
| 12 | 0.6800 | 0.5258 | 0.5805 | 0.2629 | 1.9166 | 1.6133 | 1.9101 |
| 13 | 0.7006 | 0.5172 | 0.5758 | 0.2813 | 1.8118 | 1.5332 | 1.7945 |
| 14 | 0.7216 | 0.5699 | 0.6107 | 0.2880 | 1.8141 | 1.4473 | 1.8191 |
| 15 | 0.7355 | 0.5787 | 0.6104 | 0.3175 | 1.7158 | 1.4431 | 1.7245 |
| 16 | 0.7860 | 0.5802 | 0.6351 | 0.3197 | 1.7398 | 1.3520 | 1.7522 |
| 17 | 0.7080 | 0.6074 | 0.6384 | 0.3194 | 1.7159 | 1.3583 | 1.7507 |
| 18 | 0.7830 | 0.5870 | 0.6351 | 0.3220 | 1.7146 | 1.3480 | 1.7363 |
| 19 | 0.7973 | 0.5615 | 0.6470 | 0.3222 | 1.6879 | 1.3443 | 1.7618 |
| 20 | 0.7795 | 0.6067 | 0.6582 | 0.3515 | 1.5989 | 1.2428 | 1.6382 |
| 21 | 0.7986 | 0.6110 | 0.6697 | 0.3336 | 1.7430 | 1.2969 | 1.7332 |
| 22 | 0.8043 | 0.5973 | 0.6543 | 0.3363 | 1.7375 | 1.3742 | 1.7564 |
| 23 | 0.8131 | 0.6315 | 0.6978 | 0.3716 | 1.5899 | 1.1625 | 1.6233 |
| 24 | 0.7218 | 0.6354 | 0.6635 | 0.3535 | 1.6480 | 1.2585 | 1.6986 |
| 25 | 0.8071 | 0.6306 | 0.6842 | 0.3522 | 1.6830 | 1.2334 | 1.6888 |
| 26 | 0.8030 | 0.6047 | 0.6702 | 0.3522 | 1.6760 | 1.2135 | 1.6497 |
| 27 | 0.8344 | 0.5925 | 0.6799 | 0.3797 | 1.5813 | 1.1847 | 1.6112 |
| 28 | 0.8152 | 0.6423 | 0.6994 | 0.3999 | 1.5618 | 1.0860 | 1.5829 |
| 29 | 0.8264 | 0.6306 | 0.6933 | 0.3810 | 1.6150 | 1.1111 | 1.6729 |
| 30 | 0.8127 | 0.6063 | 0.6726 | 0.3387 | 1.7596 | 1.2157 | 1.7747 |
| 31 | 0.8001 | 0.6452 | 0.7049 | 0.3827 | 1.6114 | 1.1023 | 1.6462 |
| 32 | 0.8283 | 0.6336 | 0.7111 | 0.3810 | 1.5793 | 1.0839 | 1.6159 |
| 33 | 0.8742 | 0.6045 | 0.6995 | 0.3839 | 1.5891 | 1.1259 | 1.6245 |
| 34 | 0.8294 | 0.6371 | 0.7101 | 0.3795 | 1.6114 | 1.1097 | 1.6448 |
| 35 | 0.8143 | 0.6593 | 0.7224 | 0.3955 | 1.5416 | 1.0619 | 1.6169 |
| 36 | 0.8521 | 0.6192 | 0.6990 | 0.3856 | 1.5521 | 1.0773 | 1.5995 |
| 37 | 0.7832 | 0.6852 | 0.7270 | 0.3985 | 1.5750 | 1.0224 | 1.6334 |
| 38 | 0.8475 | 0.6551 | 0.7284 | 0.4114 | 1.5168 | 1.0466 | 1.5657 |
| 39 | 0.8285 | 0.6469 | 0.7139 | 0.3792 | 1.6192 | 1.0924 | 1.6474 |
| 40 | 0.8767 | 0.6440 | 0.7416 | 0.4103 | 1.5604 | 1.0249 | 1.6108 |
| 41 | 0.8456 | 0.6635 | 0.7343 | 0.4103 | 1.5296 | 1.0116 | 1.5838 |
| 42 | 0.8324 | 0.6937 | 0.7605 | 0.4262 | 1.4938 | 0.9414 | 1.5290 |
| 43 | 0.8383 | 0.6755 | 0.7372 | 0.3951 | 1.6246 | 1.0292 | 1.6531 |
| 44 | 0.8717 | 0.6443 | 0.7336 | 0.4211 | 1.5041 | 1.0214 | 1.5365 |
| 45 | 0.8576 | 0.6745 | 0.7419 | 0.4288 | 1.4868 | 0.9902 | 1.5297 |
| 46 | 0.8538 | 0.7006 | 0.7623 | 0.4325 | 1.5424 | 0.9681 | 1.5680 |
| 47 | 0.8705 | 0.6638 | 0.7373 | 0.4146 | 1.5345 | 0.9993 | 1.5789 |
| 48 | 0.8529 | 0.6775 | 0.7476 | 0.4253 | 1.4816 | 0.9780 | 1.5380 |
| 49 | 0.8631 | 0.6757 | 0.7516 | 0.4407 | 1.4565 | 0.9607 | 1.5105 |
| 50 | 0.9040 | 0.6660 | 0.7496 | 0.4244 | 1.5131 | 0.9711 | 1.5644 |
| 51 | 0.8152 | 0.6864 | 0.7554 | 0.4268 | 1.5408 | 0.9528 | 1.5744 |
| 52 | 0.8809 | 0.6623 | 0.7549 | 0.4229 | 1.5465 | 0.9658 | 1.5897 |
| 53 | 0.8555 | 0.6800 | 0.7524 | 0.4486 | 1.4246 | 0.9289 | 1.4781 |
| 54 | 0.8848 | 0.6793 | 0.7602 | 0.4344 | 1.4942 | 0.9527 | 1.5402 |
| 55 | 0.8750 | 0.6789 | 0.7500 | 0.4503 | 1.4176 | 0.9014 | 1.4990 |
| 56 | 0.8580 | 0.6911 | 0.7581 | 0.4387 | 1.4826 | 0.9293 | 1.5625 |
| 57 | 0.8616 | 0.6916 | 0.7579 | 0.4494 | 1.4582 | 0.9308 | 1.5197 |
| 58 | 0.8757 | 0.7011 | 0.7713 | 0.4529 | 1.4286 | 0.9067 | 1.4934 |
| 59 | 0.8933 | 0.6876 | 0.7774 | 0.4564 | 1.4272 | 0.8875 | 1.4886 |
| 60 | 0.8742 | 0.7022 | 0.7752 | 0.4462 | 1.4761 | 0.9129 | 1.5581 |
| 61 | 0.8883 | 0.6659 | 0.7577 | 0.4509 | 1.4334 | 0.9106 | 1.4790 |
| 62 | 0.8725 | 0.6963 | 0.7674 | 0.4552 | 1.4693 | 0.8917 | 1.5346 |
| 63 | 0.8548 | 0.7052 | 0.7714 | 0.4561 | 1.4256 | 0.8870 | 1.5047 |
| 64 | 0.8727 | 0.7010 | 0.7737 | 0.4627 | 1.4322 | 0.8701 | 1.5043 |
| 65 | 0.8758 | 0.7120 | 0.7809 | 0.4595 | 1.4215 | 0.8757 | 1.4893 |
| 66 | 0.8598 | 0.6989 | 0.7735 | 0.4744 | 1.3675 | 0.8737 | 1.4522 |
| 67 | 0.8900 | 0.7050 | 0.7780 | 0.4535 | 1.4662 | 0.8636 | 1.5125 |
| 68 | 0.8603 | 0.7201 | 0.7823 | 0.4798 | 1.3615 | 0.8400 | 1.4379 |
| 69 | 0.8414 | 0.7128 | 0.7814 | 0.4633 | 1.4401 | 0.8657 | 1.4880 |
| 70 | 0.8764 | 0.7069 | 0.7793 | 0.4594 | 1.4777 | 0.8791 | 1.5375 |
| 71 | 0.8888 | 0.7077 | 0.7891 | 0.4746 | 1.4099 | 0.8495 | 1.4776 |
| 72 | 0.8886 | 0.7123 | 0.7786 | 0.4546 | 1.4647 | 0.8723 | 1.5309 |
| 73 | 0.9018 | 0.7038 | 0.7830 | 0.4786 | 1.3746 | 0.8495 | 1.4532 |
| 74 | 0.8338 | 0.7295 | 0.7821 | 0.4744 | 1.4095 | 0.8518 | 1.4798 |
| 75 | 0.8559 | 0.7164 | 0.7861 | 0.4755 | 1.3907 | 0.8505 | 1.4762 |
| 76 | 0.8762 | 0.7145 | 0.7778 | 0.4611 | 1.4281 | 0.8585 | 1.5114 |
| 77 | 0.8988 | 0.7142 | 0.7893 | 0.4820 | 1.3576 | 0.8274 | 1.4511 |
| 78 | 0.9157 | 0.7135 | 0.8018 | 0.4916 | 1.3369 | 0.8047 | 1.4198 |
| 79 | 0.8981 | 0.7103 | 0.7989 | 0.4824 | 1.3955 | 0.8138 | 1.4735 |
| 80 | 0.9087 | 0.7130 | 0.7988 | 0.4854 | 1.3743 | 0.8167 | 1.4519 |
| 81 | 0.8973 | 0.7165 | 0.7992 | 0.4899 | 1.3552 | 0.8168 | 1.4507 |
| 82 | 0.8967 | 0.7258 | 0.7979 | 0.4836 | 1.3707 | 0.8227 | 1.4532 |
| 83 | 0.8886 | 0.7225 | 0.8032 | 0.4954 | 1.3320 | 0.7959 | 1.4171 |
| 84 | 0.9148 | 0.7271 | 0.8042 | 0.4886 | 1.3298 | 0.7910 | 1.4259 |
| 85 | 0.9094 | 0.7113 | 0.7915 | 0.4984 | 1.3052 | 0.7997 | 1.3937 |
| 86 | 0.9211 | 0.7199 | 0.8062 | 0.4976 | 1.3341 | 0.7936 | 1.4058 |
| 87 | 0.8857 | 0.7345 | 0.8092 | 0.5021 | 1.3204 | 0.7830 | 1.4054 |
| 88 | 0.8983 | 0.7493 | 0.8067 | 0.4912 | 1.3656 | 0.7808 | 1.4364 |
| 89 | 0.9039 | 0.7274 | 0.8055 | 0.4960 | 1.3460 | 0.7817 | 1.4292 |
| 90 | 0.8523 | 0.7421 | 0.7838 | 0.4929 | 1.3115 | 0.7949 | 1.3984 |
| 91 | 0.9024 | 0.7164 | 0.7916 | 0.4899 | 1.3461 | 0.8072 | 1.4270 |
| 92 | 0.8865 | 0.7241 | 0.7970 | 0.4891 | 1.3230 | 0.7843 | 1.4172 |
| 93 | 0.9132 | 0.7266 | 0.8010 | 0.5017 | 1.3190 | 0.7653 | 1.3953 |
| 94 | 0.9167 | 0.7315 | 0.8068 | 0.4983 | 1.2994 | 0.7819 | 1.3901 |
| 95 | 0.9084 | 0.7307 | 0.8045 | 0.5062 | 1.3053 | 0.7663 | 1.3954 |
| 96 | 0.8859 | 0.7345 | 0.8063 | 0.5049 | 1.2930 | 0.7657 | 1.3878 |
| 97 | 0.9052 | 0.7344 | 0.8061 | 0.5018 | 1.3074 | 0.7668 | 1.3994 |
| 98 | 0.8988 | 0.7299 | 0.8040 | 0.5042 | 1.3049 | 0.7685 | 1.3989 |
| 99 | 0.8999 | 0.7311 | 0.8088 | 0.5071 | 1.2910 | 0.7616 | 1.3917 |

## How this was generated

```bash
python scripts/eval_baseline.py
```

The script always rewrites this file from `results.csv`. Per-class rows appear only when val images are present.
