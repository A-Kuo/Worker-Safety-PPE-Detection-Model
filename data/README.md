# Data

Raw Roboflow exports go in `data/raw/{construction,combined,hardhat}/`, which is
gitignored. The inherited Construction yaml and notes are under
`baselines/snehilsanyal_yolov8n_css/data/`. The dataset pointers that training
and eval actually read live in `configs/data/`.
