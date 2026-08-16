# docs/assets/demo/

Drop real screenshots/GIFs/plots here as they're produced — the top-level
`README.md` already references these exact filenames as placeholders:

| Filename | What it should be |
|---|---|
| `hero_demo.gif` | Short screen-capture of the live demo UI (Milestone M5) detecting PPE with compliance captions |
| `sample_detection.jpg` | One real annotated detection output from this project's own trained model (boxes + labels visible) |
| `class_distribution.png` | Full per-class distribution chart once all 4 dataset sources are merged (see `docs/DATA_DISTRIBUTION.md`) |
| `training_curves.png` | This project's own loss/mAP-vs-epoch training curves (from a real Kaggle/Colab run) |
| `reliability_diagram.png` | Calibration reliability diagram from `src/evaluation/calibration.py`, run against a real trained model |
| `webui_screenshot.png` | Screenshot of the deployed FastAPI/web demo UI (Milestone M5) |

None of these exist yet as of this commit — the README placeholders will
render as broken images until you add real files here with these exact names
(or update the README's `<img src=...>` paths to match whatever you add).
