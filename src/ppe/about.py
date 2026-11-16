"""Project identity shown in user interfaces.

The detector depends on Ultralytics (AGPL-3.0) and ships YOLOv8-derived weights, so the project is
AGPL-3.0. Section 13 of that licence requires offering the source to people who use the software over
a network; every interface (API root, web UI, native window) links here.
"""

from __future__ import annotations

NAME = "PPE vest detector"
LICENSE = "AGPL-3.0"
SOURCE_URL = "https://github.com/A-Kuo/Worker-Safety-PPE-Detection-Model"


def notice() -> str:
    return f"{NAME} - licensed {LICENSE}. Source code: {SOURCE_URL}"
