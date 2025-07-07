"""Constants for the multimodal-iad package."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
ROOT_DIR = Path(__file__).parent.parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
DATASETS_DIR = Path(os.environ.get("IAD_DATASETS_PATH", ROOT_DIR / "datasets"))
RESULTS_DIR = ROOT_DIR / "results"

# Ensure the directory exists
DATASETS_DIR.mkdir(exist_ok=True, parents=True)

# Image dimensions
RGB_IMAGE_DIMS = 3
GRAYSCALE_IMAGE_DIMS = 2

VISUALIZATION_CONFIG_KWARGS = {
    "overlay_fields": [("image", ["anomaly_map", "gt_mask"])],
    "overlay_fields_config": {
        "anomaly_map": {"normalize": True, "colormap": True},
        "gt_mask": {"mode": "contour", "color": (255, 255, 255), "alpha": 0.9},
    },
    "text_config": {"enable": False},
}
