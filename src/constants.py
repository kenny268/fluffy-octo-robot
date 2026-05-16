from pathlib import Path

NUM_CLASSES = 6
IGNORE_INDEX = 255

CLASS_NAMES = [
    "Water",
    "Land (unpaved)",
    "Road",
    "Building",
    "Vegetation",
    "Unlabeled",
]

# RGB colors in annotation masks -> class index
COLOR_TO_CLASS = {
    (110, 193, 228): 0,
    (226, 169, 41): 1,
    (132, 41, 246): 2,
    (60, 16, 152): 3,
    (254, 221, 58): 4,
    (155, 155, 155): 5,
}

DATA_ROOT = Path(__file__).resolve().parents[1] / "Data" / "aerial imagery"
