# Dataset

Aerial semantic segmentation tiles with RGB orthophotos and polygon masks.

## Layout

```
aerial imagery/
  classes.json          # Class names and colors
  Tile 1/ … Tile 8/
    images/*.jpg        # RGB orthophoto patches
    mask/*.png          # Polygon masks (RGB → class in code)
```

## Split (default in code)

- **Train:** Tile 1–6 (18 images)
- **Val:** Tile 7–8 (6 images)

Configured in `src/dataset.py` → `train_val_split()`.
