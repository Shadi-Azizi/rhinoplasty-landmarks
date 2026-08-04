import numpy as np
import pandas as pd
from pathlib import Path
from rhinolandmarks.datasets.labelme_parser import parse_labelme_json
from rhinolandmarks.taxonomy import get_view_family

ROOT = Path(r"C:\Users\asus\Documents\Shadi Azizi\گرنت\Cropped_photos\views-02")
VIEW_FOLDERS = ["frontal", "basal", "lateral_left", "lateral_right",
                "oblique_left", "oblique_right", "superior"]

rows = []

for folder_name in VIEW_FOLDERS:
    folder = ROOT / folder_name
    if not folder.exists():
        continue
    view_family, _ = get_view_family(folder_name)

    for json_path in folder.glob("*.json"):
        parsed = parse_labelme_json(json_path)
        pts = np.array(list(parsed["points"].values()))
        if len(pts) == 0:
            continue

        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        w, h = x_max - x_min, y_max - y_min
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

        rows.append({
            "view_family": view_family,
            "folder": folder_name,
            "file": json_path.name,
            "box_w": w,
            "box_h": h,
            "box_w_frac": w / parsed["image_width"],
            "box_h_frac": h / parsed["image_height"],
            "center_x_frac": cx / parsed["image_width"],
            "center_y_frac": cy / parsed["image_height"],
        })

df = pd.DataFrame(rows)

print("Per-view-family box size & position variability (as fraction of image dimensions):\n")
summary = df.groupby("view_family").agg(
    n=("file", "count"),
    w_frac_mean=("box_w_frac", "mean"),
    w_frac_std=("box_w_frac", "std"),
    w_frac_cv=("box_w_frac", lambda x: x.std() / x.mean()),  # coefficient of variation
    h_frac_mean=("box_h_frac", "mean"),
    h_frac_std=("box_h_frac", "std"),
    h_frac_cv=("box_h_frac", lambda x: x.std() / x.mean()),
    cx_std=("center_x_frac", "std"),
    cy_std=("center_y_frac", "std"),
).round(4)

print(summary.to_string())

out_path = Path("data") / "framing_consistency_check.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\nRaw per-image data saved to {out_path}")