import json
from pathlib import Path
from collections import defaultdict

# EDIT this to your actual root folder containing the 7 view subfolders
ROOT = Path(r"C:\Users\asus\Documents\Shadi Azizi\گرنت\Cropped_photos\views-02")

VIEW_FOLDERS = [
    "frontal", "basal", "lateral_left", "lateral_right",
    "oblique_left", "oblique_right", "superior",
]

for view in VIEW_FOLDERS:
    folder = ROOT / view
    if not folder.exists():
        print(f"{view}: folder not found, skipping")
        continue

    sizes = defaultdict(int)
    for json_path in folder.glob("*.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        w, h = data["imageWidth"], data["imageHeight"]
        sizes[(w, h)] += 1

    print(f"\n{view}: {sum(sizes.values())} images")
    for (w, h), count in sorted(sizes.items(), key=lambda x: -x[1]):
        print(f"   {w}x{h}  (aspect {w/h:.2f})  -> {count} images")