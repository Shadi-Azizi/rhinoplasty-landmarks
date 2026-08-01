import yaml
import torch
import pandas as pd
from pathlib import Path

SENSITIVITY_ROOT = Path("/content/drive/MyDrive/rhino-landmarks-data/checkpoints/sensitivity")

rows = []
for view_dir in SENSITIVITY_ROOT.iterdir():
    if not view_dir.is_dir():
        continue
    view = view_dir.name
    for run_dir in view_dir.iterdir():
        ckpt_path = run_dir / f"unet_{view}_best.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        param_name, param_value = run_dir.name.rsplit("_", 1)
        rows.append({
            "view": view,
            "param_varied": param_name,
            "value": param_value,
            "best_val_nme": ckpt["val_nme"],
            "epoch": ckpt["epoch"],
        })

df = pd.DataFrame(rows).sort_values(["view", "param_varied", "value"])
print(df.to_string(index=False))

out_path = SENSITIVITY_ROOT.parent.parent / "results" / "sensitivity_results.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")