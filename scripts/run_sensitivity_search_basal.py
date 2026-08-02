import yaml
import subprocess
from pathlib import Path

BASE_CONFIG_PATH = "configs/unet_basal.yaml"
VIEW = "basal"

SEARCH_SPACE = {
    "heatmap_sigma": [2.0, 3.0],
    "learning_rate": [1e-4, 1e-3],
}

OUT_CONFIG_DIR = Path("configs/sensitivity")
OUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def make_variant_config(base_cfg, param_name, param_value):
    cfg = dict(base_cfg)
    cfg[param_name] = param_value
    tag = f"{param_name}_{param_value}"
    cfg["checkpoint_dir"] = str(Path(cfg["checkpoint_dir"]) / "sensitivity" / VIEW / tag)
    return cfg, tag


def main():
    with open(BASE_CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)

    runs = []
    for param_name, values in SEARCH_SPACE.items():
        for value in values:
            cfg, tag = make_variant_config(base_cfg, param_name, value)
            config_path = OUT_CONFIG_DIR / f"{VIEW}_{tag}.yaml"
            with open(config_path, "w") as f:
                yaml.safe_dump(cfg, f)
            runs.append((param_name, value, config_path))

    print(f"Generated {len(runs)} sensitivity run(s) for {VIEW}.")
    for param_name, value, config_path in runs:
        print(f"\n=== {VIEW} | {param_name}={value} ===")
        subprocess.run(["python", "-u", "src/rhinolandmarks/train.py", "--config", str(config_path)])


if __name__ == "__main__":
    main()