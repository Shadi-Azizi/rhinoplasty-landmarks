import re
import csv
import random
from pathlib import Path

# EDIT to your actual data root
ROOT = Path("paste your folder paths")
VIEW_FOLDERS = ["frontal", "basal", "lateral_left", "lateral_right",
                "oblique_left", "oblique_right", "superior"]

SPLIT_PATH = Path(__file__).resolve().parents[1] / "data" / "splits" / "patient_split.csv"

SEED = 43
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def extract_patient_id(filename_stem):
    return re.sub(r"_\d+$", "", filename_stem)  # e.g. "patient_0037_5" -> "patient_0037"


def main():
    anon_ids = set()
    for view in VIEW_FOLDERS:
        folder = ROOT / view
        if not folder.exists():
            continue
        for json_path in folder.glob("*.json"):
            anon_ids.add(extract_patient_id(json_path.stem))

    anon_ids = sorted(anon_ids)
    print(f"Found {len(anon_ids)} unique patients.")

    rng = random.Random(SEED)
    shuffled = anon_ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    split_assignment = {}
    for i, anon in enumerate(shuffled):
        if i < n_train:
            split_assignment[anon] = "train"
        elif i < n_train + n_val:
            split_assignment[anon] = "val"
        else:
            split_assignment[anon] = "test"

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["anonymized_id", "split"])
        for anon, split in sorted(split_assignment.items()):
            writer.writerow([anon, split])

    counts = {"train": 0, "val": 0, "test": 0}
    for s in split_assignment.values():
        counts[s] += 1
    print(f"Wrote split file to: {SPLIT_PATH}")
    print(f"train: {counts['train']}, val: {counts['val']}, test: {counts['test']}")


if __name__ == "__main__":
    main()