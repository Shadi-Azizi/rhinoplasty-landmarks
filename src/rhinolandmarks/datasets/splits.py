import csv
import re
from pathlib import Path


def extract_patient_id(filename_stem):
    return re.sub(r"_\d+$", "", filename_stem)


def load_split_assignment(split_path):
    """Returns dict: anonymized_id -> 'train' | 'val' | 'test'"""
    assignment = {}
    with open(split_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assignment[row["anonymized_id"]] = row["split"]
    return assignment


def get_split_json_paths(view_folder, split, split_path):
    """
    view_folder: Path to a single view folder (e.g. .../lateral_left)
    split: 'train' | 'val' | 'test'
    split_path: Path to data/splits/patient_split.csv
    Returns list of json Paths in that folder belonging to that split.
    """
    anon_to_split = load_split_assignment(split_path)
    view_folder = Path(view_folder)

    result = []
    for json_path in view_folder.glob("*.json"):
        patient_id = extract_patient_id(json_path.stem)
        if anon_to_split.get(patient_id) == split:
            result.append(json_path)
    return sorted(result)