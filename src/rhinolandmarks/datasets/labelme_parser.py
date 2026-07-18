import json
from pathlib import Path
from ..taxonomy import get_view_family, landmarks_for, EXCLUDED_LABELS


def parse_labelme_json(json_path):
    """
    Reads one labelme JSON, infers its view family from the PARENT
    FOLDER NAME (since your images+jsons sit directly inside each
    view's folder), applies horizontal flip for *_right views, and
    returns points/visibility in that view family's fixed channel order.
    """
    json_path = Path(json_path)
    folder_name = json_path.parent.name
    view_family, needs_flip = get_view_family(folder_name)
    expected_landmarks = landmarks_for(view_family)

    with open(json_path, "r") as f:
        data = json.load(f)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    points = {}
    for shape in data.get("shapes", []):
        label = shape["label"].strip()
        if label in EXCLUDED_LABELS:
            continue
        if label not in expected_landmarks:
            raise ValueError(
                f"Unexpected label '{label}' in {json_path.name} "
                f"(view family '{view_family}'). Expected one of {expected_landmarks}."
            )
        x, y = shape["points"][0]
        if needs_flip:
            x = image_width - x
        points[label] = (float(x), float(y))

    visible = {name: (name in points) for name in expected_landmarks}

    return {
        "view_family": view_family,
        "flipped": needs_flip,
        "points": points,
        "visible": visible,
        "landmark_order": expected_landmarks,
        "image_path": data.get("imagePath"),
        "image_height": image_height,
        "image_width": image_width,
    }


def to_ordered_arrays(parsed):
    """Fixed-order (coords, visible) arrays for this JSON's view family."""
    coords, visible = [], []
    for name in parsed["landmark_order"]:
        if parsed["visible"][name]:
            coords.append(parsed["points"][name])
            visible.append(True)
        else:
            coords.append((0.0, 0.0))
            visible.append(False)
    return coords, visible