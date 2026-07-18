"""
Single source of truth for the multi-view landmark taxonomy.

Core rule: a landmark's identity is (view_family, name), not just name.
Frontal-glabella and lateral-glabella are different detection targets.
EXCEPTION: lateral_left/right and oblique_left/right are true mirror
pairs of the same optical feature, so they merge into one "lateral" /
"oblique" family each — doubling effective training samples for those
heads. Frontal, basal, and superior are single fixed projections, so
no flip/merge applies to them.
"""

# Edit these strings if your actual Drive folder names differ.
FOLDER_TO_VIEW_FAMILY = {
    "frontal":        ("frontal",  False),
    "basal":          ("basal",    False),   # worm's-eye view
    "lateral_left":   ("lateral",  False),
    "lateral_right":  ("lateral",  True),
    "oblique_left":   ("oblique",  False),
    "oblique_right":  ("oblique",  True),
    "superior":       ("superior", False),
}

# view_family -> ordered landmark names. This order fixes the channel
# order for that view's decoder head — never reorder after training starts.
VIEW_LANDMARKS = {
    "frontal":  ["glabella", "nasion", "r_alare", "l_alare", "LTD", "RTD"],
    "basal":    ["r_alare", "l_alare"],
    "lateral":  ["glabella", "nasion", "pronasale", "subnasale", "SL"],
    "oblique":  ["glabella", "nasion", "pronasale", "subnasale", "SL"],
    "superior": ["glabella", "nasion", "pronasale", "SL"],  # stomion dropped
}

VIEW_FAMILIES = list(VIEW_LANDMARKS.keys())
EXCLUDED_LABELS = {"stomium"}

def get_view_family(folder_name):
    """Returns (view_family, needs_horizontal_flip) for a raw folder name."""
    key = folder_name.strip().lower()
    if key not in FOLDER_TO_VIEW_FAMILY:
        raise ValueError(
            f"Unknown view folder '{folder_name}'. "
            f"Expected one of {list(FOLDER_TO_VIEW_FAMILY.keys())}."
        )
    return FOLDER_TO_VIEW_FAMILY[key]


def landmarks_for(view_family):
    if view_family not in VIEW_LANDMARKS:
        raise ValueError(f"Unknown view family '{view_family}'.")
    return VIEW_LANDMARKS[view_family]


def total_channels():
    """Sum of channels across all 5 view-family heads (sanity-check number)."""
    return sum(len(v) for v in VIEW_LANDMARKS.values())