def scale_points(points, orig_size, target_size):
    """
    points: dict[name -> (x, y)] in ORIGINAL image pixel space
    orig_size: (orig_w, orig_h)
    target_size: (target_w, target_h)
    Returns dict[name -> (x, y)] rescaled into target pixel space.
    """
    orig_w, orig_h = orig_size
    target_w, target_h = target_size
    sx = target_w / orig_w
    sy = target_h / orig_h
    return {name: (x * sx, y * sy) for name, (x, y) in points.items()}