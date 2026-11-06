"""Input validation shared by the Anvil point-classification server."""

import numpy as np
import pandas as pd


MAX_POINTS = 10_000


def validate_xy(values):
    """Return a finite, non-empty ``(n, 2)`` float array of coordinates."""
    try:
        xy = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Coordinates must be numeric.") from exc

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Expected exactly two coordinate columns: x and y.")
    if xy.shape[0] == 0:
        raise ValueError("At least one coordinate pair is required.")
    if xy.shape[0] > MAX_POINTS:
        raise ValueError(f"Upload no more than {MAX_POINTS} coordinate pairs.")
    if not np.isfinite(xy).all():
        raise ValueError("Coordinates must all be finite numbers.")
    return xy


def read_points_csv(path):
    """Read the website's coordinate CSV and reject malformed input."""
    try:
        points = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("Upload a valid UTF-8 CSV file with x and y columns.") from exc

    if list(points.columns) != ["x", "y"]:
        raise ValueError("CSV columns must be exactly 'x' and 'y', in that order.")
    return validate_xy(points.to_numpy())
