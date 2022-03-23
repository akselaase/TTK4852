import numpy as np


def weighted_center(weights: np.ndarray, center: tuple[int, int], radius: int = 2) -> np.ndarray:
    """Returns a (2,) array with weighted X, Y coordinate."""
    cx, cy = center
    X = np.arange(max(0, cx - radius), min(weights.shape[0], cx + radius + 1))
    Y = np.arange(max(0, cy - radius), min(weights.shape[1], cy + radius + 1))
    X, Y = np.meshgrid(X, Y, sparse=True)
    weights = weights[X, Y]
    weight_sum = weights.sum()
    x = (X * weights).sum()
    y = (Y * weights).sum()
    epsilon = 0.00001
    return (
        (epsilon + np.array((x, y)))
        / (epsilon + weight_sum)
    )
