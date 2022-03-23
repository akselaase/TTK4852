import numpy as np


def weighted_center(weights: np.ndarray, center: tuple[int, int], radius: int = 2) -> tuple[float, float]:
    cx, cy = center
    X = np.arange(max(0, cx - radius), min(weights.shape[0], cx + radius + 1))
    Y = np.arange(max(0, cy - radius), min(weights.shape[1], cy + radius + 1))
    X, Y = np.meshgrid(X, Y, sparse=True)
    weights = weights[X, Y]
    weight_sum = weights.sum()
    return (
        (X * weights).sum() / weight_sum,
        (Y * weights).sum() / weight_sum
    )
