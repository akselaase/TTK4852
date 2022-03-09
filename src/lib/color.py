import numpy as np


def srgb_to_linear(image) -> np.ndarray:
    scaled = image / 12.92
    gamma = np.power((image + 0.055) / 1.055, 2.4)
    return np.where(image <= 0.04045, scaled, gamma)

def linear_to_srgb(image) -> np.ndarray:
    scaled = image * 12.92
    gamma = np.power(image, 1 / 2.4) * 1.055 - 0.055
    return np.where(image <= 0.0031308, scaled, gamma)
