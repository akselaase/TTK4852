from dataclasses import dataclass
from typing import Iterable
import cv2
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from process import save_image, load_image


@dataclass
class Blob:
    # Set of pixel tuples
    pixels: set[tuple[int, int]]
    # ndarray of all pixels in the blob.
    # Shape: (N, 2)
    np_pixels: np.ndarray


def find_blob(mask_array: np.ndarray, start: tuple[int, int]) -> Blob:
    n_rows, n_cols = mask_array.shape

    pixels: set[tuple[int, int]] = set()
    frontier: set[tuple[int, int]] = set()
    visited: set[tuple[int, int]] = set()

    print(f'Searching from {start}')

    # start searching at `start`, and add neighboring pixels
    frontier.add(start)
    while frontier:
        current = frontier.pop()
        row, column = current

        if not (
            0 <= row < n_rows
            and 0 <= column < n_cols
        ):
            # Pixel out of bounds
            continue

        if current in visited:
            continue

        visited.add(current)

        if mask_array[row, column]:
            pixels.add((row, column))
            # Add neighbors no matter if they're in bounds or not
            frontier.add((row - 1, column))
            frontier.add((row + 1, column))
            frontier.add((row, column - 1))
            frontier.add((row, column + 1))

    np_pixels = np.array(list(pixels))

    return Blob(
        pixels=pixels,
        np_pixels=np_pixels,
    )


def find_all_blobs(mask_array: np.ndarray) -> list[Blob]:
    """Finds all connected regions in the mask array.
        `mask_array`: np.ndarray, shape (N, M)
    """
    blobs: list[Blob] = []
    remaining_indices = set(map(tuple[int, int], np.argwhere(mask_array)))

    while remaining_indices:
        guess = remaining_indices.pop()
        blob = find_blob(mask_array, guess)
        blobs.append(blob)
        remaining_indices.difference_update(blob.pixels)
    return blobs


def weighted_center(pixels: Iterable[tuple[int, int]], weights: np.ndarray) -> tuple[float, float]:
    center = np.zeros(2)
    total_weight = 0
    for (row, col) in pixels:
        center += np.array((row, col)) * weights[row, col]
        total_weight += weights[row, col]
    if total_weight > 0:
        center /= total_weight
    return (center[0], center[1])

def find_plane(channel: np.ndarray, channel_diff: np.ndarray, threshold: float, key):
    max_row, max_col = np.unravel_index(np.argmax(channel_diff), channel_diff.shape)
    mask = channel > (threshold * channel[max_row, max_col])
    blob = find_blob(mask, (max_row, max_col))
    pos = weighted_center(blob.pixels, channel**2)
    save_image(channel, f'testing/chan_{key}.png')
    save_image(channel_diff, f'testing/chan_diff_{key}.png')
    save_image(mask, f'testing/chan_mask_{key}.png')
    save_image(mask * channel**2, f'testing/chan_weights_{key}.png')
    return pos, blob

def main():
    image = load_image('output.bak/2020-01-10-10-36-57_image_correct0.png')
    diffed = load_image('output.bak/2020-01-10-10-36-57_image_correct0_diff.png')

    save_image(image, 'testing/image.png')
    save_image(diffed, 'testing/diffed.png')

    scale = 5
    blobs = image.copy()
    subsampled = cv2.resize(image, (diffed.shape[0] * scale, diffed.shape[1] * scale), interpolation=cv2.INTER_NEAREST)
    
    positions: list[tuple[int, int]] = []

    for channel in [0, 1, 2]:
        (r, c), blob = find_plane(
            image[:,:,channel],
            diffed[:,:,channel],
            threshold=0.30,
            key=channel,
        )
        scaled = round(r * scale), round(c * scale)
        print(f'{channel}: {r:.1f}, {c:.1f}')
        subsampled[scaled[0], scaled[1], :] = 1
        blobs[tuple(zip(*blob.np_pixels))] = 1
        positions.append(scaled)

    p_b, p_g, p_r = map(np.array, map(tuple, map(reversed, positions)))
    expected_r = (p_g - p_b) * (1.005 / 0.527) + p_b
    expected_g = (p_r - p_b) * (0.527 / 1.005) + p_b
    expected_b = (p_g - p_r) * (1.005 / (1.005 - 0.527)) + p_r
    cv2.line(subsampled, p_b, p_r, (0.4, 0.4, 0.4), 1, cv2.LINE_AA)
    subsampled[round(expected_r[1]), round(expected_r[0])] = (1, 0, 0) 
    subsampled[round(expected_g[1]), round(expected_g[0])] = (0, 0, 1) 
    subsampled[round(expected_b[1]), round(expected_b[0])] = (0, 1, 0) 

    save_image(subsampled, 'testing/subsampled.png')
    save_image(blobs, 'testing/blobs.png')


if __name__ == '__main__':
    main()
