from dataclasses import dataclass
import numpy as np

@dataclass
class Blob:
    # Set of pixel tuples
    pixels: set[tuple[int, int]]
    # ndarray of all pixels in the blob.
    # Shape: (N, 2)
    np_pixels: np.ndarray


def bucket_fill(mask_array: np.ndarray, start: tuple[int, int]) -> Blob:
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
        blob = bucket_fill(mask_array, guess)
        blobs.append(blob)
        remaining_indices.difference_update(blob.pixels)
    return blobs

