from dataclasses import dataclass
from typing import Iterable
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

def main():
    image = load_image('output.bak/2020-01-10-10-36-57_image_correct0.png')
    diffed = load_image('output.bak/2020-01-10-10-36-57_image_correct0_diff.png')
    
    image_B = image[:, :, 0]
    image_G = image[:, :, 1]
    image_R = image[:, :, 2]

    diff_B = diffed[:, :, 0]
    diff_G = diffed[:, :, 1]
    diff_R = diffed[:, :, 2]

    cr, cc = (image.shape[0] // 2, image.shape[1] // 2)

    cr, cc = np.unravel_index(np.argmax(diff_B), diff_B.shape)
    bright_blue = image_B >= (0.50 * image_B[cr, cc])
    image_R[bright_blue] = 1
    save_image(image, 'tmp.png')

    plt.plot(image_B[cr, :], label='Original row')
    plt.plot(image_B[:, cc], label='Original column')
    plt.plot(diff_B[cr, :], label='Diffed row')
    plt.plot(diff_B[:, cc], label='Diffed column')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
