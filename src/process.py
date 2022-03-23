from datetime import datetime
import gc
from itertools import count
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NewType, Sequence, cast

import cv2 # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import math

from lib.color import linear_to_srgb, srgb_to_linear
from lib.timeit import timeit
from lib.weighted_center import weighted_center


# Controls whether to use a faster approximate SRGB-conversion (3x faster)
fast_srgb_conv = True

# Controls whether we process images in parallel
parallel_processing = False

# Number of pixels to pad on each edge of the image
padding = 64

# Minimum original green pixel intensity
threshold = 0.1

### Controls whether to generate output images in the `output/` folder.

# Flag to disable all image saving
disable_all_saving = False

# Output full image with overlaid red/yellow/green rectangles
generate_highlights = True
# Output crops of correct detections (true positives)
generate_correct = True
# Output crops of missed detections (false negatives)
generate_missed = True
# Output crops of wrong detections (false positives)
generate_wrong = True


OriginalImage = NewType('OriginalImage', np.ndarray)
DiffedImage = NewType('DiffedImage', np.ndarray)
Coordinate = tuple[int, int]
Prediction = tuple[Coordinate, float]


output_path: Path


def load_labels(path: Path) -> set[Coordinate]:
    coords: set[Coordinate] = set()
    with open(path) as f:
        for line in f:
            a, b = line.split(maxsplit=2)
            coords.add((int(a) + padding, int(b) + padding))
    return coords


@timeit
def load_image(path) -> OriginalImage:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR) / 255.0
    image = srgb_to_linear(image, fast=fast_srgb_conv)
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)))
    return image


def save_image(image: np.ndarray, path) -> None:
    if disable_all_saving:
        return
    image = np.clip(image, 0, 1)
    image = linear_to_srgb(image, fast=fast_srgb_conv)
    image *= 255
    cv2.imwrite(str(path), image)


def diff_channels(image: OriginalImage) -> DiffedImage:
    """Diff-process the image by subtracting each channel from
    all the others. Highlights pure-colors and movement."""
    num_channels = image.shape[2]
    channel_indices = set(range(num_channels))
    res = image.copy()
    # For each channel ci, subtract the other channels cj
    for ci in channel_indices:
        for cj in channel_indices - {ci}:
            res[:,:,ci] -= image[:,:,cj]
    # Cap to [0, inf)
    return cast(DiffedImage, np.maximum(res, 0))


def find_bright_pixels(image: OriginalImage, threshold: float, best_n: int) -> list[Prediction]:
    """Finds the `best_n` brightest green pixels after diff-processing in the given image."""

    diffed = diff_channels(image)
    save_image(diffed, 'diffed.png')

    # Boolean mask array og bright green pixels
    bright_greens = diffed[:,:,1] >= threshold
    # Indices of bright green pixels
    np_indices = np.argwhere(bright_greens)

    # Change data type from np arrays to tuples
    indices: list[Coordinate] = \
        list(map(lambda idx: (idx[0], idx[1]), np_indices))
    # Pair each index with its green pixel value
    pairs: list[Prediction] = \
         list(map(lambda idx: (idx, diffed[idx[0], idx[1], 1]), indices))

    # Return brightest indices 
    best_indices = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    return best_indices[:best_n]


def png_to_txt(path: Path) -> Path:
    """Generates the expected .txt annotation filename for the given .png filename."""
    return path.with_stem(path.stem.removesuffix('_image')).with_suffix('.txt')


def load_dataset_paths(dir, recursive: bool = True) -> list[tuple[Path, Path]]:
    """Recursively scans the filesystem for *.png and *.txt files.
    Returns a list of tuples consisting of the .png file and
    corresponding .txt annotation file."""
    pairs: list[tuple[Path, Path]] = []
    texts: set[Path] = set()
    for entry in Path(dir).iterdir():
        if entry.is_dir() and recursive:
            pairs.extend(load_dataset_paths(entry, True))
        if entry.is_file():
            if entry.suffix == '.txt':
                texts.add(entry)
            elif (
                entry.suffix == '.png' 
                and (text := png_to_txt(entry)) in texts
            ):
                texts.remove(text)
                pairs.append((entry, text))
    return pairs


def evaluate_prediction(prediction: Coordinate, labels: set[Coordinate]) -> tuple[Coordinate, float]:
    """Loop through all labels and find the closest one."""

    if not labels:
        raise ValueError('Empty set of labels.')

    x, y = prediction

    min_dist = np.inf
    label: Coordinate = None # type: ignore

    # Find the closest label by Pythagorean distance
    for (lx, ly) in labels:
        dx, dy = lx - x, ly - y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist < min_dist:
            min_dist = dist
            label = (lx, ly)

    return (label, min_dist)


def rect(cx: int, cy: int, radius: int, image: np.ndarray) -> tuple[int, int, int, int]:
    """Turn a center coordinate and side length into
    a tuple of two of the corners of the rectangle."""
    return (
        max(0, cx - radius), min(image.shape[0], cx + radius + 1),
        max(0, cy - radius), min(image.shape[1], cy + radius + 1)
    )


def save_prediction(image, center: Coordinate, image_name: str) -> None:
    size = 32
    lx, hx, ly, hy = rect(*center, size, image)
    cropped = image[lx:hx, ly:hy,:]
    save_image(cropped, output_path / f'{image_name}.png')


def paint_rect(image: OriginalImage, center: Coordinate, color: np.ndarray) -> None:
    size = 32
    width = 2
    lx, hx, ly, hy = rect(*center, size + width, image)
    image[lx:hx, ly:ly+width+1, :] = color
    image[lx:hx, hy:hy+width+1, :] = color
    image[lx:lx+width+1, ly:hy, :] = color
    image[hx:hx+width+1, ly:hy, :] = color


def hightlight_predictions(image: OriginalImage, diffed: DiffedImage, correct: Sequence[Prediction], wrong: Sequence[Prediction], labels: Sequence[Coordinate], image_name: str):
    if not (
        generate_wrong or generate_correct or 
        generate_missed or generate_highlights
    ):
        return

    if generate_highlights:
        painted = image.copy()

    for i, pred in enumerate(correct):
        if generate_correct:
            save_prediction(image, pred[0], f'{image_name}_correct{i}')
            save_prediction(diffed, pred[0], f'{image_name}_correct{i}_diff')
        if generate_highlights:
            paint_rect(painted, pred[0], np.array([0, 1, 0]))
    for i, pred in enumerate(wrong):
        if generate_wrong:
            save_prediction(image, pred[0], f'{image_name}_wrong{i}')
            save_prediction(diffed, pred[0], f'{image_name}_wrong{i}_diff')
        if generate_highlights:
            paint_rect(painted, pred[0], np.array([0, 0, 1]))
    for i, lbl in enumerate(labels):
        if generate_missed:
            save_prediction(image, lbl, f'{image_name}_missed{i}')
            save_prediction(diffed, lbl, f'{image_name}_missed{i}_diff')
        if generate_highlights:
            paint_rect(painted, lbl, np.array([0, 1, 1]))
    
    if generate_highlights:
        save_image(painted, output_path / f'{image_name}_painted.png')


def clear_region(diffed: DiffedImage, center: Coordinate, radius: int) -> None:
    x, y = center
    diffed[x-radius:x+radius, y-radius:y+radius, 1] = 0


@dataclass
class ValidationResult:
    valid: bool
    green_center: Coordinate
    red_center: Coordinate
    blue_center: Coordinate
    radius: float

def distance(a,b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def is_between(a,c,b):
    return (distance(a,b) - 1 <= distance(a,c) + distance(c,b) and distance(a,c) + distance(c,b) <= distance(a,b) + 1) and (distance(a,c) - 3 <= distance(c,b) and distance(c,b) <= distance(a,c) + 3) and (distance(a,c) >= 3 and distance(c,b) >= 3)

def print_values(a,c,b):
    print(distance(a,c))
    print(distance(c,b))
    print(distance(a,c) + distance(c,b))
    print(distance(a,b))

def validate_prediction(
    image: OriginalImage,
    diffed: DiffedImage,
    coord: Coordinate
) -> ValidationResult:
    """Perform validation on the given prediction
    and return some estimates about the plane."""
    x, y = coord

    size = 32
    lx, hx, ly, hy = rect(x, y, size, image)
    cropped = image[lx:hx, ly:hy,:]
    cropped_diffed = diffed[lx:hx, ly:hy,:]

    image_B = cropped[:, :, 0]
    image_G = cropped[:, :, 1]
    image_R = cropped[:, :, 2]

    diff_B = cropped_diffed[:, :, 0]
    diff_G = cropped_diffed[:, :, 1]
    diff_R = cropped_diffed[:, :, 2]

    g = np.unravel_index(np.argmax(diff_G), diff_G.shape)

    diff_B = cropped_diffed[:, :, 0]
    b = np.unravel_index(np.argmax(diff_B), diff_B.shape)

    diff_R = cropped_diffed[:, :, 2]
    r = np.unravel_index(np.argmax(diff_R), diff_R.shape)

    while distance(g, b) > 15:
        diff_B[b] = 0
        b = np.unravel_index(np.argmax(diff_B), diff_B.shape)
        if b == (0,0):
            break

    while distance(g, r) > 15:
        diff_R[r] = 0
        r = np.unravel_index(np.argmax(diff_R), diff_R.shape)
        if r == (0,0):
            break

    return ValidationResult(
        valid=is_between(b,g,r),
        green_center=(x, y),
        red_center=(0, 0), # todo: estimate position in red channel
        blue_center=(0, 0), # todo: estimate position in blue channel
        radius=0 # todo: estimate size (radius) of plane
    )
    

def find_plane(diffed: DiffedImage) -> Prediction:
    """Find the plane in a diffed image by returning the brightest green pixel."""
    green_channel = diffed[:, :, 1]
    (x, y) = np.unravel_index(np.argmax(green_channel), green_channel.shape)
    return (x, y), green_channel[x, y] # type: ignore


def categorize_predictions(
    predictions: list[Prediction],
    labels: set[Coordinate],
    max_dist: float
) -> tuple[
    list[Prediction], # correct predictions
    list[Prediction], # wrong predictions
    set[Coordinate] # remaining labels
]:
    labels = set(labels)
    correct: list[Prediction] = []
    wrong: list[Prediction] = []
    while labels and predictions:
        # Pop off top prediction
        pred, *predictions = predictions
        label, dist = evaluate_prediction(pred[0], labels)
        if dist <= max_dist:
            correct.append(pred)
            labels.remove(label)
        else:
            wrong.append(pred)
    # Remaining predictions must be wrong
    wrong.extend(predictions)
    return correct, wrong, labels


def find_planes(
    image: OriginalImage,
    diffed: DiffedImage,
    clear_radius: int
) -> tuple[list[Prediction], list[ValidationResult]]:
    """Find `n` planes in the given image, clearing an square of 
    `clear_radius` side for each detection."""
    predictions: list[Prediction] = []
    validations: list[ValidationResult] = []

    X, Y = np.nonzero(image[:, :, 1] >= threshold)
    values = image[X, Y, 1]

    # Sort by descending intensity
    sorted_indices = values.argsort()[::-1]
    X, Y = X[sorted_indices], Y[sorted_indices]
    visited = np.zeros(image.shape[:2], dtype=bool)

    # Iterate through all pixels matching the threshold
    # in descending order of intensity
    for coord in zip(X, Y):
        if visited[coord]:
            continue
        
        lx, hx, ly, hy = rect(*coord, clear_radius, image)
        visited[lx:hx, ly:hy] = 1
        
        validation = validate_prediction(image, diffed, coord)
        if validation.valid:
            predictions.append(coord)
            validations.append(validation)

    return predictions, validations


@dataclass
class ImageResults:
    num_correct: int
    num_total: int
    predictions: tuple[Prediction, ...]
    validations: tuple[ValidationResult, ...]
    labels: tuple[Coordinate, ...]
    correct: tuple[Prediction, ...]
    wrong: tuple[Prediction, ...]
    remaining_labels: tuple[Coordinate, ...]


def test_image_predictions(
    image: OriginalImage,
    diffed: DiffedImage,
    labels: set[Coordinate],
    radius: int
) -> ImageResults:
    n_labels = len(labels)

    predictions, validations = find_planes(image, diffed, clear_radius=radius)
    correct, wrong, remaining_labels = categorize_predictions(predictions, labels, max_dist=radius)

    return ImageResults(
        num_correct=len(correct),
        num_total=n_labels,
        predictions=tuple(predictions),
        validations=tuple(validations),
        labels=tuple(labels),
        correct=tuple(correct),
        wrong=tuple(wrong),
        remaining_labels=tuple(remaining_labels)
    )


pixel_value_distribution: list[float] = []
pixel_value_distribution_diff: list[float] = []

def evaluate_dataset_entry(pair: tuple[Path, Path]) -> tuple[int, int]:
    png, txt = pair

    n_correct = 0
    n_total = 0

    labels = load_labels(txt)
    # Check if there are any planes in the image
    if labels:
        if not parallel_processing:
            print(f'{png.name} ({png.stat().st_size / 1024**2:.1f} MiB): ', end='')

        image = load_image(png)
        diffed = diff_channels(image)   

        for (x, y) in labels:
            pixel_value_distribution.append(image[x, y, 1])
            pixel_value_distribution_diff.append(diffed[x, y, 1])

        res = test_image_predictions(image, diffed, labels, radius=5)
        n_correct = res.num_correct
        n_total = res.num_total

        if not parallel_processing:
            print(f'{res.num_correct} / {res.num_total}')
            print(f'    Labels: {res.labels}')
            print(f'    Predictions: {res.predictions}')

        for validation_result in res.validations:
            # do_parameter_est(image, diffed, validation_result, f'{png.stem}_parameters.txt')
            pass

        hightlight_predictions(image, diffed, res.correct, res.wrong, res.remaining_labels, png.stem)

    gc.collect()
    return (n_correct, n_total)


def test_dataset(dir: Path):
    """Evaluate accuracy on a whole dataset."""

    image_label_pairs = load_dataset_paths(dir, True)
    print(f'Found {len(image_label_pairs)} images.')

    output_path = Path('output') / (datetime.now().isoformat(timespec='seconds') + f'-{dir.name}')
    output_path.mkdir(exist_ok=True)

    # Sort by smallest filesize first
    image_label_pairs.sort(key=lambda pair: pair[0].stat().st_size)

    if parallel_processing:
        with multiprocessing.Pool() as pool:
            res = pool.map(evaluate_dataset_entry, image_label_pairs)
    else:
        res = list(map(evaluate_dataset_entry, image_label_pairs))

    pixel_value_distribution.sort()
    pixel_value_distribution_diff.sort()
    x = np.linspace(0, 1, len(pixel_value_distribution))
    plt.plot(x, pixel_value_distribution, label='Pre-filter intensitet')
    plt.plot(x, pixel_value_distribution_diff, label='Post-filter intensitet')
    residue_abs = np.array(pixel_value_distribution) - np.array(pixel_value_distribution_diff)
    residue_rel = residue_abs / np.array(pixel_value_distribution)
    plt.plot(x, residue_abs, '--', label='Absolutt filter tap')
    plt.plot(x, residue_rel, '--', label='Relativt filter tap')
    plt.legend()
    plt.title('Intensitetsfordeling (grønn kanal)')
    plt.savefig('pixel_intensity_distribution.pdf')
    
    sum_correct = 0
    sum_total = 0
    for n_correct, n_total in res:
        sum_correct += n_correct
        sum_total += n_total

    accuracy = sum_correct / sum_total
    print(f'{sum_correct} / {sum_total}')
    print(f'{accuracy=}')

def process_single_image(path: Path):
    image = load_image(path)
    coords = find_bright_pixels(image, threshold=0.01, best_n=10)
    print('\n'.join(map(str, coords)))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: process.py <image or directory>', file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    if path.is_dir():
        test_dataset(path)

    elif path.is_file():
        process_single_image(path)

    else:
        print('Invalid filetype.')
