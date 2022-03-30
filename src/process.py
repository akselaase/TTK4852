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
from lib.weighted_center import weighted_center


# Controls whether to use a faster approximate SRGB-conversion (3x faster)
fast_srgb_conv = True

# Controls whether we process images in parallel
parallel_processing = False

# Number of pixels to pad on each edge of the image
padding = 64

# Minimum original green pixel intensity
threshold = 0.1
# Minimum diffed green pixel intensity
diffed_threshold = 0.0000001 # anything above zero

# Amount to scale images by when saving
# (to better visualize exact plane coordinate)
supersampling = 10

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

# Output path for all saved images
output_path = Path('output')


OriginalImage = NewType('OriginalImage', np.ndarray)
DiffedImage = NewType('DiffedImage', np.ndarray)
CoordinateI = tuple[int, int]
CoordinateF = tuple[float, float]


@dataclass
class Prediction:
    green_center_f: CoordinateF
    red_center_f: CoordinateF
    blue_center_f: CoordinateF
    radius: float

    @property
    def blue_center_i(self) -> CoordinateI:
        x, y = map(round, self.blue_center_f)
        return (x, y)

    @property
    def green_center_i(self) -> CoordinateI:
        x, y = map(round, self.green_center_f)
        return (x, y)

    @property
    def red_center_i(self) -> CoordinateI:
        x, y = map(round, self.red_center_f)
        return (x, y)

    def __str__(self) -> str:
        g = self.green_center_f
        b = self.blue_center_f
        r = self.red_center_f
        return ('Prediction'
            f'(g=({g[0]:.1f}, {g[1]:.1f}), '
            f'(b=({b[0]:.1f}, {b[1]:.1f}), '
            f'(r=({b[0]:.1f}, {r[1]:.1f}))'
        )


print_lock = multiprocessing.Lock()


def load_labels(path: Path) -> set[CoordinateI]:
    coords: set[CoordinateI] = set()
    with open(path) as f:
        for line in f:
            a, b = line.split(maxsplit=2)
            coords.add((int(a) + padding, int(b) + padding))
    return coords


def load_image(path, pad=True) -> OriginalImage:
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


def png_to_txt(path: Path) -> Path:
    """Generates the expected .txt annotation filename for the given .png filename."""
    return path.with_stem(path.stem.removesuffix('_image')).with_suffix('.txt')


def load_dataset_paths(dir, recursive: bool = True) -> list[tuple[Path, Path]]:
    """Recursively scans the filesystem for *.png and *.txt files.
    Returns a list of tuples consisting of the .png file and
    corresponding .txt annotation file."""
    pairs: list[tuple[Path, Path]] = []
    txts: set[Path] = set()
    pngs: set[Path] = set()
    for entry in Path(dir).iterdir():
        if entry.is_dir() and recursive:
            pairs.extend(load_dataset_paths(entry, True))
        if entry.is_file():
            if entry.suffix == '.txt':
                txts.add(entry)
            elif entry.name.endswith('_image.png'):
                pngs.add(entry)
    for png in pngs:
        txt = png_to_txt(png)
        if txt in txts:
            pairs.append((png, txt))
    return pairs


def evaluate_prediction(prediction: CoordinateF, labels: set[CoordinateI]) -> tuple[CoordinateI, float]:
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


def save_prediction(image, pred: Prediction | CoordinateI, image_name: str) -> None:
    size = 32

    if isinstance(pred, Prediction):
        center = pred.green_center_i
    else:
        center = pred

    lx, hx, ly, hy = rect(*center, size, image)
    offset = np.array((lx, ly))
    cropped = image[lx:hx, ly:hy,:]

    if isinstance(pred, Prediction):
        # Supersample, draw line and highlight centers
        b = tuple(round(f * supersampling) for f in pred.blue_center_f - offset)
        g = tuple(round(f * supersampling) for f in pred.green_center_f - offset)
        r = tuple(round(f * supersampling) for f in pred.red_center_f - offset)
        r_b = tuple(reversed(b))
        r_g = tuple(reversed(g))
        r_r = tuple(reversed(r))
        cropped = cv2.resize(
            cropped,
            (cropped.shape[0] * supersampling, cropped.shape[1] * supersampling),
            interpolation=cv2.INTER_NEAREST)
        cv2.line(cropped, r_b, r_g, (1, 1, 0), 1, cv2.LINE_AA)
        cv2.line(cropped, r_g, r_r, (0, 1, 1), 1, cv2.LINE_AA)
        cropped[b] = np.array([1, 0, 1])
        cropped[g] = np.array([1, 0, 1])
        cropped[r] = np.array([1, 0, 1])

    save_image(cropped, output_path / f'{image_name}.png')


def paint_rect(image: OriginalImage, center: CoordinateI, color: np.ndarray) -> None:
    size = 32
    width = 2
    lx, hx, ly, hy = rect(*center, size + width, image)
    image[lx:hx, ly:ly+width+1, :] = color
    image[lx:hx, hy:hy+width+1, :] = color
    image[lx:lx+width+1, ly:hy, :] = color
    image[hx:hx+width+1, ly:hy, :] = color


def hightlight_predictions(image: OriginalImage, diffed: DiffedImage, correct: Sequence[Prediction], wrong: Sequence[Prediction], labels: Sequence[CoordinateI], image_name: str):
    if not (
        generate_wrong or generate_correct or 
        generate_missed or generate_highlights
    ):
        return

    diffed = diff_channels(image)

    if generate_highlights:
        painted = image.copy()

    for i, pred in enumerate(correct):
        if generate_correct:
            save_prediction(image, pred, f'{image_name}_correct{i}')
            save_prediction(diffed, pred, f'{image_name}_correct{i}_diff')
        if generate_highlights:
            paint_rect(painted, pred.green_center_i, np.array([0, 1, 0]))
    for i, pred in enumerate(wrong):
        if generate_wrong:
            save_prediction(image, pred, f'{image_name}_wrong{i}')
            save_prediction(diffed, pred, f'{image_name}_wrong{i}_diff')
        if generate_highlights:
            paint_rect(painted, pred.green_center_i, np.array([0, 0, 1]))
    for i, lbl in enumerate(labels):
        if generate_missed:
            save_prediction(image, lbl, f'{image_name}_missed{i}')
            save_prediction(diffed, lbl, f'{image_name}_missed{i}_diff')
        if generate_highlights:
            paint_rect(painted, lbl, np.array([0, 1, 1]))
    
    if generate_highlights:
        save_image(painted, output_path / f'{image_name}_painted.png')


def validate_prediction(
    image: OriginalImage,
    diffed: DiffedImage,
    pred: Prediction
):
    b = pred.blue_center_f
    g = pred.green_center_f
    r = pred.red_center_f

    scaling = 0.527 / (1.005 - 0.527) 
    v_b_to_g = np.array(g) - np.array(b)
    v_g_to_r = np.array(r) - np.array(g)
    v_diff = v_b_to_g - scaling * v_g_to_r

    mismatch = np.sqrt(np.sum(v_diff ** 2))
    shortest_distance = np.minimum(
        np.sqrt(np.sum(v_b_to_g**2)),
        np.sqrt(np.sum(v_g_to_r**2))
    )

    good_line = mismatch < 3 and shortest_distance >= 2
    correct_diff = (
        bool(diffed[pred.blue_center_i][0] > 0)
        + bool(diffed[pred.green_center_i][1] > 0)
        + bool(diffed[pred.red_center_i][2] > 0)
    )
    correct_image = (
        bool(np.argmax(image[pred.blue_center_i]) == 0)
        + bool(np.argmax(image[pred.green_center_i]) == 1)
        + bool(np.argmax(image[pred.red_center_i]) == 2)
    )
    return (
        good_line
        and correct_image == 3
        and correct_diff == 3
    )

def make_prediction(
    image: OriginalImage,
    diffed: DiffedImage,
    coord: CoordinateI
) -> Prediction:
    x, y = coord

    size = 16
    lx, hx, ly, hy = rect(x, y, size, image)
    offset = np.array((lx, ly))

    cropped = image[lx:hx, ly:hy,:]
    cropped_diffed = diffed[lx:hx, ly:hy,:]

    image_B = cropped[:, :, 0]
    image_G = cropped[:, :, 1]
    image_R = cropped[:, :, 2]

    diff_B = cropped_diffed[:, :, 0]
    diff_G = cropped_diffed[:, :, 1]
    diff_R = cropped_diffed[:, :, 2]

    # Find initial guesses from diffed max intensities
    g = np.array(coord) - offset
    b = np.unravel_index(np.argmax(diff_B), diff_B.shape)
    r = np.unravel_index(np.argmax(diff_R), diff_R.shape)

    # Refine estimates using an intensity-weighted average
    wg = weighted_center(image_G, g, radius=2) # type: ignore
    wb = weighted_center(image_B, b, radius=2) # type: ignore
    wr = weighted_center(image_R, r, radius=2) # type: ignore

    return Prediction(
        green_center_f=tuple(wg + offset),
        red_center_f=tuple(wr + offset),
        blue_center_f=tuple(wb + offset),
        radius=0 # todo: estimate size (radius) of plane
    )
    

def categorize_predictions(
    predictions: list[Prediction],
    labels: set[CoordinateI],
    max_dist: float
) -> tuple[
    list[Prediction], # correct predictions
    list[Prediction], # wrong predictions
    set[CoordinateI] # remaining labels
]:
    labels = set(labels)
    correct: list[Prediction] = []
    wrong: list[Prediction] = []
    while labels and predictions:
        # Pop off top prediction
        pred, *predictions = predictions
        label, dist = evaluate_prediction(pred.green_center_f, labels)
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
) -> list[Prediction]:
    """Find `n` planes in the given image, clearing an square of 
    `clear_radius` side for each detection."""
    predictions: list[Prediction] = []

    X, Y = np.nonzero(
        (image[:, :, 1] >= threshold) 
        & (diffed[:, :, 1] >= diffed_threshold))
    values = image[X, Y, 1]

    # Sort by descending intensity
    sorted_indices = values.argsort()[::-1]
    X, Y = X[sorted_indices], Y[sorted_indices]
    visited = np.zeros(image.shape[:2], dtype=bool)

    iterations = 0

    # Iterate through all pixels matching the threshold
    # in descending order of intensity
    for coord in zip(X, Y):
        if visited[coord]:
            continue

        iterations += 1
        
        lx, hx, ly, hy = rect(*coord, clear_radius, image)
        visited[lx:hx, ly:hy] = 1

        prediction = make_prediction(image, diffed, coord)
        if validate_prediction(image, diffed, prediction):
            predictions.append(prediction)

    with print_lock:
        print(f'{len(predictions)=} / {iterations=} = {len(predictions)/iterations:.2%}')
    
    return predictions


@dataclass
class ImageResults:
    num_true_positive: int
    num_false_positive: int
    num_false_negative: int
    predictions: tuple[Prediction, ...]
    labels: tuple[CoordinateI, ...]
    missed: tuple[CoordinateI, ...]
    correct: tuple[Prediction, ...]
    wrong: tuple[Prediction, ...]


def test_image_predictions(
    image: OriginalImage,
    diffed: DiffedImage,
    labels: set[CoordinateI],
    radius: int
) -> ImageResults:
    """Find predictions, check which labels they correspond to,
    and return statistics.
    """
    predictions = find_planes(image, diffed, clear_radius=radius)
    correct, wrong, remaining_labels = \
        categorize_predictions(predictions, labels, max_dist=radius)

    return ImageResults(
        num_true_positive=len(correct),
        num_false_positive=len(wrong),
        num_false_negative=len(remaining_labels),
        predictions=tuple(predictions),
        labels=tuple(labels),
        correct=tuple(correct),
        wrong=tuple(wrong),
        missed=tuple(remaining_labels)
    )


pixel_value_distribution: list[float] = []
pixel_value_distribution_diff: list[float] = []

def evaluate_image_labels(
    png_path: Path,
    labels: set[CoordinateI],
    image: OriginalImage
) -> tuple[int, int, int]:
    """Evaluate our algorithm on a specific image with labels.
    Returns a tuple
    (true positive, false positive, false negative, total positive).
    """

    num_labels = len(labels)
    num_tp = 0
    num_fp = 0
    num_fn = 0

    diffed = diff_channels(image)   

    for (x, y) in labels:
        pixel_value_distribution.append(image[x, y, 1])
        pixel_value_distribution_diff.append(diffed[x, y, 1])

    res = test_image_predictions(image, diffed, labels, radius=5)
    num_tp = res.num_true_positive
    num_fp = res.num_false_positive
    num_fn = res.num_false_negative
    assert num_tp + num_fn == num_labels

    with print_lock:
        sep = '\n\t - '
        predictions_str = sep.join(map(str, res.predictions))
        print(f'{png_path.name} ({png_path.stat().st_size / 1024**2:.1f} MiB):')
        print(f'    True positives: {num_tp} / {num_labels}')
        print(f'    False negatives: {num_fn} / {num_labels}')
        print(f'    False positives: {num_fp}')
        print(f'    Labels: {res.labels}')
        print(f'    Predictions: {sep}{predictions_str}')

    hightlight_predictions(image, diffed, res.correct, res.wrong, res.missed, png_path.stem)

    gc.collect()
    return (num_tp, num_fp, num_fn)


def evaluate_dataset_entry(pair: tuple[Path, Path]):
    png, txt = pair
    evaluate_image_labels(
        png,
        load_labels(txt),
        load_image(png)
    )


def test_single_file(file: Path):
    """Make predictions on a single file."""
    evaluate_image_labels(
        file,
        set(),
        load_image(file)
    )


def test_dataset(dir: Path):
    """Evaluate accuracy on a whole dataset."""
    image_label_pairs = load_dataset_paths(dir, True)
    print(f'Found {len(image_label_pairs)} images.')

    # Sort by smallest filesize first
    image_label_pairs.sort(key=lambda pair: pair[0].stat().st_size)

    if parallel_processing:
        with multiprocessing.Pool(2) as pool:
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
    
    sum_tp = 0
    sum_fp = 0
    sum_fn = 0
    for tp, fp, fn in res:
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

    recall = sum_tp / (sum_tp + sum_fn)
    precision = sum_tp / (sum_tp + sum_fp)
    
    print(f'True positives: {sum_tp}')
    print(f'False positives: {sum_fp}')
    print(f'False negatives: {sum_fn}')

    print(f'Recall: {recall:.2%}')
    print(f'Precision: {precision:.2%}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: process.py <image or directory>', file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])

    output_path /= (
        datetime.now()
            .isoformat(timespec='seconds')
            .replace(':', '-')
        + f'-{path.name}'
    )
    output_path.mkdir(exist_ok=True, parents=True)

    if path.is_dir():
        test_dataset(path)

    elif path.is_file():
        test_single_file(path)

    else:
        print('Path must be a directory.')
