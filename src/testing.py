from dataclasses import dataclass
from typing import Iterable
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lib.weighted_center import weighted_center # type: ignore
from process import save_image, load_image


def find_plane(channel: np.ndarray, channel_diff: np.ndarray, threshold: float, key):
    guess = np.unravel_index(np.argmax(channel_diff), channel_diff.shape)
    pos = weighted_center(channel_diff, guess) # type: ignore
    save_image(channel, f'testing/chan_{key}.png')
    save_image(channel_diff, f'testing/chan_diff_{key}.png')
    return pos

def main():
    image = load_image('output.bak/2020-01-10-10-36-57_image_correct0.png')
    diffed = load_image('output.bak/2020-01-10-10-36-57_image_correct0_diff.png')

    save_image(image, 'testing/image.png')
    save_image(diffed, 'testing/diffed.png')

    scale = 5
    subsampled = cv2.resize(image, (diffed.shape[0] * scale, diffed.shape[1] * scale), interpolation=cv2.INTER_NEAREST)
    
    positions: list[tuple[int, int]] = []

    for channel in [0, 1, 2]:
        (r, c) = find_plane(
            image[:,:,channel],
            diffed[:,:,channel],
            threshold=0.30,
            key=channel,
        )
        scaled = round(r * scale), round(c * scale)
        print(f'{channel}: {r:.1f}, {c:.1f}')
        subsampled[scaled[0], scaled[1], :] = 1
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


if __name__ == '__main__':
    main()
