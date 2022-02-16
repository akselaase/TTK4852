import sys
from pathlib import Path
import numpy as np
import cv2 # type: ignore


if len(sys.argv) != 2:
    print('Usage: process.py <image_file>', file=sys.stderr)
    sys.exit(1)

def srgb_to_linear(image) -> np.ndarray:
    scaled = image / 12.92
    gamma = np.power((image + 0.055) / 1.055, 2.4)
    return np.where(image <= 0.04045, scaled, gamma)

def linear_to_srgb(image) -> np.ndarray:
    scaled = image * 12.92
    gamma = np.power(image, 1 / 2.4) * 1.055 - 0.055
    return np.where(image <= 0.0031308, scaled, gamma)

def save_channel(data, index, name):
    new_image = np.zeros_like(image)
    new_image[:,:,index] = data
    save_image(new_image, name)

def save_image(image, name):
    image = np.clip(image, 0, 1)
    image = linear_to_srgb(image)
    image *= 255
    cv2.imwrite(str(name), image)


source_path = Path(sys.argv[1])
image = cv2.imread(str(source_path), cv2.IMREAD_COLOR) / 255.0

image = srgb_to_linear(image)

target_path_base = Path(source_path.name)

combined_image = np.zeros_like(image)

for index, name in enumerate('BGR'):
    channel = image[:,:,index]

    print(f'Saving channel {name}')
    new_path = target_path_base.with_stem(target_path_base.stem + '_' + name)
    save_channel(channel, index, new_path)

    print(f'Saving channel {name}_sub')
    channel = channel.copy()
    for sub_index  in range(len('RGB')):
        if sub_index == index:
            continue
        sub_channel = image[:,:,sub_index]
        channel -= sub_channel
    new_path = target_path_base.with_stem(target_path_base.stem + '_' + name + '_sub')
    save_channel(channel, index, new_path)
    if name == 'G':
        green_channel = channel

    combined_image[:,:,index] = channel

save_image(combined_image, target_path_base.with_stem(target_path_base.stem + '_combined'))


print('Done.')
