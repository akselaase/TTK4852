import sys
from pathlib import Path
import numpy as np
import cv2 # type: ignore


if len(sys.argv) != 2:
    print('Usage: process.py <image_file>', file=sys.stderr)
    sys.exit(1)

def save_channel(data, index, name):
    new_image = np.zeros_like(image)
    new_image[:,:,index] = data
    cv2.imwrite(str(name), np.clip(new_image, 0, 1) * 255)

source_path = Path(sys.argv[1])
image = cv2.imread(str(source_path), cv2.IMREAD_COLOR) / 255.0

gamma = 1.5
image = image ** gamma

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
        channel -= sub_channel * 0.5
    new_path = target_path_base.with_stem(target_path_base.stem + '_' + name + '_sub')
    save_channel(channel, index, new_path)

    combined_image[:,:,index] = channel

cv2.imwrite(str(target_path_base.with_stem(target_path_base.stem + '_combined')), np.clip(combined_image, 0, 1) * 255)

print('Done.')
