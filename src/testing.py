import numpy as np
import matplotlib.pyplot as plt # type: ignore
from process import save_image, load_image

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
