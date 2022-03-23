import numpy as np
from process import ValidationResult, save_image, load_image
from parameter_estimation import do_parameter_est

def main():
    import os 
    import sys

    # image = load_image(os.path.join(sys.path[0], 'output.bak/2020-01-20-10-36-56_image_correct2.png'))
    # diffed = load_image(os.path.join(sys.path[0], 'output.bak/2020-01-20-10-36-56_image_correct2_diff.png'))
    image = load_image(os.path.join(sys.path[0], 'output.bak/2020-01-10-10-36-57_image_correct0.png'))
    diffed = load_image(os.path.join(sys.path[0], 'output.bak/2020-01-10-10-36-57_image_correct0_diff.png'))
    

    image_B = diffed[:, :, 0]
    image_G = diffed[:, :, 1]
    image_R = diffed[:, :, 2]

    brighest_green = (image.shape[0] // 2, image.shape[1] // 2)
    brightest_blue = np.unravel_index(np.argmax(image_B), image_B.shape)
    brightest_red = np.unravel_index(np.argmax(image_R), image_R.shape)

    image_B[brightest_red[0], brightest_red[1]] = 1
    image_R[brightest_blue[0], brightest_blue[1]] = 1
    save_image(diffed, 'tmp.png')

    result = ValidationResult(
        valid=True,
        green_center=brighest_green,
        blue_center=brightest_blue,
        red_center=brightest_red,
        radius=15
    )

    do_parameter_est(image, diffed, result, "parameters.txt")


if __name__ == '__main__':
    main()
