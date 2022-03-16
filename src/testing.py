import numpy as np
from process import ValidationResult, save_image, load_image
from parameter_estimation import do_parameter_est

def main():
    image = load_image('output.bak/2020-01-10-10-36-57_image_correct0.png')
    diffed = load_image('output.bak/2020-01-10-10-36-57_image_correct0_diff.png')
    
    image_B = image[:, :, 0]
    image_G = image[:, :, 1]
    image_R = image[:, :, 2]

    brighest_green = (image.shape[0] // 2, image.shape[1] // 2)
    brightest_blue = np.unravel_index(np.argmax(image_B), image_B.shape)
    brightest_red = np.unravel_index(np.argmax(image_R), image_R.shape)

    image_B[brightest_red[0], brightest_red[1]] = 1
    image_R[brightest_blue[0], brightest_blue[1]] = 1
    save_image(image, 'tmp.png')

    result = ValidationResult(
        valid=True,
        green_center=brighest_green,
        blue_center=brightest_blue,
        red_center=brightest_red,
        radius=3
    )

    do_parameter_est(image, diffed, result, "parameters.txt")


if __name__ == '__main__':
    main()
