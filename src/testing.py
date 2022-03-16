import numpy as np
from process import save_image, load_image

def main():
    image = load_image('output.bak/2020-01-10-10-36-57_image_correct0.png')
    
    image_B = image[:, :, 0]
    image_G = image[:, :, 1]
    image_R = image[:, :, 2]
    print(image_R[16, 16])

    brightest = np.unravel_index(np.argmax(image_B), image_B.shape)
    print(brightest)
    x, y = brightest
    image_R[x, y] = 1
    save_image(image, 'tmp.png')


if __name__ == '__main__':
    main()
