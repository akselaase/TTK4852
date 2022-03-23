from matplotlib.font_manager import findSystemFonts
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import And # type: ignore
from process import find_bright_pixels, save_image, load_image
import math

def main():
    image = load_image('output.new/2020-06-03-10-37-09_image_wrong0.png', False)
    diffed = load_image('output.new/2020-06-03-10-37-09_image_wrong0_diff.png', False)
    # image = load_image('output.new/2020-05-18-10-14-23_image_wrong0.png', False)
    # diffed = load_image('output.new/2020-05-18-10-14-23_image_wrong0_diff.png', False)

    image_B = image[:, :, 0]
    image_G = image[:, :, 1]
    image_R = image[:, :, 2]

    diff_B = diffed[:, :, 0]
    diff_G = diffed[:, :, 1]
    diff_R = diffed[:, :, 2]

    green_channel = diffed[:, :, 1]
    g = np.unravel_index(np.argmax(green_channel), green_channel.shape)

    blue_channel = diffed[:, :, 0]
    b = np.unravel_index(np.argmax(blue_channel), blue_channel.shape)

    red_channel = diffed[:, :, 2]
    r = np.unravel_index(np.argmax(red_channel), red_channel.shape)

    while distance(g, b) > 15:
        blue_channel[b] = 0
        b = np.unravel_index(np.argmax(blue_channel), blue_channel.shape)
        if b == (0,0):
            break

    while distance(g, r) > 15:
        red_channel[r] = 0
        r = np.unravel_index(np.argmax(red_channel), red_channel.shape)
        if r == (0,0):
            break
    

    diff_R[g] = 1
    diff_B[g] = 1
    diff_G[g] = 1
    
    diff_R[b] = 1
    diff_B[b] = 1
    diff_G[b] = 1

    diff_R[r] = 1
    diff_B[r] = 1
    diff_G[r] = 1

    print(g)
    print(b)
    print(r)

    print(is_between(b,g,r))

    # how much longer the blue->green line is than the green->red
    scaling = 0.527 / (1.005 - 0.527) 
    v_b_to_g = np.array(g) - np.array(b)
    print(v_b_to_g)
    v_g_to_r = np.array(r) - np.array(g)
    print(v_g_to_r)

    mismatch = np.sqrt(np.sum((v_b_to_g - v_g_to_r * scaling) ** 2))
    shortest_distance = np.minimum(np.sqrt(np.sum(v_b_to_g**2)), np.sqrt(np.sum(v_g_to_r**2)))
    valid = mismatch < 3 and shortest_distance >= 2

    print(f'{mismatch=}')
    print(f'{shortest_distance=}')
    print(f'{valid=}')

    save_image(diffed, 'tmp.png')

def distance(a,b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def is_between(a,c,b):
    print(distance(a,c))
    print(distance(c,b))
    print(distance(a,c) + distance(c,b))
    print(distance(a,b))

    return (distance(a,b) - 1 <= distance(a,c) + distance(c,b) and distance(a,c) + distance(c,b) <= distance(a,b) + 1) and (distance(a,c) - 3 <= distance(c,b) and distance(c,b) <= distance(a,c) + 3) and (distance(a,c) >= 3 and distance(c,b) >= 3)
    
if __name__ == '__main__':
    main()
