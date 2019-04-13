''' Module contains helper image methods '''

import colorsys


def image_crop_to_square(image):
    ''' Crop numpy array to have 1:1 ratio '''

    shape = image.shape
    width = shape[1]
    height = shape[0]
    diff = abs(width - height)
    start = int(diff * .5)

    if width > height:
        ret = image[:, start:start+height]
    else:
        ret = image[start:start+width]

    return ret


def get_avg_image_channel_diff(image):
    ''' Returns average image channel difference on 3 channels '''

    count = 0
    avg_diff = 0.

    for h in image:
        for w in h:
            count += 1
            d = abs(int(w[0]) - w[1])
            d1 = abs(int(w[1]) - w[2])
            avg_diff += abs(d - d1)

    return avg_diff / count


def get_avg_image_saturation(image):
    ''' Returns average image saturation '''

    count = 0
    avg = 0.

    for h in image:
        for w in h:
            count += 1
            _, s, _ = colorsys.rgb_to_hsv(w[0], w[1], w[2])
            avg += s

    return avg / count


def get_avg_image_value(image):
    ''' Returns average image value '''

    count = 0
    avg = 0.

    for h in image:
        for w in h:
            count += 1
            _, _, v = colorsys.rgb_to_hsv(w[0], w[1], w[2])
            avg += v

    return avg / count


def get_avg_image_hue(image):
    ''' Returns average image value '''

    count = 0
    avg = 0.

    for h in image:
        for w in h:
            count += 1
            hu, _, _ = colorsys.rgb_to_hsv(w[0], w[1], w[2])
            avg += hu

    return avg / count
