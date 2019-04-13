''' Module responsible with dissecting game picking screen and return useful information '''

import math
from PIL import Image, ImageOps
from keras.preprocessing.image import img_to_array
from image import get_avg_image_channel_diff, image_crop_to_square


def get_player_images(window):
    ''' Returns all player avatars from picking screen image for use with hero classifier '''
    images = []
    selected = []

    for i in range(10):
        hero_imgarr = get_player_image(window, i)
        hero_selected = is_hero_avatar_finalized(hero_imgarr)
        hero_imgarr = image_crop_to_square(hero_imgarr)
        hero_image = Image.fromarray(hero_imgarr)
        hero_image = hero_image.convert("L")

        if not hero_selected:
            hero_image = ImageOps.autocontrast(hero_image)

        hero_image = hero_image.resize((96, 96), Image.BICUBIC)
        # hero_image.show()

        hero_imgarr = img_to_array(hero_image)
        hero_imgarr = hero_imgarr.reshape((1,) + hero_imgarr.shape)

        images.append(hero_imgarr)
        selected.append(hero_selected)

    return images, selected


def get_player_image(screen, index, removeband=True):
    '''
    Returns player avatar from picking screen image
    by given player index (from left to right) 
    '''

    ratio169 = 16. / 9
    # only support 16:9 ratio for now
    screen_ratio = screen.shape[1] / screen.shape[0]

    if not math.isclose(ratio169, screen_ratio, rel_tol=1e-2):
        print("Error: currently only 16:9 screen ratio is supported")
        return None

    screen_h, screen_w, _ = screen.shape
    margin_w = int(0.091666 * screen_w)
    crop_width = int(0.05 * screen_w)
    crop_height = int(0.0666 * screen_h)
    h_gap = int(0.01458333 * screen_w)

    if index < 5:
        img_hstart = crop_width * index + h_gap * index + margin_w
    else:
        tmp_index = index - 5
        section_width = crop_width * 5 + h_gap * 4
        h_shift = screen_w - margin_w - section_width
        img_hstart = crop_width * tmp_index + h_gap * tmp_index + h_shift

    # crop image
    img = screen[0: crop_height+1, img_hstart: img_hstart+crop_width+1]

    if removeband:
        band_height = int(math.ceil(0.00555 * screen_h))

        # remove band from both top and bottom of the image
        # so we have more stable values
        tmp = img[band_height:crop_height - band_height + 1]

        # remove horizontal band
        if is_hero_avatar_finalized(tmp):
            # cut from top
            img = img[band_height:]
        else:
            # cut from bottom
            img = img[0: crop_height - band_height + 1]

    return img


def get_player_name(screen, index):
    ''' Returns player name (image) from picking screen '''

    ratio169 = 16. / 9
    # only support 16:9 ratio for now
    screen_ratio = screen.shape[1] / screen.shape[0]

    if not math.isclose(ratio169, screen_ratio, rel_tol=1e-2):
        print("Error: currently only 16:9 screen ratio is supported")
        return None

    screen_h, screen_w, _ = screen.shape
    margin_w = int(0.091666 * screen_w)
    top = int(0.072222 * screen_h)
    height = int(0.025 * screen_h)
    width = int(0.05520833 * screen_w)
    h_gap = int(0.009375 * screen_w)

    if index < 5:
        img_hstart = width * index + h_gap * index + margin_w
    else:
        tmp_index = index - 5
        section_width = width * 5 + h_gap * 4
        h_shift = screen_w - margin_w - section_width
        img_hstart = width * tmp_index + h_gap * tmp_index + h_shift

    # crop image
    img = screen[top: top+height+1, img_hstart: img_hstart+width+1]

    return img


def is_hero_avatar_finalized(image):
    ''' Returns 1 if hero is selected based on average image channel difference otherwise 0 '''
    return 1 if get_avg_image_channel_diff(image) > 10 else 0
