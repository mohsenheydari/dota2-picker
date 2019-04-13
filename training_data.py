''' Helper module to preprocess training data '''

import os
import glob

from PIL import Image
import numpy as np

from keras.preprocessing.image import load_img, img_to_array

from image import image_crop_to_square


def load_training_image(path):
    img_w = 150
    crop_w = 84  # 84x84 image
    crop_start = int((img_w - crop_w) * .5)
    crop_end = img_w - crop_start

    img = load_img(path, color_mode="grayscale")
    img = img.crop((crop_start, 0, crop_end, crop_w))
    img = img.resize((96, 96), Image.BICUBIC)  # Min size for MobileNet
    return img_to_array(img)


def load_training_data(filenames):
    imgs = []
    labels = []
    num_classes = len(filenames)

    for idx, file in enumerate(filenames):
        img_arr = load_training_image(file)
        imgs.append(img_arr)  # this is a PIL image
        # create one hot encoding labels
        lbl = np.zeros((num_classes))
        lbl[idx] = 1.
        labels.append(lbl)

    # Convert image array to 4D array
    stacked = np.stack(imgs, axis=0)

    return stacked, labels


def convert_to_grayscale_image(image, return_mode="L"):
    # convert to grayscale
    if image.mode != "L":
        image = image.convert("L")

    if image.mode == return_mode:
        return image

    return image.convert(return_mode)


def generate_training_class(file, name, ext, dest, target=96):
    # Hardcode image size for now
    img_w = 150
    img_h = 84
    crop_size = img_h
    slide_len = img_w - crop_size

    orig = load_img(file, color_mode="rgb")

    # stride of 4
    for x in range(0, slide_len, 4):
        img = orig.crop((x, 0, crop_size + x, crop_size))
        img = img.resize((target, target), Image.BICUBIC)
        path = os.path.join(dest, name+'_'+str(x)+ext)

        if not os.path.isfile(path):
            img.save(path, "PNG")


def generate_hero_training_images(dest: str, images_dir: str = "./resources/avatars/*.png"):
    filenames = sorted(glob.glob(images_dir))

    for file in filenames:
        _, name = os.path.split(file)
        name, ext = os.path.splitext(name)

        directory = os.path.join(dest, name)

        if(not os.path.isdir(directory)):
            os.mkdir(directory)

        generate_training_class(file, name, ext, directory)


def crop_images_to_square(source: str, dest: str, ext: str = ".jpg"):
    files = glob.glob(os.path.join(source, f'*{ext}'))

    for file in files:
        _, name = os.path.split(file)
        arr = img_to_array(load_img(file), dtype="uint8")
        arr = image_crop_to_square(arr)
        savepath = os.path.join(dest, name)
        Image.fromarray(arr).save(savepath)


# crop_images_to_square("./resources/pick_screen_training_raw/not/",
#                      "./resources/pick_screen_training/not/")

# crop_images_to_square("./resources/pick_screen_training_raw/pick/",
#                     "./resources/pick_screen_training/pick/")

# generate_hero_training_images('./resources/hero_training/')
