''' Train hero model using this module '''

import os
import glob

from training_data import load_training_image
from heroclassifier import HeroClassifier

weightspath = './resources/models/hero_weights.h5'
classifier = HeroClassifier()
classifier.init_model()
classifier.load(weightspath)
# classifier.train(100)
# classifier.save(weightspath)

images_dir = "./resources/avatars/*.png"
filenames = sorted(glob.glob(images_dir))

print("Testing model: ")

err = False

for x in range(116):
    arr = load_training_image(filenames[x])
    arr = arr.reshape((1,) + arr.shape)
    out = classifier.classify(arr)

    if x != out:
        print("Error: %s Actual: %s" % (filenames[out], filenames[x]))
        err = True

if not err:
    print("No error found")
