''' Train pick screen model using this module '''


from pickscreenclassifier import PickScreenClassifier
import numpy as np

weightspath = './resources/models/screen_weights.h5'
classifier = PickScreenClassifier()
classifier.init_model()
classifier.load(weightspath)

arr = np.zeros((1, 224, 224, 3))
print(classifier.classify(arr))

# classifier.train(50)
# classifier.save('./screen_weights.h5')
