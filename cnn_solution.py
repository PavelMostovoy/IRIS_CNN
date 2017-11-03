"""
import image from directory
and use previously generated model for clarification
"""

import keras
import numpy as np
from keras.preprocessing import image

classifier = keras.models.load_model("model.h5")



test_image = image.load_img('dataset/single_prediction/test.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'woman'
else:
    prediction = 'man'

print (prediction)
