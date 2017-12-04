"""
import image from directory
and use previously generated model for clarification
"""

import keras
import numpy as np
from keras.preprocessing import image
import os

classifier = keras.models.load_model("model3.h5")
img_dim_x = 64
img_dim_y = 64


def i_pred (full_name):
    test_image = image.load_img(full_name, target_size = (img_dim_x, img_dim_y))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    return result



path = 'dataset/test_set/man/'


for filename in os.listdir(path):
    result = i_pred(path+filename)
    if result[0][2] == 1:
       prediction = 'woman'
    elif result[0][0] == 1 :
       prediction = 'man'
    else :
        prediction = 'alien'

    print (filename,prediction)
