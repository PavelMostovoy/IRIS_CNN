"""
import image from directory
and use previously generated model for clarification
"""

import keras
import numpy as np
from keras.preprocessing import image
import os

classifier = keras.models.load_model("model_face.h5")
img_dim_x = 100
img_dim_y = 100


def i_pred (full_name):
    test_image = image.load_img(full_name, target_size = (img_dim_x, img_dim_y))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    #print (result)
    return result



path = 'dataset/single_prediction/'

total = []



for filename in os.listdir(path):
    result = i_pred(path+filename)

    if result[0][0] == 1:
       prediction = 'woman'
       total.append(prediction)
    else :
        prediction = 'man'
        total.append(prediction)

    print (filename,prediction)

print ("w = ",total.count("woman"))

print ("m = ",total.count("man"))

#cathegorical
w =  328
m =  168

# sparse

w =  359
m =  143