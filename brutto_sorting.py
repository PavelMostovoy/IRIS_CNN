

import datetime
import keras
import numpy as np
from keras.preprocessing import image
import os

classifier = keras.models.load_model("model_face_2.h5")
img_dim_x = 100
img_dim_y = 100

def i_pred (full_name):
    test_image = image.load_img(full_name, target_size = (img_dim_x, img_dim_y))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    #print (result)
    return result


path = 'dataset/sorting/'
r_file_name = ' '


total = 0
for filename in os.listdir(path):
    total +=1
    try:
        result = i_pred(path + filename)
        if result[0][0] == 1:
           prediction = 'woman'
           os.rename(path+filename,
                     path + prediction + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")).replace(
                         ' ', '_').replace(':', '_') + str(
                         total) + '.png')

        else :
            prediction = 'man'
            os.rename(path + filename,
                      path + prediction + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")).replace(
                          ' ', '_').replace(':','_') + str(
                          total) + '.png')
    except:
        pass

