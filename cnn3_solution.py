"""
import image from directory
and use previously generated model for clarification
"""
import cv2
import keras
import numpy as np
from keras.preprocessing import image
import os

classifier = keras.models.load_model("model_face_cat.h5")
img_dim_x = 100
img_dim_y = 100


def i_pred (full_name):
    test_image = image.load_img(full_name, target_size = (img_dim_x, img_dim_y))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    #print (result)
    return result

def face_predictor(full_name,):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    result = []
    try:
        img = cv2.imread(full_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            if x and y <= 20 : continue
            temp_img = img.copy()
            temp_img = temp_img[y:y + h, x:x + w]  # NOTE: its img[y: y + h, x: x + w]
            temp_img = cv2.resize(temp_img, (img_dim_x, img_dim_y))
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
            temp_img = np.expand_dims(temp_img, axis=0)
            prediction  = classifier.predict(temp_img)
            result.append(prediction)


    finally:
        return result








path = 'dataset/single_prediction/'


total = []



for filename in os.listdir(path):
    result = face_predictor(path+filename)
    prediction = "another"

    for pred in result:
        print (pred)

        if pred[0][1] == 1:
           prediction = 'woman'
           total.append(prediction)
        elif pred[0][0] == 1:
            prediction = 'man'
            total.append(prediction)
        else :
            prediction = 'another'
            total.append(prediction)
        print (filename,prediction)

print ("w = ",total.count("woman"))

print ("m = ",total.count("man"))

print ("a = ",total.count("another"))
