# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

img_dim_x = 100
img_dim_y = 100
filters = 64

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters, (5, 5), input_shape = (img_dim_x, img_dim_y, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(filters, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid')) # sigmoid for binary outcome softmax fpr more than 2 units


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])#cathegorica_crosentropy


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.3,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_face',
                                                 target_size = (img_dim_x, img_dim_y),
                                                 batch_size = 4,
                                                 class_mode = 'binary') # save_to_dir="dataset/augemented" - this could show generated files

test_set = test_datagen.flow_from_directory('dataset/test_face',
                                            target_size = (img_dim_x, img_dim_y),
                                            batch_size =4,
                                            class_mode = 'binary')# was binary can be sparse

classifier.fit_generator(training_set,
                         steps_per_epoch = 4000,
                         epochs = 30,
                         validation_data = test_set,
                         validation_steps = 400) # steps per epoch -  Total number of steps (batches of samples)

classifier.save("model_face_2.h5")   # save model for future purposes

import numpy as np
from keras.preprocessing import image

'''test_image = image.load_img('dataset/single_prediction/test.jpg', target_size = (64, 64)) # load image and convert to apropriate format
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'man'
else:
    prediction = 'woman'

print (prediction)
'''