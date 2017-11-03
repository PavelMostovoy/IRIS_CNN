import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())