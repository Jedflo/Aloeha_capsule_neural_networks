import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

def load_mnist():
    # the data, shuffled and split between train and test sets
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_mnist()
# print("x_train")
# print(type(x_train))
# print(x_train.shape)
# print(x_train[0])
# print("y_train")
# print(type(y_train))
# print(y_train.shape)
# print(y_train[0])
# i=0
# for img in x_train[:5]:
#     print(y_train[i])
# # cv2.imshow('test',img)
# # cv2.waitKey(0)
# i = i + 1
print()
print("x_test")
print(type(x_test))
print(x_test.shape)
print()
print("y_test")
print(type(y_test))
print(y_test.shape)
print(np.argmax(y_test, 1))

validation_data=((x_test, y_test), (y_test, x_test))
print(validation_data[0])


