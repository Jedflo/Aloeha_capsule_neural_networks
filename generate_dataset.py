import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from tensorflow import keras
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import cv2
import pickle
from tensorflow.python.keras.utils.np_utils import to_categorical
import time
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATADIR = "aloeha_dataset/"
CATEGORIES = ["healthy", "rot", "rust"]
ML_PHASE = ["Training", "Testing", "Validation"]
IMG_SIZE = 64
COLOR_MODE = 3

################################################ Getting training data ##########################################

def load_training_data():
  DATADIR_TRAINING = "aloeha_dataset/Training/"
  training_data = []

  def create_training_data():
    print("generating training data:")
    for category in CATEGORIES:
      category_path = os.path.join(DATADIR_TRAINING, category)
      #print(category)
      #print(category_path)
      class_num = CATEGORIES.index(category)
      progress = 0
      for img in os.listdir(category_path):
        progress = progress + 1
        if(progress == 20):
          sys.stdout.write("\r %s |" %category)
        elif(progress == 40):
          sys.stdout.write("\r %s /" %category)
        elif(progress == 60):
          sys.stdout.write("\r %s -" %category)
        elif(progress == 80):
          sys.stdout.write("\r %s \\" %category)
        if(progress>80):
          progress = 0
        try:
          #img_array = cv2.imread(os.path.join(category_path,img),cv2.IMREAD_GRAYSCALE)          
          img_array = cv2.imread(os.path.join(category_path,img))
          #img_array = cv2.imread(os.path.join(category_path,img),cv2.COLOR_RGB2HSV)
          img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
          new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
          
          training_data.append([new_array, class_num])
        except Exception as e:
          pass
      sys.stdout.write("\r %s done" %category)
      print()


  create_training_data()

  random.shuffle(training_data)

  # for sample in training_data[:10]:
  #   print(sample[1])

  x_train = []
  y_train = []

  for features, labels in training_data:
    x_train.append(features)
    y_train.append(labels)

  x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_MODE).astype("float32")/255.
  y_train = np.array(y_train)
  y_train = tf.one_hot(y_train, depth=3, name="one_hot_aloe")
  y_train = np.array(y_train)



  # print(type(x_train))
  # print(x_train.shape)
  # print(len(x_train))
  # print(len(y_train))

  # i = 0;
  # for img in x_train[:5]:
  #   print(y_train[i], end = '\r')
  #   cv2.imshow('test',img)
  #   cv2.waitKey(1000)
  #   i = i + 1
  # cv2.destroyAllWindows()
  
  return (x_train, y_train)

####################################################### Getting testing data ##############################################

def load_test_data():
  DATADIR_TESTING = "aloeha_dataset/Testing/"

  testing_data = []


  def create_testing_data():
    print("generating testing data:")
    for category in CATEGORIES:
      category_path = os.path.join(DATADIR_TESTING, category)
      #print(category)
      #print(category_path)
      class_num = CATEGORIES.index(category)
      progress = 0
      for img in os.listdir(category_path):
        progress = progress + 1
        if(progress == 10):
          sys.stdout.write("\r %s |" %category)
        elif(progress == 20):
          sys.stdout.write("\r %s /" %category)
        elif(progress == 30):
          sys.stdout.write("\r %s -" %category)
        elif(progress == 40):
          sys.stdout.write("\r %s \\" %category)
        if(progress>40):
          progress = 0
        try:
          #img_array = cv2.imread(os.path.join(category_path,img),cv2.IMREAD_GRAYSCALE)
          img_array = cv2.imread(os.path.join(category_path,img))
          img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
          new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))     
          testing_data.append([new_array, class_num])
        except Exception as e:
          pass
      sys.stdout.write("\r %s done" %category)
      print()


  create_testing_data()

  random.shuffle(testing_data)

  # for sample in testing_data[:10]:
  #   print(sample[1])

  x_test = []
  y_test = []

  for features, labels in testing_data:
    x_test.append(features)
    y_test.append(labels)

  x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_MODE).astype("float32")/255.
  y_test = np.array(y_test)
  y_test = tf.one_hot(y_test, depth=3, name="one_hot_aloe")
  y_test = np.array(y_test)

  # print(type(x_test))
  # print(x_test.shape)
  # print(len(x_test))
  # print(type(y_test))
  # print(y_test.shape)
  

  # i = 0;
  # for img in x_train[:5]:
  #   print(y_train[i], end = '\r')
  #   cv2.imshow('test',img)
  #   cv2.waitKey(1000)
  #   i = i + 1
  # cv2.destroyAllWindows()

  return (x_test, y_test)

# ##################################################### Getting validation data #################################################

# def load_validation_data():

#   DATADIR_VALIDATION = "aloeha_dataset/Validation/"

#   validation_data = []


#   def create_validation_data():
#     for category in CATEGORIES:
#       print(category)
#       category_path = os.path.join(DATADIR_VALIDATION, category)
#       print(category_path)
#       class_num = CATEGORIES.index(category)
#       for img in os.listdir(category_path):
      
#         try:
#           img_array = cv2.imread(os.path.join(category_path,img))
#           new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#           validation_data.append([new_array, class_num])
#         except Exception as e:
#           pass

#   create_validation_data()

#   print(len(validation_data))

#   random.shuffle(validation_data)

#   for sample in validation_data[:10]:
#     print(sample[1])

#   x_valid = []
#   y_valid = []

#   for features, labels in validation_data:
#     x_valid.append(features)
#     y_valid.append(labels)

#   x_valid = np.array(x_valid).reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_MODE).astype("float32")/255.
#   y_valid = np.array(y_valid)
#   y_valid = tf.one_hot(y_valid, depth=3, name="one_hot_aloe")
#   y_valid = np.array(y_valid)

#   print(type(x_valid))
#   print(x_valid.shape)
#   print(len(x_valid))
#   print(type(y_valid))
#   print(y_valid.shape)

#   # i = 0;
#   # for img in x_valid[:5]:
#   #   print(y_valid[i])
#   #   # cv2.imshow('test',img)
#   #   # cv2.waitKey(0)
#   #   i = i + 1
  
#   return (x_valid, y_valid)

# (x_train, y_train)=load_training_data()
# (x_test, y_test)=load_test_data()
# # (x_valid, y_valid)=load_validation_data()

# print("x_train")
# print("shape:", x_train.shape)
# print("type:", type(x_train))
# # print(x_train[0])


# print("y_train")
# print("shape:",y_train.shape)
# print("type:", type(y_train))
# print(y_train[0])
# i=0
# # for img in x_train[:5]:
# #   print(y_train[i])
# #   # cv2.imshow('test',img)
# #   # cv2.waitKey(0)
# #   i = i + 1

# print("x_test")
# print(x_test.shape)
# print(type(x_test))
# print(len(x_test))
# # #print(x_test[0])


# print("y_test")
# print("shape:", y_test.shape)
# print("type:", type(y_test))
# print(len(y_test))
# #print(y_test[0])
# i=0
# for img in x_test[:5]:
#   print(y_test[i])
#   # cv2.imshow('test',img)
#   # cv2.waitKey(0)
#   i = i + 1

# print("x_valid")
# print(x_valid.shape)
# print(type(x_valid))
# #print(x_test[0])


# print("y_valid")
# print(y_valid.shape)
# print(type(y_valid))
# #print(y_test[0])

# print("x_train", x_train.shape)
# print("y_ train", y_train.shape)
# print("x_test", x_test.shape)
# print("y_test", y_test.shape)
# print()
# print("x_test")
# print("shape:", x_test.shape)
# print("type:", type(x_test))
# # print(x_test[0])


# print("y_test")
# print("shape:",y_test.shape)
# print("type:", type(y_test))

# print("x_train")
# print("shape:", x_train.shape)
# print("type:", type(x_train))
# # print(x_train[0])


# print("y_train")
# print("shape:",y_train.shape)
# print("type:", type(y_train))







