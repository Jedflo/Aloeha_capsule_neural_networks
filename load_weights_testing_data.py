import tensorflow as tf
import preprocessor
import os
import numpy as np
import cv2
from tensorflow.python.keras.utils.np_utils import to_categorical
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATADIR = "aloeha_dataset/"
CATEGORIES = ["healthy", "rot", "rust"]
ML_PHASE = ["Training", "Testing", "Validation"]
IMG_SIZE = 32
COLOR_MODE = 3


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
          # img_array = cv2.imread(os.path.join(category_path,img),cv2.IMREAD_GRAYSCALE)
          img_array = cv2.imread(os.path.join(category_path,img))
          new_array = preprocessor.autocrop(img_array,(IMG_SIZE, IMG_SIZE))
          #new_array = cv2.cvtColor(new_array, cv2.COLOR_BGR2HSV)
          # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))     
          testing_data.append([new_array, class_num])
        except Exception as e:
          pass
      sys.stdout.write("\r %s done" %category)
      print()


  create_testing_data()

  #random.shuffle(testing_data)

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
  

#   i = 0;
#   for img in x_test[:10]:
#     print(y_test[i], end = '\r')
#     cv2.imshow('test',img)
#     cv2.waitKey(1000)
#     i = i + 1
#   cv2.destroyAllWindows()

  return (x_test, y_test)

#load_test_data()
