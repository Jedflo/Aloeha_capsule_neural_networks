from re import I
from PIL import Image
import cv2 as cv
import numpy as np
from load_weights_testing_data import load_test_data
from load_weights_capsnet_model import CapsNet, combine_images
import csv


CAPSNET_WEIGHTS = "./result/trained_model.h5"
#CAPSNET_WEIGHTS = "./result/weights-71.h5"
COLOR_MODE = 3
IMG_SIZE = 32
CLASSIFICATION = ["Healthy", "Rot", "Rust"]



x_test, y_test = load_test_data()
model, eval_model, manipulate_model = CapsNet(input_shape=x_test.shape[1:],n_class=len(CLASSIFICATION),routings=3, batch_size=20)

eval_model.load_weights(CAPSNET_WEIGHTS)


# def preprocess(filepath):
    
#     #img_ar = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
#     img_ar = cv.imread(filepath)
#     img_ar = cv.cvtColor(img_ar, cv.COLOR_BGR2HSV)
#     new_ar = cv.resize(img_ar, (IMG_SIZE, IMG_SIZE))
#     return new_ar.reshape(1, IMG_SIZE, IMG_SIZE, COLOR_MODE)

# #input= preprocess("sample inputs/rust/rust.jpg")
# input= preprocess("sample inputs/rot/rot.jpg")
# #input= preprocess("sample inputs/healthy/h (4).jpg")



eval_model.summary()

healthy = []
rust = []
rot = []
yesno = []

prediction, recon = eval_model.predict(x_test, 20)
count = 1
for pred in prediction:
    # print(count)
    # print("predictions:", pred)
    pred_res = np.argmax(pred)
    # print("highes prediction index:", pred_res)
    # print("classification:", CLASSIFICATION[pred_res])
    # print()
    
    if count <=100: 
        healthy.append([count,pred,pred_res])
   
    if count >100 and count <=200:
        rot.append([count,pred,pred_res])

    if count >200:
        rust.append([count,pred,pred_res])

    count=count+1

    # if count <=100:
    #     if(pred_res==0):
    #         yesno.append("yes") 
    #     else:
    #         yesno.append("no")  
    #     #healthy.append([count,pred,pred_res])
   
    # if count >100 and count <=200:
    #     if(pred_res==1):
    #         yesno.append("yes") 
    #     else:
    #         yesno.append("no")  
    # if count >200:
    #     if(pred_res==2):
    #         yesno.append("yes") 
    #     else:
    #         yesno.append("no")  
    # count=count+1

# print(len(healthy))
# print(len(rot))
# print(len(rust))
# print(len(yesno))

for item in healthy:
    print(item[2])

for item in rot:
    print(item[2])

for item in rust:
    print(item[2])

# for item in yesno:
#     print(item)
    
#     #cv.imshow('test',x_test[item[0]])
#     img = 255*x_test[item[0]]
#     img = cv.resize(img, (256,256), interpolation = cv.INTER_AREA)
#     cv.imwrite("./misclassified/healthy/h("+str(item[0])+").jpg", img)
# #     cv.waitKey(1000)
# # cv.destroyAllWindows()

# for item in misclassified_rust:
#     print(item)
#     #cv.imshow('test',x_test[item[0]])
#     img = 255*x_test[item[0]]
#     img = cv.resize(img, (256,256), interpolation = cv.INTER_AREA)
#     cv.imwrite("./misclassified/rust/ru("+str(item[0])+").jpg", img)
# #     cv.waitKey(1000)
# # cv.destroyAllWindows()

# for item in misclassified_rot:
#     print(item)
#     #cv.imshow('test',x_test[item[0]])
#     img = 255*x_test[item[0]]
#     img = cv.resize(img, (256,256), interpolation = cv.INTER_AREA)
#     cv.imwrite("./misclassified/rot/ro("+str(item[0])+").jpg", img)
#     cv.waitKey(1000)
# cv.destroyAllWindows()
#   i = 0;
#   for img in x_test[:10]:
#     print(y_test[i], end = '\r')
#     cv2.imshow('test',img)
#     cv2.waitKey(1000)
#     i = i + 1
#   cv2.destroyAllWindows()
    

# print("predictions: ", prediction)
# prediction_result=np.argmax(prediction)
# print("Highest prediction index:", prediction_result)
# print("Output:", CLASSIFICATION[prediction_result])
# img = combine_images(recon, height=1)
# image = img * 255
# Image.fromarray(image.astype(np.uint8)).save("reconstructions/reconsctructed-%d.png" % prediction_result )


#prediction = model.predict()


