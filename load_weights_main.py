from PIL import Image
import cv2 as cv
import numpy as np
from load_weights_capsnet_model import CapsNet, combine_images

CAPSNET_WEIGHTS = "./result/weights-79.h5"
COLOR_MODE = 3
IMG_SIZE = 64
CLASSIFICATION = ["Healthy", "Rot", "Rust"]

#(x_train, y_train), (x_test, y_test) = load_aloe()

# print(x_train.shape[1:])
# print(len(np.unique(np.argmax(y_train, 1))))

model, eval_model, manipulate_model = CapsNet(input_shape=(IMG_SIZE, IMG_SIZE, COLOR_MODE),n_class=len(CLASSIFICATION),routings=3, batch_size=1)

#loaded_model = keras.models.load_model("my_capsnet", custom_objects={"Capsnet":CapsNet})
eval_model.load_weights(CAPSNET_WEIGHTS)


def preprocess(filepath):
    
    #img_ar = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    img_ar = cv.imread(filepath)
    img_ar = cv.cvtColor(img_ar, cv.COLOR_BGR2HSV)
    new_ar = cv.resize(img_ar, (IMG_SIZE, IMG_SIZE))
    return new_ar.reshape(1, IMG_SIZE, IMG_SIZE, COLOR_MODE)

#input= preprocess("sample inputs/rust/rust.jpg")
#input= preprocess("sample inputs/rot/rot.jpg")
input= preprocess("sample inputs/healthy/h (4).jpg")

prediction, recon = eval_model.predict([input])

print("predictions: ", prediction)
prediction_result=np.argmax(prediction)
print("Highest prediction index:", prediction_result)
print("Output:", CLASSIFICATION[prediction_result])
img = combine_images(recon, height=1)
image = img * 255
Image.fromarray(image.astype(np.uint8)).save("reconstructions/reconsctructed-%d.png" % prediction_result )


#prediction = model.predict()


