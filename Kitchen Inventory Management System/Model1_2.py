import os

import cv2
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array, load_img

# load the model
model = tf.keras.models.load_model("C:/Users/Asus/Downloads/Object detection/Fruits360.h5")
print(model.summary())

# load the categories :

source_folder = "C:/Users/Asus/Downloads/Object detection/fruits-360_dataset/fruits-360/Test"
categories = os.listdir(source_folder)
categories.sort()
print(categories)
numofClasses = len(categories)
print(numofClasses)


# load and prepare image

def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(100, 100))
    imgResult = img_to_array(image)
    # print(imgResult.shape)
    imgResult = np.expand_dims(imgResult, axis=0)
    # print(imgResult.shape)
    imgResult = imgResult / 255.
    return imgResult


testImagePath = "C:/Users/Asus/Downloads/Object detection/Onion.jpg"
imageForModel = prepareImage(testImagePath)

resultArray = model.predict(imageForModel, verbose=1)
answers = np.argmax(resultArray, axis=1)
print(answers[0])

text = categories[answers[0]]
print("Predicted image : " + text)

# show the image with the text
#
img = cv2.imread(testImagePath)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, text, (0, 50), font, 1, (209, 19, 77), 2)
cv2.imshow('img', img)
cv2.imwrite('TensorFlowProjects/Fruits classification/predicted.png', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
