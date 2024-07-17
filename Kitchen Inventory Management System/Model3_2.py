import os

import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img

# Load the model
model = tf.keras.models.load_model("C:/Users/Asus/Downloads/Object detection/my_model_0.25.h5")
print(model.summary())

# Load the categories
source_folder = "C:/Users/Asus/Downloads/Object detection/Fruits-36dataset/test"
categories = os.listdir(source_folder)
categories.sort()
print(categories)
numofClasses = len(categories)
print(numofClasses)


# Function to prepare image
def prepare_image(image_path):
    image = load_img(image_path, target_size=(64, 64))  # Resize image to match model input size
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result, axis=0)
    img_result = img_result / 255.
    return img_result


test_image_path = "C:/Users/Asus/Downloads/pexels-cottonbro-studio-6590916.jpg"
image_for_model = prepare_image(test_image_path)

result_array = model.predict(image_for_model, verbose=1)
answers = np.argmax(result_array, axis=1)
print(answers[0])

text = categories[answers[0]]
print("Predicted image: " + text)

# Load the image
img = cv2.imread(test_image_path)
font = cv2.FONT_HERSHEY_COMPLEX

# Put text on the image
cv2.putText(img, text, (0, 50), font, 1, (209, 19, 77), 2)

# Show and save the image
cv2.imshow('img', img)
cv2.imwrite('predicted.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
