import os
import sqlite3
import time

import cv2
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array

# Load the model
model = tf.keras.models.load_model("C:/Users/Asus/Downloads/Object detection/Fruits360.h5")
print(model.summary())

# Load the categories
source_folder = "C:/Users/Asus/Downloads/Object detection/fruits-360_dataset/fruits-360/Test"
categories = os.listdir(source_folder)
categories.sort()
print(categories)
numofClasses = len(categories)
print(numofClasses)


# Function to prepare image
def prepare_image(image):
    image = cv2.resize(image, (100, 100))  # Resize image to match model input size
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result, axis=0)
    img_result = img_result / 255.
    return img_result


# Function to initialize database
def initialize_database(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY, fruit TEXT, count INTEGER)''')
    conn.commit()
    conn.close()


# Function to insert prediction into database
def insert_prediction(db_file, prediction):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (fruit, count) VALUES (?, ?)", prediction)
    conn.commit()
    conn.close()


# Function to fetch predictions from database
def fetch_predictions(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT fruit, SUM(count) FROM predictions GROUP BY fruit")
    predictions = c.fetchall()
    conn.close()
    return predictions


# Open webcam
cap = cv2.VideoCapture(0)

start_time = time.time()
end_time = start_time + 10  # Detect objects for 5 seconds

predictions = {}  # Dictionary to store predictions and their counts
detected_objects = set()  # Set to store already detected objects

# Initialize database
db_file = "predictions.db"
initialize_database(db_file)

while time.time() < end_time:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        break

    # Prepare frame for model prediction
    image_for_model = prepare_image(frame)

    # Predict
    result_array = model.predict(image_for_model, verbose=1)
    prediction_index = np.argmax(result_array, axis=1)
    text = categories[prediction_index[0]]

    # Store predictions and their counts
    if text not in detected_objects:
        detected_objects.add(text)
        predictions[text] = predictions.get(text, 0) + 1
        # Insert prediction into database
        insert_prediction(db_file, (text, predictions[text]))

    # Display prediction
    cv2.putText(frame, "Predicted image: " + text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Fruit Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print predicted object names and their counts
print("Predicted objects and their counts:")
for prediction, count in predictions.items():
    print(prediction, ":", count)

# Fetch and print predictions from database
print("\nPredictions fetched from database:")
db_predictions = fetch_predictions(db_file)
for prediction in db_predictions:
    print(prediction[0], ":", prediction[1])
