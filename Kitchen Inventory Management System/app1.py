import os
import sqlite3
import time

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response
from keras.utils import img_to_array

app = Flask(__name__)

# Load the model and categories
model = tf.keras.models.load_model("C:/Users/Asus/Downloads/Object detection/Fruits4_updated_WG_2.h5")
source_folder = "C:/Users/Asus/Downloads/Object detection/fruits-4_dataset/fruits-4/Test"
categories = os.listdir(source_folder)

# Initialize camera and variables
cap = None
start_time = 0
end_time = 0
detected_objects = set()


# Function to prepare image
def prepare_image(image):
    image = cv2.resize(image, (100, 100))  # Resize image to match model input size
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result, axis=0)
    img_result = img_result / 255.
    return img_result


# Function to insert prediction into database
def insert_prediction(prediction):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("INSERT INTO predictions (fruit, count) VALUES (?, ?)", prediction)
    conn.commit()
    conn.close()


# Function to fetch predictions from database
def fetch_predictions():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT fruit, SUM(count) FROM predictions GROUP BY fruit")
    predictions = c.fetchall()
    conn.close()
    return predictions


def gen_frames():
    global cap, start_time, end_time, detected_objects
    cap = cv2.VideoCapture(1)
    start_time = time.time()
    end_time = start_time + 10  # Detect objects for 10 seconds

    while time.time() < end_time:
        success, frame = cap.read()
        if not success:
            break

        image_for_model = prepare_image(frame)
        result_array = model.predict(image_for_model, verbose=1)
        prediction_indices = np.argsort(result_array[0])[-3:][::-1]  # Get indices of top 3 predictions

        # Check if prediction indices are within range
        valid_indices = [i for i in prediction_indices if i < len(categories)]
        texts = [categories[i] for i in valid_indices]

        new_objects = set(texts) - detected_objects
        detected_objects.update(new_objects)

        for text in new_objects:
            insert_prediction((text, 1))  # Insert prediction into database

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predictions')
def get_predictions():
    predictions = fetch_predictions()
    return render_template('predictions2.html', predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)
