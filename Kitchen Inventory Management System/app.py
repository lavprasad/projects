import datetime
import os
import sqlite3
import time

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from keras.utils import img_to_array

app = Flask(__name__)

# Load the model and categories
model = tf.keras.models.load_model("C:/Users/Asus/Downloads/Object detection/my_model_0.25.h5")
source_folder = "C:/Users/Asus/Downloads/Object detection/Fruits-36dataset/test"
categories = os.listdir(source_folder)

# Initialize camera and variables
cap = None
start_time = 0
end_time = 0
predictions = {}
detected_objects = set()


# Function to prepare image
def prepare_image(image):
    image = cv2.resize(image, (64, 64))  # Resize image to match model input size
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result, axis=0)
    img_result = img_result / 255.
    return img_result


# Function to insert prediction into database
def insert_prediction(prediction):
    conn = sqlite3.connect("predictions2.db")
    c = conn.cursor()
    c.execute("INSERT INTO predictions (fruit, count, expiry_date) VALUES (?, ?, ?)", prediction)
    conn.commit()
    conn.close()


# Function to fetch predictions from database
def fetch_predictions():
    conn = sqlite3.connect("predictions2.db")
    c = conn.cursor()
    c.execute("SELECT fruit, SUM(count), expiry_date FROM predictions GROUP BY fruit")
    predictions = c.fetchall()
    conn.close()
    return predictions


# Function to calculate expiry date based on freshness of the item
def calculate_expiry_date(days):
    current_date = datetime.date.today()
    expiry_date = current_date + datetime.timedelta(days=days)
    return expiry_date


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predictions')
def get_predictions():
    predictions = fetch_predictions()
    return render_template('predictions.html', predictions=predictions)


@app.route('/modify', methods=['POST'])
def modify_prediction():
    fruit = request.form['fruit']
    action = request.form['action']
    if action == 'add':
        # Fetch freshness days for the fruit
        freshness_days = get_freshness_days(fruit)
        expiry_date = calculate_expiry_date(freshness_days)
        insert_prediction((fruit, 1, expiry_date))
    elif action == 'remove':
        conn = sqlite3.connect("predictions2.db")
        c = conn.cursor()
        c.execute("DELETE FROM predictions WHERE fruit=?", (fruit,))
        conn.commit()
        conn.close()
    return redirect(url_for('get_predictions'))


def gen_frames():
    global cap, start_time, end_time, predictions, detected_objects
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    end_time = start_time + 10  # Detect objects for 10 seconds

    while time.time() < end_time:
        success, frame = cap.read()
        if not success:
            break

        image_for_model = prepare_image(frame)
        result_array = model.predict(image_for_model, verbose=1)
        prediction_index = np.argmax(result_array, axis=1)
        text = categories[prediction_index[0]]

        # Fetch freshness days for the detected fruit
        freshness_days = get_freshness_days(text)
        expiry_date = calculate_expiry_date(freshness_days)

        insert_prediction((text, 1, expiry_date))  # Insert prediction into database

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


def get_freshness_days(fruit):
    # Dictionary to store freshness days for each fruit
    freshness_days = {
        'apple': 10,
        'banana': 5,
        'beetroot': 7,
        'bell pepper': 8,
        'cabbage': 10,
        'capsicum': 8,
        'carrot': 14,
        'cauliflower': 7,
        'chilli pepper': 5,
        'corn': 4,
        'cucumber': 7,
        'eggplant': 7,
        'garlic': 60,
        'ginger': 30,
        'grapes': 5,
        'jalepeno': 7,
        'kiwi': 14,
        'lemon': 14,
        'lettuce': 7,
        'mango': 7,
        'onion': 60,
        'orange': 14,
        'paprika': 10,
        'pear': 14,
        'peas': 4,
        'pineapple': 7,
        'pomegranate': 14,
        'potato': 30,
        'raddish': 10,
        'soy beans': 5,
        'spinach': 7,
        'sweetcorn': 4,
        'sweetpotato': 30,
        'tomato': 7,
        'turnip': 10,
        'watermelon': 7
    }
    return freshness_days.get(fruit, 7)  # Default to 7 days if freshness days not found


if __name__ == "__main__":
    # Create predictions2.db if not exists
    conn = sqlite3.connect("predictions2.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (fruit TEXT, count INTEGER, expiry_date TEXT)''')
    conn.commit()
    conn.close()

    app.run(debug=True)


@app.route('/run-chapter6-app', methods=['POST'])
def run_chapter6_app():
    import subprocess
    subprocess.run(['python', 'path_to_chapter6_app.py'])  # Replace 'path_to_chapter6_app.py' with the actual path
    return jsonify({'success': True, 'message': 'Chapter 6 app is running'})
