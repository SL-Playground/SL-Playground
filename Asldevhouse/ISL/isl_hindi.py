from flask import Flask, Response, render_template
import cv2
import numpy as np
import tensorflow as tf
from googletrans import Translator
import threading
import pyttsx3
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load the TensorFlow SavedModel
model = tf.saved_model.load('Asldevhouse/ISL/Isl_words/model.savedmodel')

# Load the labels from the .txt file
with open('Asldevhouse/ISL/Isl_words/labels.txt', 'r') as f:
    label_mapping = [line.strip() for line in f.readlines()]

# Function to preprocess input frames
def preprocess_frame(frame):
    # Resize the frame to match the input size expected by the model
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize pixel values to the range [0, 1]
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Function to predict words for a frame
def predict_word(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    # Add batch dimension and convert to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(preprocessed_frame[np.newaxis, ...], dtype=tf.float32)
    # Perform inference
    predictions = model(input_tensor)
    # Convert predictions to words
    word_index = np.argmax(predictions)
    word = label_mapping[word_index]  # Get the word corresponding to the index
    return word

# Initialize the translator
translator = Translator()

# Function to translate text with retry
def translate_with_retry(text, src='en', dest='hi', max_retries=3, delay=1):
    for _ in range(max_retries):
        try:
            translated = translator.translate(text, src=src, dest=dest)
            return translated.text
        except AttributeError as e:
            print("Error fetching TKK token:", e)
            print("Retrying after a delay...")
            time.sleep(delay)
    print("Failed to translate after multiple retries.")
    return None

# Function to handle translation and text-to-speech conversion
def translate_and_speak():
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame")
            break

        predicted_word = predict_word(frame)
        translated_word = translate_with_retry(predicted_word)

        if translated_word:
            # Text-to-speech
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(translated_word)
            engine.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    # Start the translation and text-to-speech thread
    threading.Thread(target=translate_and_speak, daemon=True).start()

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame")
            continue

        predicted_word = predict_word(frame)
        cv2.putText(frame, f"Predicted Word: {predicted_word}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == '__main__':
    print("Camera is opened:", camera.isOpened())  # Check if camera is opened successfully
    app.run(debug=True)
