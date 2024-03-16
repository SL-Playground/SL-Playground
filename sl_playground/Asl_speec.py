from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import queue
from googletrans import Translator
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load the TensorFlow SavedModel
model = tf.saved_model.load("Asldevhouse\ASL\ASL_num_trained\model.savedmodel")

# Dictionary to map index to label
label_mapping = {
    0: "Can",
    1: "hello",
    2: "Help",
    3: "Me",
    4: "Nobody",
    5: "Sad",
    6: "Understand",
    7: "Why"
}

# Function to preprocess input frames
def preprocess_frame(frame):
    # Resize the frame to match the input size expected by the model
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize pixel values to the range [0, 1]
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Function to convert text to speech
def text_to_speech(text):
    # Initialize the Text-to-Speech engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

    # Convert text to speech
    engine.say(text)

    # Wait for speech to finish
    engine.runAndWait()

# Initialize the translator
translator = Translator()

# Function to translate text with retry
def translate_with_retry(text, src='en', dest='es', max_retries=3, delay=1):
    for _ in range(max_retries):
        try:
            return translator.translate(text, src=src, dest=dest).text
        except AttributeError as e:
            print("Error fetching TKK token:", e)
            print("Retrying after a delay...")
            time.sleep(delay)
    print("Failed to translate after multiple retries.")
    return None

# Function to predict labels for a frame
def predict_label(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    # Add batch dimension and convert to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(preprocessed_frame[np.newaxis, ...], dtype=tf.float32)
    # Perform inference
    predictions = model(input_tensor)
    # Convert predictions to labels
    label_index = np.argmax(predictions)
    label = label_mapping[label_index]  # Get the label corresponding to the index
    # Get the confidence score for the predicted label
    confidence_score = np.max(predictions)
    return label, confidence_score

def generate_frames():
    prev_label = None
    sentence = ""
    while True:
        ## read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Predict label and confidence score for the frame
            label, confidence_score = predict_label(frame)

            # Only annotate the frame if the confidence score is greater than 0.8
            if confidence_score > 0.95:
                # If the predicted label has changed, enqueue it
                if label != prev_label:
                    # Translate the label
                    translated_word = translate_with_retry(label)
                    if translated_word:
                        # Append the translated word to the sentence
                        sentence += translated_word + " "
                        prev_label = label
                        # Speak the translated word
                        threading.Thread(target=text_to_speech, args=(translated_word,)).start()

                # Annotate the frame with the predicted label
                cv2.putText(frame, f"Predicted Label: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Sentence: {sentence}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('toSpeech.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)