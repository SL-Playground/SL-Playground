from flask import Flask, Response, render_template
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Load the model from the .modelsave file
model = tf.saved_model.load('Asldevhouse\ISL\isl_alphabet\model.savedmodel')

# Load the labels from the .txt file
with open('Asldevhouse\ISL\isl_alphabet\labels.txt', 'r') as f:
    label_mapping = [line.strip() for line in f.readlines()]

# Function to preprocess input frames
def preprocess_frame(frame):
    # Resize the frame to match the input size expected by the model
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize pixel values to the range [0, 1]
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Function to predict numbers for a frame
def predict_number(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    # Add batch dimension and convert to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(preprocessed_frame[np.newaxis, ...], dtype=tf.float32)
    # Perform inference
    predictions = model(input_tensor)
    # Convert predictions to numbers
    number_index = np.argmax(predictions)
    number = label_mapping[number_index]  # Get the number corresponding to the index
    return number

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            predicted_number = predict_number(frame)
            cv2.putText(frame, f"Predicted Number: {predicted_number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
