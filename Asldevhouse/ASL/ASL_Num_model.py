import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow SavedModel
model = tf.saved_model.load("ASL\ASL_num_trained\model.savedmodel")

# Dictionary to map the predicted label index to the corresponding number
label_mapping = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5"
}

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

# Function to capture frames from the camera, predict numbers, and display them
def predict_numbers_from_camera():
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Unable to capture frame")
            break

        # Predict the number for the frame
        predicted_number = predict_number(frame)

        # Display the predicted number on the frame
        cv2.putText(frame, f"Predicted Number: {predicted_number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame with the predicted number
        cv2.imshow('Camera Feed', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to capture frames from the camera and predict numbers
predict_numbers_from_camera()
