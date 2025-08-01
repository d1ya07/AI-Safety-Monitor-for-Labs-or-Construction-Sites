import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="D:/ai safety monitor/model/helmet_detector.tflite")
interpreter.allocate_tensors()

# Get input and output tensors info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Image dimensions
IMG_SIZE = (128, 128)

# Start video capture
cap = cv2.VideoCapture(0)  # 0 is for webcam

print("🚨 Press 'q' to quit the webcam window.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32") / 255.0
    input_data = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    label = "Helmet" if prediction > 0.5 else "No Helmet"
    color = (0, 255, 0) if label == "Helmet" else (0, 0, 255)

    # Display result
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 4)
    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
