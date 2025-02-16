import cv2
import numpy as np

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("classifier.xml")
q
# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Labels (Modify if needed)
LABELS = {0: "Mask", 1: "No Mask"}

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]  # Extract face region
        face_roi = cv2.resize(face_roi, (100, 100))  # Resize to match training size
        
        label, confidence = recognizer.predict(face_roi)  # Predict

        # Check confidence threshold
        if confidence < 80:
            text = LABELS[label]
            color = (0, 255, 0) if label == 0 else (0, 0, 255)  # Green for Mask, Red for No Mask
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{text} ({100 - confidence:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
