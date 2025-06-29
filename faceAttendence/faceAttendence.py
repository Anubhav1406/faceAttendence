import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime

# --- Step 1: Get today's date ---
date_input = input("Enter today's date: ")
print("Using date:", date_input)

# Continue...
print("Loading known faces...")

# --- Step 2: Load known faces ---
known_encodings = []
known_names = []

for filename in os.listdir("known_faces"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = face_recognition.load_image_file(f"known_faces/{filename}")
        encoding = face_recognition.face_encodings(img)
        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])

# --- Step 3: Initialize attendance ---
attendance = {name: 'A' for name in known_names}  # A = Absent by default

# --- Step 4: Start webcam and capture faces ---
video_capture = cv2.VideoCapture(1)

print("Press 'q' to stop attendance capture.")
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to open webcam. Try changing VideoCapture index to 0 or 1.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        best_match_index = None
        if face_distances.size > 0:
            best_match_index = face_distances.argmin()

        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]
            attendance[name] = 'P'

    # Optional: show the camera view
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if face_distances.size > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# --- Step 5: Update/Save Excel Sheet ---
filename = "attendance.xlsx"
try:
    df = pd.read_excel(filename, index_col=0)
except FileNotFoundError:
    df = pd.DataFrame(index=known_names)

df[date_input] = df.index.map(attendance.get)
df.to_excel(filename)
print(f"Attendance saved in {filename}")