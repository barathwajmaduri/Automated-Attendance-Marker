import os
import cv2
import dlib
import numpy as np
import pickle

# Load Dlib models
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()

# Paths
dataset_path = "faces_dataset"

# Step 2: Extract Face Embeddings and Train Model
print("🔍 Extracting face embeddings...")

face_encodings = {}  # Store embeddings

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue  # Skip non-directory files

    encodings_list = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            print(f"⚠️ No face detected in {img_path}")
            continue  # Skip if no face found

        shape = predictor(gray, faces[0])
        face_encoding = np.array(face_rec_model.compute_face_descriptor(image, shape))
        encodings_list.append(face_encoding)

    if encodings_list:
        face_encodings[person_name] = np.mean(encodings_list, axis=0)

# Save encodings
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(face_encodings, f)

print("✅ Face embeddings saved successfully! Ready for real-time recognition.")
