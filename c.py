import os
import cv2
import dlib

# Load Dlib models
detector = dlib.get_frontal_face_detector()

# Paths
dataset_path = "faces_dataset"
os.makedirs(dataset_path, exist_ok=True)

# Step 1: Capture Face Images
person_name = input("Enter person's name: ").strip()
if not person_name:
    print("❌ Person name cannot be empty.")
    exit()

person_folder = os.path.join(dataset_path, person_name)
os.makedirs(person_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

count = 0
print("📸 Capturing images. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from webcam.")
        break

    cv2.imshow("Capture Face", frame)

    # Save every 5th frame
    if count % 5 == 0:
        img_path = os.path.join(person_folder, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"✅ Saved {img_path}")

    count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"🎉 {count // 5} images saved for {person_name}!")
