
import cv2
import os
import json
import numpy as np

DATASET_PATH = "dataset"
MODEL_PATH = "model/lbph_model.yml"
LABELS_PATH = "model/labels.json"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}

current_label = 0

for employee_folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, employee_folder)

    if not os.path.isdir(folder_path):
        continue

    label_names[current_label] = employee_folder

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, (200, 200))

        faces.append(image)
        labels.append(current_label)

    current_label += 1

if len(faces) == 0:
    print("No face images found. Register an employee first.")
    raise SystemExit

recognizer.train(faces, np.array(labels))

os.makedirs("model", exist_ok=True)

recognizer.save(MODEL_PATH)

with open(LABELS_PATH, "w") as file:
    json.dump(label_names, file)

print("Face Recognition Model Trained Successfully!")
print("Total face images:", len(faces))
print("Total employees:", len(label_names))
print("Model saved:", MODEL_PATH)
