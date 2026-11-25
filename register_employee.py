
import cv2
import os

# Face detector load
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Employee details
employee_id = input("Enter Employee ID: ").strip()
employee_name = input("Enter Employee Name: ").strip()

if not employee_id or not employee_name:
    print("Employee ID and Name are required.")
    raise SystemExit

# Create employee dataset folder
folder_name = f"{employee_id}_{employee_name.replace(' ', '_')}"
save_path = os.path.join("dataset", folder_name)
os.makedirs(save_path, exist_ok=True)

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened.")
    raise SystemExit

count = 0
max_images = 30

print("\nLook at the camera.")
print("Press Q to stop capturing.")

while count < max_images:
    ret, frame = camera.read()

    if not ret:
        print("Could not read camera frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # Save face image
        image_path = os.path.join(
            save_path, f"{count + 1}.jpg"
        )
        cv2.imwrite(image_path, face)

        count += 1

        cv2.rectangle(
            frame, (x, y), (x+w, y+h), (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"Images: {count}/{max_images}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Capture only one face per frame
        break

    cv2.imshow("Employee Face Registration", frame)

    if cv2.waitKey(300) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print(f"\nRegistration completed: {count} images saved.")
print("Saved in:", save_path)
