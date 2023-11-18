import cv2
import os

# Create a folder if it doesn't exist

# os.makedirs(output_folder, exist_ok=True)
def registeration():
    # Load the pre-trained Haar Cascade Classifier for face detection
    output_folder = 'E:\RoboProj\ImageAtt'
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = frame.copy()
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, 'Face Detected Press Y to take picture', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),
                        2, cv2.LINE_AA)
            print('Face Detected Press Y to take picture')
        # Display the frame
        if len(faces) == 0:
            cv2.putText(img, 'No Face Detected Press q to quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                        cv2.LINE_AA)
            print('No Face Detected Press q to quit')
        cv2.imshow('Face Detection', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('Y') or key == ord('y'):
            # Prompt user for the name of the person
            person_name = input("Enter the name of the person: ")

            # Save the image with the entered name
            img_name = os.path.join(output_folder, f'{person_name}.jpg')
            cv2.imwrite(img_name, frame)
            print(f"Image saved: {img_name}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()