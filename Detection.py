import cv2
from datetime import datetime
import numpy as np
import face_recognition
import os
import pickle
import utils_f as u
#import csv
def detection():
    path = 'E:/RoboProj/ImageAtt'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    # Check if encoding file exists
    encoding_file = 'encodings.pkl'
    if os.path.exists(encoding_file):
        print("Loading Existing model")
        with open(encoding_file, 'rb') as f:
            encodeListKnown = pickle.load(f)
    else:
        print("Training the model")
        encodeListKnown = u.findEncodings(images)
        with open(encoding_file, 'wb') as f:
            pickle.dump(encodeListKnown, f)

    print('Model Loaded')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                #u.markAttendance(name)
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()