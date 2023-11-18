import cv2
import csv
from datetime import datetime
import numpy as np
import face_recognition

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'a', newline='') as f:
        csv_writer = csv.writer(f)

        # Read existing data to check if the name is already present
        nameList = []
        with open('Attendance.csv', 'r') as file:
            reader = csv.reader(file)
            for line in reader:
                # Check if the line is non-empty before accessing its elements
                if line:
                    entry = line[0].split(',')
                    if entry:
                        nameList.append(entry[0])

        # Check if the name is not already in the list, then add it
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')

            # Write the new attendance entry
            csv_writer.writerow([name, dtString])