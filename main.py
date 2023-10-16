import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("ServiceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "",
    'storageBucket': ""
})

bucket = storage.bucket()
print(bucket)

#array = np.frombuffer(blob.download_as_string(), np.uint8)
#print(array)


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("ATTENDANCE SYSTEM.png")

# Importing the mode images
folderModePath = "Modes"
modePath = os.listdir(folderModePath)
imgModeList = []
for path in modePath:  # THis will tell us if we have imported all the images or not
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
#print(len(imgModeList))

#Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIDs = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIDs
print(studentIds)
print("Encode File Loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read() # Read a frame from the camera

    if not ret: # Check if the frame was read successfully
        print("Error: Could not read frame.")
        break

    imgS = cv2.resize(frame,(0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    FaceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, FaceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    imgBackground[113:113 + 535, 865:865 + 344] = imgModeList[modeType]


    if FaceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, FaceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            #print("matches : ", matches)
            #print("faceDis : ", faceDis)

            matchIndex = np.argmin(faceDis)
            #print("match Index : ", matchIndex)

            if matches[matchIndex]:
                #print("Known Face Detected")
                #print(studentIds[matchIndex])
                for faceLoc in FaceCurFrame:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                id = studentIds[matchIndex]
                if counter == 0:
                    cvzone.putTextRect(imgBackground,"Loading", (275,400))
                    cv2.imshow('Attendance', imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1


        if counter != 0:

            if counter == 1:
                # Get the Data
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)
                # Get the Image from storage
                blob = bucket.get_blob(f'Images/{id}.jpeg')
                print(blob)
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                #print(array)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                #print(imgStudent)
                #print(len(imgStudent))

                # Update attendance
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                  "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now()-datetimeObject).total_seconds()
                print(secondsElapsed)
                if secondsElapsed > 30:
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3
                    counter = 0
                    imgBackground[113:113 + 535, 865:865 + 344] = imgModeList[modeType]

            if modeType != 3:
                if counter < 20:
                    modeType = 2

                imgBackground[113:113 + 535, 865:865 + 344] = imgModeList[modeType]

                if counter <= 10:
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (859, 124),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w,h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1,1)
                    offset = (414-w)//2
                    cv2.putText(imgBackground, str(studentInfo['name']), (808+offset, 445),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

            #imgBackground[175:175+216, 909:909+216] = imgStudent
            #imgBackground[175:1280, 909:1263] = imgStudent

                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    imgBackground[44:44 + 535, 808:808 + 344] = imgModeList[modeType]


        #imgBackground[] = "Active.jpeg"
        #imgBackground[] = "AlreadyMarked.jpeg"

    else:
        modeType = 0
        counter = 0

    #cv2.imshow('Camera', frame)
    cv2.imshow('Attendance', imgBackground) #  Display the frame in a window
    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit the loop if 'q' key is pressed
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
