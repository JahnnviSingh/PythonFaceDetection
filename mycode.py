import os
import pickle
import numpy as np
import cv2

# Create a VideoCapture object to access the camera (0 for the default camera)
import face_recognition

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("ATTENDANCE SYSTEM.png")


# Importing the mode images
folderModePath = "Modes"
modePath = os.listdir(folderModePath)
imgModeList = []

# THis will tell us if we have imported all the imges or not
for path in modePath:
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

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break


    imgS = cv2.resize(frame,(0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    FaceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, FaceCurFrame)


    for encodeFace, faceLoc in zip(encodeCurFrame, FaceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("matches : ", matches)
        print("faceDis : ", faceDis)

        matchIndex = np.argmin(faceDis)
        print("match Index : ", matchIndex)
        
        
        
        


    imgBackground[162:162+480, 55:55+640] = frame
    imgBackground[113:113+535, 865:865 + 350] = imgModeList[0]

    #imgBackground[] = "Active.jpeg"
    #imgBackground[] = "AlreadyMarked.jpeg"

    # Display the frame in a window
    #cv2.imshow('Camera', frame)
    cv2.imshow('Attendance', imgBackground)


    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()



