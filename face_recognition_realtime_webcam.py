"""
    File    : face_recognition_realtime_webcam.py
    Author  : kaiboon
    Date    : 28/5/2020
    Brief   : Return the known face from realtime webcam. 
"""

import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known"
#UNKNOWN_FACES_DIR = "unknown_faces" #comment out unknown faces cuz source is video not photo
TOLERANCE = 0.425 #The higher the tolerance, the higher to get match, but also higher to get false positive; the lower, the higher to get false negative(mis match)
FRAME_THICKNESS = 3 
FONT_THICKNESS = 2 
MODEL = "cnn"

video = cv2.VideoCapture(-1) #could put in a filename #-1 is to get input from webcam
#video = cv2.VideoCapture("OC Video Final.mp4")

print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}") #not need for filename video 
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print("processing unknown faces")

while True:
    ret, image = video.read()
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces,face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            #For the frame of faces
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0,255,0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            #For the text frame
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), FONT_THICKNESS)
    
    cv2.imshow(filename,image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    #cv2.waitKey(1000)
    #cv2.destroyWindow(filename) #no work in unbuntu so waitkey put 10000
