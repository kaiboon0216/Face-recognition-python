"""
    File    : face_recognition_local_terminal.py
    Author  : kaiboon
    Date    : 28/5/2020
    Brief   : This version is running on local terminal for facial recognition. 
              A square box and name will appear on the known face.
              Make sure your GPU is enough memory, or else it will crash!
              This file using cv2 to draw the square and name.
"""

import face_recognition
import cv2
import os

TOLERANCE = 0.425

#function to resize the image
def read_img(path):
    img = cv2.imread(path)
    (h,w) = img.shape[:2]
    width = 625
    ratio = width /float(w)
    height = int(h*ratio)
    return cv2.resize(img,(width,height))

known_encodings = []
known_names = []
known_dir = 'known'
unknown_dir = 'groups'

for name in os.listdir(known_dir):
    for filename in os.listdir(f"{known_dir}/{name}"):
        #img = read_img(f"{known_dir}/{name}/{filename}")
        img = face_recognition.load_image_file(f"{known_dir}/{name}/{filename}")
        #cv2.imshow('',img)
        #cv2.waitKey(1)
        img_enc = face_recognition.face_encodings(img)[0]
        known_encodings.append(img_enc)
        known_names.append(name)


for filename in os.listdir(unknown_dir):
    print("Processing",filename)
    img = read_img(f"{unknown_dir}/{filename}")
    img_location = face_recognition.face_locations(img, model="cnn")
    img_enc = face_recognition.face_encodings(img,img_location)

    #results = face_recognition.compare_faces(known_encodings,img_enc)
    #print(face_recognition.face_distance(known_encodings,img_enc))
    for (top, right, bottom, left), face_encoding in zip(img_location, img_enc):
        results = face_recognition.compare_faces(known_encodings, face_encoding, TOLERANCE)
        
        if True in results:
            name = known_names[results.index(True)]
            #(top,right,bottom,left) = face_recognition.face_locations(img)[0]
            cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
            cv2.putText(img,name,(left+2, bottom+20),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,0,0),2)

    cv2.imshow('',img) 
    cv2.waitKey(1000)   