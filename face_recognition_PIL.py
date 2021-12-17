"""
    File    : face_recognition_PIL.py
    Author  : kaiboon
    Date    : 28/5/2020
    Brief   : This version is running on local terminal for facial recognition. 
              A square box and name will appear on the known face.
              Make sure your GPU is enough memory, or else it will crash!
              This file using PIL to draw the square and name.
"""


import face_recognition
from PIL import Image, ImageDraw
import os
import cv2

TOLERANCE = 0.42

KNOWN_DIR = "known"
UNKNOWN_DIR = "groups"
MODEL = "cnn"

known_face_encodings = []
known_face_names = []

def read_img(path):
    img = cv2.imread(path)
    (h,w) = img.shape[:2]
    width = 500
    ratio = width /float(w)
    height = int(h*ratio)
    return cv2.resize(img,(width,height))

print("Start process known images")

for name in os.listdir(KNOWN_DIR):
    for filename in os.listdir(f"{KNOWN_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_DIR}/{name}/{filename}")
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

print("Start process unknown images")

#Load unknown images
for filename in os.listdir(UNKNOWN_DIR):
    image = face_recognition.load_image_file(f"{UNKNOWN_DIR}/{filename}")
    #img = read_img(f"{UNKNOWN_DIR}/{filename}")
    # Find faces in test image
    test_face_location = face_recognition.face_locations(image)
    test_face_encoding = face_recognition.face_encodings(image, test_face_location)
    #Convert to PIL format
    pil_image = Image.fromarray(image)
    #Create a ImageDraw instance
    draw = ImageDraw.Draw(pil_image)

    # Loop through faces in test image 
    for(top, right, bottom, left), face_encoding in zip(test_face_location, test_face_encoding):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
        name = "Unknown Person"

        #If match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
    
        # Draw Main Box
        draw.rectangle(
            ((left, top), (right, bottom)), 
            outline=(0,0,0)
        ) #Outline is the color in RGB

        #Draw label
        text_width, text_height = draw.textsize(name)
        draw.rectangle(
            ((left, bottom-text_height-10),(right,bottom)),
            fill = (0,0,0),
            outline = (0,0,0)
        )
        draw.text(
            (left+6, bottom-text_height-5), 
            name, 
            fill = (255,255,255,255)
        )
    # Delete draw from memory, this one is recommended by the official documentation
    del draw

    #Display image
    pil_image.show()




