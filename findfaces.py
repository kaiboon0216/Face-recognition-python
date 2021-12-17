"""
    File    : findfaces.py
    Author  : kaiboon
    Date    : 28/5/2020
    Brief   : Return the coordinates and number of faces in the images. 
              It cannot recognize people who wear mask.
"""

import face_recognition

image = face_recognition.load_image_file("./groups/coursemate1.jpg")
face_locations = face_recognition.face_locations(image) #To get array of coords of eeach face in the group image

print(face_locations)
print(f'There are {len(face_locations)} people in this image.')