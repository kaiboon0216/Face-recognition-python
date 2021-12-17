"""
    File    : facematch.py
    Author  : kaiboon
    Date    : 27/5/2020
    Brief   : Return a string to identify whether two images are the same person
"""

import face_recognition

#face encodings is to take facial feature so that can compare with other

image_of_kb = face_recognition.load_image_file('./known/kaiboon/kb1.jpg')
kb_face_encoding = face_recognition.face_encodings(image_of_kb)[0]

unknown_image = face_recognition.load_image_file('./unknown/kb2.jpg')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

#Compare face
results = face_recognition.compare_faces(
    [kb_face_encoding], unknown_face_encoding
 )

if results[0]:
     print('This is kaiboon')
else:
    print('This is not kaiboon')