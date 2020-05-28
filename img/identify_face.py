"""
    File    : identify_face.py
    Author  : kaiboon
    Date    : 27/5/2020
    Brief   : Identify the face of peoples in the images. 
              If the face is in database, it will return the name of that person, else, it return unknown face.
              We use PIL instead of opencv in this file.
"""

import face_recognition
from PIL import Image, ImageDraw

TOLERANCE = 0.425

image_of_kb = face_recognition.load_image_file('./known/kaiboon/kb1.jpg')
kb_face_encoding = face_recognition.face_encodings(image_of_kb)[0]

image_of_wb = face_recognition.load_image_file('./known/waibing/wb2.jpg')
wb_face_encoding = face_recognition.face_encodings(image_of_wb)[0]

# Create array of encodings and names 
known_face_encodings = [
    kb_face_encoding,
    wb_face_encoding    
]

known_face_names = [
    "Kaiboon",
    "Waibing"
]

#Load test image to find faces in the
test_image = face_recognition.load_image_file('./groups/kbwbszxt.jpg')

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

#Convert to PIL format
pil_image = Image.fromarray(test_image)

#Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image 
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
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

#Save image
#pil_image.save('identify.jpg')