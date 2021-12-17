"""
    File    : pullfaces.py
    Author  : kaiboon
    Date    : 28/5/2020
    Brief   : Return the images of the face in the picture. 
              The image's name will be the top coordinates of the face.
"""

from PIL import Image
import face_recognition

image = face_recognition.load_image_file("./groups/kbwbszxt.jpg")
face_locations = face_recognition.face_locations(image) #To get array of coords of eeach face in the group image

for face_location in face_locations:
    top, right, bottom, left = face_location

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    #pil_image.show()
    pil_image.save(f'{top}.jpg')
