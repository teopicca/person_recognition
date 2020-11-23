import cv2
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import imutils
from imutils.object_detection import non_max_suppression
# Load the cascade
#body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#HOG CLASSIFIER
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# To capture video from webcam. 
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video1.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    img = imutils.resize(img, width=min(400, img.shape[1]))
    orig = img.copy()
    (bodies, weights) = hog.detectMultiScale(img, winStride=(4, 4),
            padding=(8, 8), scale=1.1)

    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    #bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # Draw the rectangle around each face

   # bodies = np.array([[x,y, x+h, y+w] for (x, y, w, h) in bodies])
   # suppression = non_max_suppression(bodies, probs=None, overlapThresh=0.65)
    print(len(bodies))
    for x, y, w, h in bodies:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
