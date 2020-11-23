import cv2
import numpy as np 
import imutils
from imutils.object_detection import non_max_suppression
import time

net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]



# To capture video from webcam. 
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video1.mp4')
start = time.time()
frame_id = 0
font = cv2.FONT_HERSHEY_PLAIN

ppl_cnt = 0
people = []

while True:
    # Read the frame
    _, img = cap.read()
    frame_id += 1
    (height, width, channel) = img.shape   
    if frame_id%40: 
        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0,0,0), True, crop=False)

    #for b in blob:
    #   for n, blob_img in enumerate(b):
    #      cv2.imshow(str(n), blob_img)


        net.setInput(blob)
        outs = net.forward(outputlayers)

        class_ids = []
        confidences=[]
        boxes=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    #detection
                    w = int(detection[2]* width)
                    h = int(detection[3] * height)
                    x = int(detection[0]*width - w/2)
                    y = int(detection[1]*height - h/2)
                    boxes.append([x,y,w,h]) 
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    print(class_ids)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(img, label, (x, y+30), font, 1, (255,255,255),2)
                cv2.putText(img, str(class_ids[i]), (x, y+40), font, 1, (255,255,255),2)
    
    time_length = time.time() - start
    fps = frame_id / time_length
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
