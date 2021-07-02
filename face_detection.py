#The following program detects faces better using caffe model with opencv dnn and saves detected faces on pressing 'enter' key.

# -*- coding: utf-8 -*-
'''By Ajay'''

#importing required libraries
import cv2
import numpy as np
import keyboard as kb
import os
import time
import argparse

#args = argparse.ArgumentParser()
#args.add_argument("-p", "--path", required=True, help="path to folder containing caffe and prototxt")
#args.add_argument("-c", "--confidence", type=float, default=0.7, help="minimum threshold value to detect faces")
#arguments = vars(args.parse_args())

##importing caffe model 
#face_proto = os.path.sep.join([arguments["path"],"deploy.prototxt.txt"]) # face detection proto file
#face_caffe = os.path.sep.join([arguments["path"],"res10_300x300_ssd_iter_140000.caffemodel"])  # face detection caffe model file

face_proto = "deploy.prototxt.txt" # face detection proto file
face_caffe = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(face_proto, face_caffe)

#initiating face detection
cap=cv2.VideoCapture(0)
j=1
while 1:
    _,img=cap.read()
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
    (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    #to draw box around faces on image
    for i in range(faces.shape[2]):
        try:
            confidence = faces[0, 0, i, 2]
            #if confidence > arguments["confidence"]:
            if confidence > 0.6:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
                detected_face=img[y:y1,x:x1]
                detected_face=cv2.resize(detected_face,(160,160),cv2.INTER_AREA)
            #saving the detected face by pressing 'enter' button
            if kb.is_pressed('enter')==True:
                time.sleep(0.1)
                os.chdir('F:\\Ajay')
                cv2.imwrite(str(j)+'.jpg',detected_face)
                print('done')
                j+=1
        except:
            pass
                
    cv2.imshow('detecting_face',img)
    k=cv2.waitKey(10)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
