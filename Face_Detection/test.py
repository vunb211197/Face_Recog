import numpy as np
import cv2
import time
from pre_processing import face_detec
import os


cap = cv2.VideoCapture('chuong.mp4')


while cap.isOpened():
    ret,frame = cap.read()

    if not ret:
        continue
    
    frame = face_detec(frame)

    cv2.imshow('My-video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  # destroy all the opened windows