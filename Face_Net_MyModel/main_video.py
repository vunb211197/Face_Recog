import numpy as np
import cv2
import time
import os
from svc import predict

#lấy được video 
cap = cv2.VideoCapture('chuong.mp4')

#mở video 
while cap.isOpened():
    #đọc từng frame của video
    ret,frame = cap.read()

    if not ret:
        continue
    # predict theo frame
    img = predict(frame,'False')
    
    cv2.imshow('My-video',img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  # destroy all the opened windows