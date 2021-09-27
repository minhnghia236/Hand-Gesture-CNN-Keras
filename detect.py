import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

model = load_model('model.h5')

cap = cv2.VideoCapture(1)
frame_width = 640
frame_height = 480

roi_width = 250
roi_height = 250
roi_x = 20
roi_y = 20

font = cv2.FONT_HERSHEY_COMPLEX

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.resize(frame, (frame_width, frame_height))
    frame = cv2.flip(frame, 1)

    start_point, end_point = (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height)
    cv2.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=2)
    
    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 36, 255, cv2.THRESH_BINARY_INV)
    roi_reshape = roi.reshape(1, 64, 64, 1)

    predict = model.predict(roi_reshape)
    val = np.argmax(predict)
    print(val)
    if val==0:
        cv2.putText(frame, 'hi', (500,100), font, 1, (255, 0, 0), 3)
    if val==1:
        cv2.putText(frame, 'ok', (500,100), font, 1, (255, 0, 0), 3)
    if val==2:
        cv2.putText(frame, 'peace', (500,100), font, 1, (255, 0, 0), 3)

    cv2.imshow('Output', frame)
    cv2.imshow('Roi', roi)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()