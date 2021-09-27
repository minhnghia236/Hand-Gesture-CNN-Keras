import cv2
import os 
import time

labels = ['hi', 'peace', 'ok']
IMAGES_PATH = 'images/train/'
#IMAGES_PATH = 'images/validation/'

number_images = 200

frame_width = 640
frame_height = 480

roi_width = 250
roi_height = 250
roi_x = 20
roi_y = 20

for label in labels:
    cap = cv2.VideoCapture(1)
    os.mkdir(IMAGES_PATH + label)
    print('Collect images for ' + label)
    time.sleep(3)
    for image in range(number_images):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (frame_width, frame_height))
        frame = cv2.flip(frame, 1)

        start_point, end_point = (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height)
        cv2.rectangle(frame, start_point, end_point, color=(0, 255, 0), thickness=2)
        roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        roi_resize = cv2.resize(roi, (128, 128))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 36, 255, cv2.THRESH_BINARY_INV)
                
        cv2.imwrite(IMAGES_PATH + label + '/' + str(image) + '.jpg', roi)
        print('image ' + str(image))

        cv2.imshow('Output', frame)
        cv2.imshow('Roi', roi)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()