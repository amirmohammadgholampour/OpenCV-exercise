import cv2 as cv 
import numpy as np 
from matplotlib import pyplot as plt 

# create video capture 
video_capture = cv.VideoCapture("../videos/blue-track.mp4") 

# window config 
window_name = "video" 
cv.namedWindow(window_name, cv.WINDOW_NORMAL) 

while True: 
    # read frame from video capture 
    ret, frame = video_capture.read() 
    if not ret: # if video is not found, break  
        break 

    # convert BGR frame to HSV
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV) 

    # create mask 
    lower = np.array([110, 50, 50])
    upper = np.array([130, 255, 255])
    mask = cv.inRange(hsv_frame, lower, upper)

    # find and draw contour 
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    for contour in contours: 
        center, radius = cv.minEnclosingCircle(contour) 
        center = int(center[0]), int(center[1])
        radius = int(radius) 
        output = cv.circle(frame, center, radius, (0, 255, 0), 2) 
        
    # show frame
    cv.imshow(window_name, output) 
    key = cv.waitKey(100) 
    if key == ord("q"): # press 'q' button to exist 
        print("exit")
        break 
cv.destroyAllWindows()