import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 

# create video capture 
video_capture = cv.VideoCapture("videos/akhavan-green-bg.mp4") 

# window config 
window_name = "video" 
cv.namedWindow(window_name, cv.WINDOW_NORMAL) 

# read background image 
background_img = cv.imread("images/balloon.png") 

# get width, height frame 
frame_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))


while True: 
    ret, frame = video_capture.read() # read frame and video capture
    if not ret: # if video is not, break loop 
        break 
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV) # convert bgr frame to hsv 

    # resize background image 
    resize_background_img = cv.resize(background_img, (frame_width, frame_height)) 

    # create mask for 'green' background image
    lower = np.array([40, 50, 50])
    upper = np.array([70, 255, 255])
    mask = cv.inRange(hsv_frame, lower, upper)
    mask_reverse = cv.bitwise_not(mask) # reversed mask 

    # update frame 
    frame_without_bg = cv.bitwise_and(frame, frame, mask=mask_reverse)
    frame_with_new_bg = cv.bitwise_and(resize_background_img, resize_background_img, mask=mask) 
    output = cv.add(frame_without_bg, frame_with_new_bg) 

    
    cv.imshow(window_name, output) # show frame 
    key = cv.waitKey(50) 
    if key == ord("q"): # press "q" to exit 
        print("exit") 
        break 
cv.destroyAllWindows() 