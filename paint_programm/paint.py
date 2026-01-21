import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

img = np.zeros((512, 512, 3), np.uint8) 

line_color = (0, 255, 0)
drawing = False 

def brush_paint(event, x,y, flags, param): 
    global drawing
    if event == cv.EVENT_LBUTTONDOWN: 
        drawing = True 
    elif event == cv.EVENT_MOUSEMOVE: 
        if drawing == True: 
            cv.line(img, (x,y), (x,y), line_color, 15)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False 

window_name = "paint" 
cv.namedWindow(window_name, cv.WINDOW_NORMAL) 
cv.setMouseCallback(window_name, brush_paint) 

while True: 
    cv.imshow(window_name, img) 

    key = cv.waitKey(1) 
    
    # color changes config
    if key == ord("r"):
        line_color = (0, 0, 255)
    elif key == ord("g"):
        line_color = (0, 255, 0)
    elif key == ord("b"):
        line_color = (255, 0, 0)
    elif key == ord("w"):
        line_color = (255, 255, 255)

    elif key == ord("q"):
        print("exit")
        break 

cv.destroyAllWindows()
plt.imshow(img[..., ::-1])
plt.show()