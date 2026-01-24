# Import required packages 
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

# Image configuration 
original_img = cv.imread("../images/notes.png") # read BGR image 
gray_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY) # convert BGR image to gray scale 
gray_img_inverse = cv.bitwise_not(gray_img)

# Binarizing gray image
binary_img = cv.adaptiveThreshold(gray_img_inverse, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)

# Get the horizental lines and create kernel size 
columns = binary_img.shape[1] 
horizontal_size = columns // 30
kernel_size_columns = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1)) 
columns_output = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel_size_columns) 

# Get the vertical lines and create kernel size 
rows = binary_img.shape[0]
vertical_size = rows // 30 
kernel_size_rows = cv.getStructuringElement(cv.MORPH_RECT, (1,vertical_size))
vertical_output = cv.morphologyEx(binary_img, cv.MORPH_OPEN, kernel_size_rows) 
raw_result = cv.bitwise_not(vertical_output) 

# Improve finally result 
edges = cv.adaptiveThreshold(raw_result, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2)
kernel_size = cv.getStructuringElement(cv.MORPH_CROSS, (2, 2))
finally_result = cv.morphologyEx(edges, cv.MORPH_DILATE, kernel_size) 

# Display images 
images = [gray_img, binary_img, columns_output, vertical_output, edges, finally_result] 
titles = ["Original image", "Binary image", "Columns", "Verticals", "Edges", "Finally result"] 
fig, axes = plt.subplots(3, 2, figsize=(25, 10))
for ax, image, title in zip(axes.ravel(), images, titles): 
    ax.imshow(image, cmap="gray") 
    ax.set_title(title) 
plt.tight_layout() 
plt.show()