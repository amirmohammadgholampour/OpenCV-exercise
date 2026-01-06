import cv2 as cv 
import numpy as np 
from matplotlib import pyplot as plt 

# read image 
original_img = cv.imread("../images/hidden.png", cv.IMREAD_GRAYSCALE) 

# global equalization method 
global_equalize = cv.equalizeHist(original_img) 

# CLAHE (Contrast Limited Adaptive Histgram Equalization) method
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 
CLAHE_method = clahe.apply(original_img) 

plt.figure(figsize=[15, 7]) 
plt.subplot(131); plt.imshow(original_img, cmap="gray"); plt.title("original image") 
plt.subplot(132); plt.imshow(global_equalize, cmap="gray"); plt.title("Global equalization method") 
plt.subplot(133); plt.imshow(CLAHE_method, cmap="gray"); plt.title("CLAHE method") 
plt.tight_layout() 
plt.show()