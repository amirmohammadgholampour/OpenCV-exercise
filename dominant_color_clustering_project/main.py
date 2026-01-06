import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.cluster import KMeans 

# read image 
original_img = cv.imread("../images/felfel-dolme.jpg") 

# reshape to a list of pixels 
flat_img = original_img.reshape((-1, 3)) 

# using kmeans algorithm to cluster pixels 
number_of_cluster = 5
kmeans = KMeans(n_clusters=number_of_cluster)
kmeans.fit(flat_img) 

# the cluster center out are dominant color
cluster_center = kmeans.cluster_centers_ 
dominant_color = np.array(cluster_center, dtype=np.uint8)

# put percent of cluster to total pixels
percentage = np.bincount(kmeans.labels_) / len(flat_img)
percentage_and_color = sorted(zip(percentage, dominant_color), reverse=True) 

# show dominant colors
block = np.ones((50,50,3),dtype='uint')
plt.figure(figsize=(12,8))
for i in range(number_of_cluster):
    plt.subplot(1,number_of_cluster,i+1)
    block[:] = percentage_and_color[i][1][::-1]
    plt.imshow(block)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(str(round(percentage_and_color[i][0]*100,2))+'%')
plt.show()