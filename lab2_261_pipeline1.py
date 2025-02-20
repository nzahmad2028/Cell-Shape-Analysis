import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import imutils

 
def general_testing():
   #get blue image
   #"./Desktop/COLLEGE/JUNIOR YEAR/02-261/lab2/LiveDead_imgs/MFGTMP_210413160001_A01f02d0.PNG"
   img_name = "./LiveDead_imgs/MFGTMP_210413160001_A01f02d0.PNG"
   img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
   img = img[:850,:]
   print(img.shape)
 
   #cv2.resize(img, (100,100))
   #cv2.imshow("image", img)
   #cv2.waitKey(0) # waits until a key is pressed
   #cv2.destroyAllWindows() # destroys the window showing image
 
   #global thresholding
   ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
   #ret2, th2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
   #ret3, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
 
   #Otsu thresholding
   ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
   # Otsu's thresholding after Gaussian filtering
   blur = cv2.GaussianBlur(img,(5,5),0)
   ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   #USING GAUSSIAN BLUR - BEST RESULTS
 
   #print images
   img_list = [img, th1, th2, th3]
   img_titles = ["original", "global thresholding",
   "otsu thresholding", "otsu after gaussian"]
   plt.figure(figsize=(10, 10))
   for i in range(len(img_list)):
       plt.subplot(2,2,i+1),plt.imshow(img_list[i], cmap="gray")
       plt.title(img_titles[i])
       plt.xticks([]),plt.yticks([])
 
   plt.show()
 
#****** Pipeline 1 ******#
#7th image - MFGTMP_210413160001_A01f04d0.PNG
#first image - MFGTMP_210413160001_A01f02d0.PNG
img_name = "images/H 7 A 40x.tif"
img = cv2.imread(img_name) #cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(img.shape)
 
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret1,th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
 
 
# watershedding
"""kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(th1,cv2.MORPH_CLOSE, kernel, iterations = 1)
dist = cv2.distanceTransform(closing, cv2.DIST_L2, 3)
ret, dist1 = cv2.threshold(dist, 0.6*dist.max(), 255, 0)
markers = np.zeros(dist.shape, dtype=np.int32)
dist_8u = dist1.astype('uint8')
contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
   cv2.drawContours(markers, contours, i, (i+1), -1)
markers = cv2.circle(markers, (15,15), 5, len(contours)+1, -1)
#markers = cv2.watershed(img, markers)
#img[markers == -1] = [0,0,255]"""
 
#find contours
contours, hierarchy = cv2.findContours(th1,
   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(th2, contours, -1, (0, 0, 255), 3)


img_list = [img, th1, th2]
img_titles = ["original", "otsu after gaussian", "contours"]#, "closing", "dist", "dist1", "markers"]
"""plt.figure(figsize=(8, 8))
for i in range(len(img_list)):
   plt.subplot(1,3,i+1),plt.imshow(img_list[i], cmap="gray")
   plt.title(img_titles[i])
   plt.xticks([]),plt.yticks([])"""


#watershedding (attempt 2)
# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(th1)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=th1)
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=th1)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
 
# loop over the unique labels returned by the Watershed
# algorithm

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue
	# otherwise, allocate memory for the label region and draw
	# it on the mask
   
	mask = np.zeros(img.shape, dtype="uint8")
	mask[labels == label] = 255
	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(img, (int(x), int(y)), int(r), (255, 0, 0), 2)
	#cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# show the output image

print("contours", len(contours)) 
print("labels", len(np.unique(labels)))
cv2.imshow("Output", img)
cv2.waitKey(0)


#this is the number of cells, the number of things with contour boundaries (only thresholding)
cv2.imshow('image', th1)
cv2.waitKey(0)

#get red img count
red_img_name = "images/H 7 A 40x.tif"
red_img = cv2.imread(red_img_name) #cv2.IMREAD_GRAYSCALE)
red_img = cv2.cvtColor(red_img, cv2.COLOR_RGB2GRAY)
red_img = red_img[:850,:]
labels_used = set()
red_bright_spots = 0
red_spot_coords = []
for x in range(len(labels)):
   for y in range(len(labels[0])):
       label = labels[x,y]
       if label == 0:
           continue
       value = red_img[x,y]
       point = (x, y)
       #point = (x,y, value)
       #print(red_img[x,y])
       if (label not in labels_used) and (value > 15):
           labels_used.add(label)
           red_bright_spots += 1
           #red_spot_coords.append((point, label))
           red_spot_coords.append(point)
print(red_spot_coords)
print("red bright spots ->", red_bright_spots)
img_backtorgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

for (x, y) in red_spot_coords:
    cv2.circle(img_backtorgb, (int(x), int(y)), 15, (0, 0, 255), 2)
cv2.imshow("Output", img_backtorgb)
cv2.waitKey(0)



