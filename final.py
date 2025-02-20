import numpy as np
import cv2
import matplotlib.pyplot as plt

#******************************************************************************************#

#https://www.thepythoncode.com/article/contour-detection-opencv-python

"""# read the image
image = cv2.imread("./imgs/H 7 A 40x.tif")
image = image[:850,:] # [crop y, crop x]
# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# create a binary thresholded image
_, binary = cv2.threshold(gray, (255 * 0.2), 255, cv2.THRESH_BINARY)
# show it
#plt.imshow(binary, cmap="gray")
#plt.show()

# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# draw all contours
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

print(len(contours))

# show the image with the drawn contours
plt.imshow(image)
plt.show()"""

def avg(lst):
    sum = 0
    for elem in lst:
        sum += elem
    return sum/len(lst)

def calcPointsWH(pts, theta=0):
    # Measuring width of points at given angle
    th = theta * np.pi /180
    e = np.array([[np.cos(th), np.sin(th)]]).T
    es = np.array([[np.cos(th), np.sin(th)],[np.sin(th), np.cos(th)]]).T
    dists = np.dot(pts,es)
    wh = dists.max(axis=0) - dists.min(axis=0)
    print("==> theta: {}\n{}".format(theta, wh))
    return wh

def get_HW_ratio(img_name):

    img = cv2.imread(img_name, 1) #cv2.IMREAD_GRAYSCALE)
    img = img[:850,:] # [crop y, crop x]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #print(img.shape)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret1,th1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #find contours
    contours, hierarchy = cv2.findContours(th1, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(th2, contours, -1, (0, 0, 255), 3)


    #height to width ratio
    height_width_BOUNDING = []
    hwRatio = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        #cv2.putText(th2, str(w), (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255))
        cv2.rectangle(th2, (x, y), (x + w, y + h), (255,255,255), 1)
        height_width_BOUNDING.append(h/w)
    
        pts = np.array(c)
        bestWidth = -10000000
        bestHeight = 0
        calcPointsWH(pts, theta=0)
        for theta in range(0, 91):
            wh = calcPointsWH(c, theta)
            if wh[0,0] > bestWidth:
                bestWidth = wh[0, 0]
                bestHeight = wh[0, 1]
        hwRatio.append(1- (bestHeight/bestWidth))
    
    cv2.imshow('image', th2)
    cv2.waitKey()
    return hwRatio, len(hwRatio)



img_name = "iseg/H 5 B 40x.png"
print(get_HW_ratio(img_name))