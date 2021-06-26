import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

pathImg = "img/citra.jpg"
img = cv2.imread(pathImg)
mpImg = mpimg.imread(pathImg)
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def hsv(x):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,x]
	return hsv


def morfologi(img):
	ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
	return opening


def watershed():
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	sure_bg = cv2.dilate(morfologi(hsv(2)), kernel, iterations=12)
	dist_transform = cv2.distanceTransform(morfologi(hsv(2)), cv2.DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg, sure_fg)
	ret, markers = cv2.connectedComponents(sure_fg)
	markers = markers+1
	markers[unknown==255] = 0
	markers = cv2.watershed(img,markers)
	img[markers == -1] = [255,0,0]
	return markers


plt.subplot(2,3,1), plt.imshow(hsv(0),'gray'), plt.title('Hue')
plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2), plt.imshow(hsv(1),'gray'), plt.title('Saturation')
plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3), plt.imshow(hsv(2),'gray'), plt.title('Value')
plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4), plt.imshow(mpImg), plt.title('Image')
plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5), plt.imshow(morfologi(image),'gray'), plt.title('Morfologi')
plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6), plt.imshow(watershed(),'gray'), plt.title('Watershed')
plt.xticks([]), plt.yticks([])
plt.show()
