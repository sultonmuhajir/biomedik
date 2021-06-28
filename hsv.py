def hsv(x):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,x]
	return hsv