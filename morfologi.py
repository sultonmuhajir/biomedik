def morfologi(img):
	ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
	return opening