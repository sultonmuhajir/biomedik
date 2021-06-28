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