import cv2
import numpy as np
 # Read 
 # image
image = cv2.imread("91njRR8xEEL._AC_SL1500_.jpg")
im_in = cv2.imread("91njRR8xEEL._AC_SL1500_.jpg", cv2.IMREAD_GRAYSCALE)
th, im_th = cv2.threshold(im_in, 200, 255, cv2.THRESH_BINARY_INV);
im_floodfill = im_th.copy()# Mask used to flood filling.
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 # Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255);
 # Invert floodfilled 
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 # Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv# Display images.

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))
res = cv2.morphologyEx(im_out,cv2.MORPH_OPEN, kernel)
cnts = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    area = cv2.contourArea(c)
    if len(approx) > 5 and area > 1000 and area < 500000:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 4)
cv2.imshow('result', res)
cv2.imshow("Foreground", image)
cv2.waitKey(0)