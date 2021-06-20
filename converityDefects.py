import cv2
import numpy as np
img = cv2.imread('test.jpg')
img_gray = cv2.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255,0)
contours, hierarchy = cv2.findContours(thresh,2,1)
cnt = contours[0]

hull = cv2.convexHull(cnt, returnPoints = False)

defects = cv2.convexityDefects(cnt, hull)
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv.line(img,start,end,[0,255,0],2)
    cv.circle(img,far,5,[0,0,255],-1)
 
cv2.imshow('final_img',img)
cv2.waitKey(0)