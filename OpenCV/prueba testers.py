import cv2 as cv
import numpy as np

def nothing(a):
    pass

cv.namedWindow('Tester')
cv.createTrackbar('HH','Tester',0,179,nothing)
cv.createTrackbar('HL','Tester',0,179,nothing)
cv.createTrackbar('SH','Tester',0,255,nothing)
cv.createTrackbar('SL','Tester',0,255,nothing)
cv.createTrackbar('VH','Tester',0,255,nothing)
cv.createTrackbar('VL','Tester',0,255,nothing)

img = cv.imread('./AnillosDeNewton/WIN_20241016_12_52_14_Pro.jpeg')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 

key = cv.waitKey(1)

while not key == ord('q'):

    hh = cv.getTrackbarPos('HH','Tester')
    hl = cv.getTrackbarPos('HL','Tester')
    sh = cv.getTrackbarPos('SH','Tester')
    sl = cv.getTrackbarPos('SL','Tester')
    vh = cv.getTrackbarPos('VH','Tester')
    vl = cv.getTrackbarPos('VL','Tester')

    lower_white = np.array([hl,sl,vl], dtype=np.uint8)
    upper_white = np.array([hh,sh,vh], dtype=np.uint8)
    
    mask = cv.inRange(hsv, lower_white, upper_white)

    cv.imshow("Mascara", mask)
    cv.imshow("Imagen", img)

    key = cv.waitKey(1)
    
cv.destroyAllWindows()