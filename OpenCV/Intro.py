import cv2 as cv
import numpy as np
import os

# Defino colores en BGR nomas por simpleza despues
BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

# Cargo la imagen y la paso a escala de grises
fname = "./AnillosDeNewton/Distancia focal- 5cm hasta el vidrio- 3,5 hasta el objeto.jpg"
img = cv.imread(fname)
img_copy = cv.imread(fname)
img_grises = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Me guardo la imagen binarizada (ByN) en la variable 'th'
_, th = cv.threshold(img_grises, 140, 250, cv.THRESH_BINARY)

# Con esto encontramos los contornos
cnts1, hier1 = cv.findContours(th, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# Dibujo y muestro la imagen con sus contornos
img_cnts1 = cv.drawContours(img_copy, cnts1, -1, GREEN, 1)

#cv.imshow("Original", img)
#cv.imshow("Binarizada", th)
#cv.imshow("Con contornos", img_cnts1)
#cv.waitKey(0)
#cv.destroyAllWindows()

############

# Pruebo otra forma, usando HSV

lower = np.array([19, 252, 142], dtype=np.uint8)
upper = np.array([179, 252, 201], dtype=np.uint8)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
mask = cv.inRange(
    src = hsv, 
    lowerb = lower,
    upperb = upper
    )

_, th_mask = cv.threshold(mask, 2, 255, cv.THRESH_MASK)

cnts_mask, h_mask = cv.findContours(
    th_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
)

cv.drawContours(mask, cnts_mask, -1, GREEN, 2)

cv.fastNlMeansDenoisingColored(
    src = cv.cvtColor(mask, cv.COLOR_GRAY2BGR),
    dst = None,
    h = 10,
    hColor = 10,
    templateWindowSize = 17,
    searchWindowSize = 21
)

cv.imshow("th", th_mask)
cv.imshow("Contornos", mask)
cv.imshow("Imagen", img)
cv.waitKey(0)
cv.destroyAllWindows()