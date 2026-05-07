import cv2 as cv
import numpy as np

# Defino colores en BGR nomas por simpleza despues
BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

fname = "./AnillosDeNewton/Distancia focal- 5cm hasta el vidrio- 3,5 hasta el objeto.jpg"
img = cv.imread(fname)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_eq = cv.equalizeHist(gray)

gray_blurred = cv.GaussianBlur(gray_eq, (5,5), 0)

thresh = cv.adaptiveThreshold(
        src = gray_blurred,
        maxValue = 255,
        adaptiveMethod = cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType = cv.THRESH_BINARY,
        blockSize = 11,
        C = 2
)   

contours, _ = cv.findContours(
    thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)

# Procesar contornos para obtener radios e interfranjas
radios = []
for contour in contours:
    (x, y), radius = cv.minEnclosingCircle(contour)
    radios.append(radius)
    
    # Dibujar el círculo en la imagen original
    center = (int(x), int(y))
    radius = int(radius)
    cv.circle(img, center, radius, (0, 255, 0), 2)  # Círculo verde

# Ordenar radios y calcular interfranjas
radios = sorted(radios)
interfranjas = [radios[i + 1] - radios[i] 
                for i in range(len(radios) - 1)]

print("Interfranjas entre círculos:", interfranjas)

cv.imshow("Anillos de Newton", img)
cv.waitKey(0)
cv.destroyAllWindows()