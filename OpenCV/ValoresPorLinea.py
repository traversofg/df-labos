import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

# Defino colores en BGR nomas por simpleza despues
BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)

fname = "./AnillosDeNewton/Distancia focal- 5cm hasta el vidrio- 3,5 hasta el objeto.jpg"

img = cv.imread(fname)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray_eq = cv.equalizeHist(gray)
gray_blurred = cv.GaussianBlur(gray_eq, (5,5), 0)

height, width = img.shape[:2]
print(height)
print(width)

start_point = (int(width / 2), int(height/2))
end_point   = (550, 0)

cv.circle(gray_blurred, start_point, 10, RED, 2)
cv.circle(gray_blurred, end_point, 10, RED, 2)

def get_intensities(img, start, end):

    line_points = []
    
    # cv.line(img, start, end, GREEN, 2)

    lenx = end[0] - start[0]
    leny = end[1] - start[1]
    lendiag = int(round(np.sqrt(lenx**2 + leny**2)))

    for i in range(lendiag):
        x = int(start[0] + lenx * (i/lendiag))
        y = int(start[1] + leny * (i/lendiag))
        line_points.append((x,y))

    intensities = [img[(y, x)] for x,y in line_points]

    return intensities, line_points

intensidades, line_point = get_intensities(gray_blurred, start_point, end_point)

print("Valores de intensidad a lo largo de la línea: \n", intensidades)

print('\n\n\n', gray_blurred[(0,549)])   

cv.imshow("", gray_blurred)
cv.waitKey(0)
cv.destroyAllWindows()

plt.figure()
plt.plot(
    range(len(intensidades)), intensidades
)
plt.show()