import cv2 as cv
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def tirar_linea(fname: str,
                start: tuple[int, int],
                end: tuple[int, int],
                show_img: bool=False,
                show_img_nb: bool=False) -> np.ndarray:
    """
    Tira una linea en la imagen después de pasarla a escala de grises
    y devuelve la intensidad de cada punto de la linea, dentro de un
    array de numpy y con cada valor siendo un uint8 (un valor entero
    entre 0 y 255).

    Params:
    -------
    fname: str
        ruta de la imagen
    start: tuple[int, int]
        punto inicial de la linea en coord en px
    end: tuple[int, int]
        punto final de la linea en coord en px
    show_img: bool
        muestra las imagenes original y modificada en ventana emergente.
        NO FUNCIONA EN NOTEBOOKS, para eso está show_img_nb (para Jupyter,
        Google Colab, etc..)

    Returns:
    --------
    NDArray[np.uint8]:
        numpy ndarray con la intensidad de cada punto de la linea
    """

    # Colores en BGR, ni hace falta

    BLUE: tuple[int, int, int] = (255, 0, 0)
    GREEN: tuple[int, int, int] = (0, 255, 0)
    RED: tuple[int, int, int] = (0, 0, 255)

    img_original = cv.imread(fname)  # Tensor: (height, width, depth=3)
    img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)  # Matriz: (height, width)
    gray_blurred = cv.GaussianBlur(
        cv.equalizeHist(img_gray),
        (5, 5), 0
    )

    height, width = img_original.shape[:2]

    def get_intensidades(img, start, end):
        len_x: float = end[0] - start[0]
        len_y: float = end[1] - start[1]

        lendiag: int = int(round(np.sqrt(len_x ** 2 + len_y ** 2)))

        line_points: list[list[int]] = []
        for i in range(lendiag):
            x: int = int(start[0] + len_x * (i / lendiag))
            y: int = int(start[1] + len_y * (i / lendiag))
            line_points.append([x, y])

        valores: list[int] = [img[(y, x)] for x, y in line_points]

        return np.array(valores)

    img_a_analizar = img_gray
    intensidades: npt.NDArray[np.uint8] = get_intensidades(img_a_analizar, start, end)

    cv.line(img=img_a_analizar, pt1=start, pt2=end, color=RED, thickness=3)

    if show_img:
        cv.imshow(img_original)
        cv.imshow(img_a_analizar)
    elif show_img_nb:
        from google.colab.patches import cv2_imshow
        cv2_imshow(img_original)
        cv2_imshow(img_a_analizar)

    x = range(len(intensidades))

    plt.figure()
    plt.plot(
        x,
        intensidades
    )

    plt.xlabel('Distancia (px)')
    plt.ylabel('Intensidad')

    plt.show()

    return intensidades

def get_R_sq(f, xdata, ydata, params):
  
  xdata, ydata = np.array(xdata, ydata)
  
  residuals = ydata - f(xdata, *params)
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum( (ydata - np.mean(ydata))**2 )

  r_sq = 1 - ( ss_res / ss_tot )

  return r_sq;
