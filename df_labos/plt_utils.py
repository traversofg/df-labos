import matplotlib.pyplot as plt
import numpy as np

def set_plt_params(
        figsize: tuple[int, int] = (11, 6),
        fontsize: int = 14,
        linewidth: float = 1.25,
        style: str = 'default'
) -> None:

    # Estilo del gráfico
    plt.style.use(style)

    # Tamaño de la figura
    plt.rcParams['figure.figsize'] = figsize

    # Aumentar el tamaño de las etiquetas
    plt.rcParams['font.size'] = fontsize

    # Controlar la transparencia de las líneas y los gráficos
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['lines.color'] = 'b'

    # Cambiar el estilo de las cuadrículas
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 1

    # Cambiar el estilo de los ejes
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.grid'] = True

    # Cambiar los colores de los ticks de los ejes
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    # Aumentar el tamaño de los ticks de los ejes
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    from cycler import cycler
    navy = (56 / 256, 74 / 256, 143 / 256)
    #teal = (106 / 256, 197 / 256, 179 / 256)
    #pink = [199 / 255, 99 / 255, 150 / 255]
    teal = (0.464, 0.820, 0.751)
    pink = (0.830, 0.388, 0.638)
    plt.rcParams['axes.prop_cycle'] = cycler(color=[teal, navy, pink])

def subplot_1d(): ...

def subplot_2d(
        nrows: int,
        ncols: int,
        data: dict[str, list[float]],
        figsize: tuple[int, int] = (15, 15),
        suav_func=None,
        **kwargs
) -> None:
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows * ncols > len(data):
        fig.delaxes(axs[nrows - 1, ncols - 1])

    keys: list[str] = list(data.keys())
    index: int = 0
    for i in range(nrows):
        for j in range(ncols):

            if index >= len(keys): break

            x, y = data[keys[index]]

            if suav_func: y = suav_func(y, **kwargs)

            axs[i, j].plot(x, y)
            axs[i, j].set_title(keys[index])

            index += 1


def moving_average(data, window_size) -> np.ndarray:
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')


def superponer(data: dict,
               labels=None,
               suav_func=None,
               window_size: int = 0,
               xlabel: str = '',
               ylabel: str = ''
               ) -> None:
    if labels is None:
        labels = []
    plt.figure()

    keys: list[str] = list(data.keys())
    for i in range(len(keys)):

        x = data[keys[i]][0]
        y = data[keys[i]][1]

        if suav_func: y = suav_func(y, window_size)

        if labels:
            plt.plot(x, y, label=labels[i])
        else:
            plt.plot(x, y)

    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)

    plt.show()
