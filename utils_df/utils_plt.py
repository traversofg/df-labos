import matplotlib.pyplot as plt
import numpy as np

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