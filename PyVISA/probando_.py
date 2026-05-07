import numpy as np
import matplotlib.pylab as plt
import pandas as pd

import os

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

os.chdir(os.getcwd())
# --------------------------------- #

def moving_avg(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')
__
def detectar_intervalo_descarga(t, v, window=51, poly=3, 
                                deriv_thresh=-0.5, min_duracion=5):
    """
    Detecta el intervalo de descarga en una señal de voltaje.
    """

    # Suavizado
    v_smooth = savgol_filter(v, window, poly)
    dv = np.gradient(v_smooth, t)

    # Buscar zonas donde la derivada es fuertemente negativa
    posibles = dv < deriv_thresh

    # Buscar bloques continuos de puntos negativos
    from itertools import groupby
    from operator import itemgetter

    indices_negativos = np.where(posibles)[0]
    bloques = []
    for k, g in groupby(enumerate(indices_negativos), lambda i: i[0] - i[1]):
        grupo = list(map(itemgetter(1), g))
        if len(grupo) >= min_duracion:
            bloques.append(grupo)

    if not bloques:
        raise ValueError("No se encontró una descarga clara.")

    # Elegir el primer bloque como inicio de descarga
    inicio_idx = bloques[0][0]
    fin_idx = bloques[0][-1]

    return inicio_idx, fin_idx


path = './RC variable'

data = {}
for file in os.listdir(path):

    data_temp  = pd.read_csv(f'{path}/{file}', sep=',')
    resistance = data_temp['resistencia'][0]

    data_temp['tiempo'] -= min(data_temp['tiempo'])
    data_temp['VCH2'] -= min(data_temp['VCH2'])

    if 'VCH1' in data_temp.columns:
        data_temp['VCH1'] -= min(data_temp['VCH1'])

    data[f'R={resistance}'] = data_temp


df = data['R=500']

t = df['tiempo']
v = df['VCH2']

i, f = detectar_intervalo_descarga(t, v, window=151)
t_unload = t[i:f]
v_unload = v[i:f]

plt.figure( figsize=(12,6) )

plt.plot(t,v,label='Señal completa',alpha=0.5)
plt.plot(t_unload, v_unload, label='Descarga')

plt.title('Descarga RC con R = 500 $\Omega$')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')

plt.legend()
plt.show()
