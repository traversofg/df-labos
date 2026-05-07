# -*- coding: utf-8 -*-
"""
Created on Fri May 23 20:53:04 2025

@author: Fede
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyvisa

from inst_utils import *

inst_start()

def medir_trafo(Np, Ns, CHp=1, CHs=2):
        
    Vp, sca_p = medir_p2p(osci, CHp)
    Vs, sca_s = medir_p2p(osci, CHs)
    
    return {
        'N_primario': Np,
        'N_secundario': Ns,
        'V_primario': Vp,
        'V_secundario': Vs,
        'ratio_espiras': Ns / Np,
        'ratio_voltaje': Vs / Vp
    }

# Me armo una listita para ir guardando todo

datos = []

#%% Para ir cambiando manualmente Np y Ns

Np = ...
Ns = ...

punto = medir_trafo(Np, Ns)

datos.append(punto)

#%% Para que el programa lo vaya cambiando solo

# No creo que este muy bueno esto porque podes perder todos los datos creo

"""
Np_cte = ...

Ns_min = ...
Ns_max = ...
step_Ns = ...

wait_time = ...

Ns_vals = np.arange( Ns_min, Ns_max + step_Ns, step_Ns)

for Ns in Ns_vals:
    
    punto = medir_trafo(Np_cte, Ns)
    datos.append(punto)
    
    time.sleep(wait_time)
"""

#%% Guardar los datos en dataframe y despues a csv

path = r""

df = pd.DataFrame(datos)
df.to_csv(path+r"/datos_trafo.csv", index=False)

#%% Plotear  (Vs/Vp - Ns/Np)

ratio_espiras = df['ratio_espiras']
ratio_voltaje = df['ratio_voltaje']

plt.figure( figsize=(11,6) )

plt.plot(ratio_espiras, ratio_voltaje, '.')

plt.xlabel('Ratio espiras $(N_s / N_p)$')
plt.ylabel('Ratio voltajes $(V_s / V_p)$')

title = plt.gca().get_title()

plt.savefig('')
