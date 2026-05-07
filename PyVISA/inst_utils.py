# -*- coding: utf-8 -*-
"""
Created on Fri May 23 21:35:52 2025

@author: Fede
"""

import numpy as np
import pyvisa
import time

# ------------------ INICIALIZACIÓN ------------------
def inst_start():
    rm = pyvisa.ResourceManager()
    inst = rm.list_resources()
    print("Instrumentos detectados:", inst)

    inst_gen = 'USB0::0xF4ED::0xEE3A::SDG000::INSTR'
    gen = rm.open_resource(inst_gen)

    inst_osc = 'USB0::0x0699::0x0368::C017059::INSTR'
    osci = rm.open_resource(inst_osc)

    print("ID generador:", gen.query('*IDN?'))
    print("ID osciloscopio:", osci.query('*IDN?'))

    osci.write('DAT:ENC RPB')
    osci.write('DAT:WID 1')
    osci.write('HEADER OFF')

    return gen, osci

# ------------------ CONFIGURACIÓN DEL GENERADOR ------------------
def config_gen(gen, ch='C1', wvtp='SINE', frq=1000, amp=2):
    gen.write(f'{ch}:OUTP OFF')
    gen.write(f'{ch}:BSWV WVTP,{wvtp},FRQ,{frq},AMP,{amp}')
    time.sleep(1)
    gen.write(f'{ch}:OUTP ON')

# ------------------ MEDICIONES DEL OSCILOSCOPIO ------------------
def medir_p2p(scope, channel):
    scope.write(f'MEASU:MEAS1:SOURCE CH{channel}')
    scope.write('MEASU:MEAS1:TYPE PK2pk')
    time.sleep(0.5)
    val = float(scope.query('MEASU:MEAS1:VALUE?'))
    sca = float(scope.query(f'CH{channel}:SCAle?'))
    return val, sca

def medir_rms(scope, channel):
    scope.write(f'MEASU:MEAS1:SOURCE {channel}')
    scope.write('MEASU:MEAS1:TYPE CRMS')
    time.sleep(0.5)
    val = float(scope.query('MEASU:MEAS1:VALUE?'))
    sca = float(scope.query(f'CH{channel}:SCALE?'))
    return val, sca

# ------------------ ADQUISICIÓN DE WAVEFORM ------------------
def get_waveform(scope, channel='1'):
    scope.write(f'DAT:SOU CH{channel}')
    scope.write('ACQ:STATE OFF')
    xze, xin, yze, ymu, yoff = scope.query_ascii_values(
        'WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFF?;',separator=';'
        )
    data = scope.query_binary_values(
        'CURV?', datatype='B', container=np.array
        ) 

    t = xze + np.arange(len(data)) * xin      # Conversion a tiempo en segundos
    V = (data - yoff) * ymu + yze             # Conversion a voltaje en Volts
    
    return t, V    

# -------------------------------------------------------------

# Esto es un pequeño script armado por chatgpt para que cuando
# importo esto se importe bien

import sys
import inspect

# Obtener el módulo actual (este archivo)
mod = sys.modules[__name__]

# Listar todas las funciones definidas en este módulo
__all__ = [name for name, obj in inspect.getmembers(mod, inspect.isfunction) if obj.__module__ == __name__]


