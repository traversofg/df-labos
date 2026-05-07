import numpy as np
import pyvisa
import time

RM = pyvisa.ResourceManager()
print(
    RM.list_resources()
)

class Osciloscopio:

    def __init__(self, rsc_mgr):
        self.rm = rsc_mgr
        self.inst = None
        self.current_channel = None

    def connect(self, rsc_name):
        self.inst = self.rm.open_resource(rsc_name)
        return self

    def disconnet(self):
        if self.inst:
            self.inst.close()

    def identify(self):
        """ Identificación del instrumento """
        if self.inst:
            print(
                'Instrument ID: ', self.inst.query('*IDN?')
            )
        return self

    def setup_default(self):
        """ Algunos parametros default de medición """
        self.inst.write('DAT:ENC RPB')
        self.inst.write('DAT:WID 1')
        self.inst.write('HEADER OFF')
        return self

    def set_channel(self, channel: int):
        """ Definir canal activo para medir """
        self.current_channel = channel
        self.inst.write(f'DAT:SOU CH{channel}')
        time.sleep(.5)

    def capture_waveform(self, channel: int = None):
        """ Obtener datos de forma de onda completa """
        if channel is None:
            channel = self.current_channel

        self.set_channel(channel)

        xze, xin, yze, ymu, yoff = self.inst.query_ascii_values(
            'WFMPRE:XZE?;XIN?;YZE?;YMU?;YOFF?;',
            separator=';'
        )

        data = self.inst.query_binary_values(
            'CURV?',
            datatype='B',
            container=np.array
        )

        time_data = xze + np.arange(len(data)) * xin
        volt_data = (data - yoff) * ymu + yze

        return time_data, volt_data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnet()

class Generador:

    def __init__(self, rsc_mgr):
        self.rm = rsc_mgr
        self.inst = None

    def connect(self, rsc_name: str):
        self.inst = self.rm.open_resource(rsc_name)
        return self

    def disconnect(self):
        if self.inst:
            self.inst.close()

    def identify(self):
        if self.inst:
            print(
                'Instrument ID: ', self.rm.query('*IDN?')
            )
        return self

    def set_waveform(self,
                     channel: int,
                     waveform: str,
                     freq: float,
                     amp: float = None):

        """ Configurar una onda de salida para el canal elegido """

        cmd = f'C{channel}:BSWV WVTP,{waveform},FRQ,{freq:.2f}'
        if amp is not None:
            cmd += f'AMP,{amp:.2f}'
        self.inst.write(cmd)
        time.sleep(1)
        return self

    def outp_ctrl(self, channel: int, state: bool):
        state_str = 'ON' if state else 'OFF'
        self.inst.write(f'C{channel}: OUTP {state_str}')
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

class BarridoFrecuencias:

    def __init__(self, osci_id, gen_id):
        self.osci = Osciloscopio(RM).connect(osci_id).setup_default()
        self.gen  = Generador(RM).connect(gen_id)

    def configure(self, start: float, end: float, amp: float,
                  step: int, range_type: str = 'step'):

        """Configuración del barrido de frecuencias.

        Args:
            start (float): Frecuencia de inicio del barrido en Hz.
            end (float): Frecuencia final del barrido en Hz (debe ser mayor que start).
            amp (float): Amplitud de la onda generada (asumida constante).
            step (int):
                Dependiendo de range_type, representa:\n
                - Espaciado entre frecuencias (si range_type='step')\n
                - Número total de puntos (si range_type='num')
            range_type (str, optional): Tipo de espaciado para el barrido. Opciones:\n
                - 'step': Genera frecuencias con paso fijo (np.arange)\n
                - 'num': Genera N puntos equiespaciados (np.linspace)
                Default: 'step'

        Returns:
            numpy.ndarray: Array con las frecuencias generadas para el barrido.

        Raises:
            ValueError: Si end <= start o si step no es positivo.
            KeyError: Si range_type no es 'step' ni 'num'.

        Examples:
            >> configure(1000, 10000, 100)  # Paso de 100 Hz
            >> configure(1000, 10000, 50, 'num')  # 50 puntos equiespaciados
        """

        if range_type == 'step':
            freqs = np.arange(start, end+step, step)
        elif range_type == 'num':
            freqs = np.linspace(start, end, step)

        else:
            raise KeyError(
                f"range_type tiene que ser 'step' o 'num'. Se recibió: '{range_type}'"
            )
        if end<=start:
            raise ValueError(f'Error: end = {end} <= start = {start}')

        self.frequencies = freqs
        self.amplitude = amp

    def run(self):
        pass

class BarridoAmplitudes:

    def __init__(self, osci_id, gen_id):
        self.scope = Osciloscopio(RM).connect(osci_id).setup_default()
        self.gen = Generador(RM).connect(gen_id)

    def configure(self, start: float, end: float, freq: float,
                  step: int, range_type: str = 'step'):

        if range_type == 'step':
            amps = np.arange(start, end+step, step)
        elif range_type == 'num':
            amps = np.linspace(start, end, step)

        else:
            raise KeyError(
                f"range_type tiene que ser 'step' o 'num'. Se recibió: '{range_type}'"
            )
        if end<=start:
            raise ValueError(f'Error: end = {end} <= start = {start}')

        self.amplitudes = amps
        self.freq = freq

    def run(self):
        pass