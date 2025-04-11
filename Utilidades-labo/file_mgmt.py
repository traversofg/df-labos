import numpy as np
from pandas import read_csv
from os import listdir
from os.path import join

def open_files(path: str,
               skip: int = 0,
               ext: str = '.csv',
               **kwargs
               ) -> dict[str, list[float]]:
    """
    Abre los archivos de una carpeta solo si terminan con la extension que
    nosotros querramos, los pasa a un DataFrame de Pandas y devuelve un
    diccionario con los nombres de los archivos y su DF correspondiente.

    Params:
    -------
    path: str
        ruta de la **carpeta**, no de un archivo
    skip: int
        nro de filas a skipear al abrir cada archivo, por default = 0
    ext: str
        extension con la que nos querramos quedar ('.csv', '.txt', ...),
        por default = '.csv'
    names: list[str]
        nombres de las columnas del DataFrame, por default = []

    Returns:
    --------
    dict[str, list[float]]:
        diccionario con los nombres de los archivos y su DF correspondiente.

    """

    # Obtengo todos los archivos en la carpeta
    files = listdir(path)

    # Filtro los que tienen la extension que me interesa
    files_ext = [file for file in files
                if file.endswith(ext)]

    data = []
    for archivo in files_ext:

        full_path = join(path, archivo)

        df = read_csv(full_path, skiprows=skip, **kwargs)
        # df.columns = [f'col {i}' for i in range(df.shape[1])]

        data.append(df.values.tolist())


    data_dict = {}
    for i in range(len(files_ext)):
        fname = files_ext[i].split('.')[0] # Le saco al key la extension de archivo
        fdata = np.array(data[i])
        data_dict[fname] = fdata.T

    return data_dict