import numpy as np

def get_derivatives(func: callable):
    
    import sympy as sp

    varnames: tuple[str] = func.__code__.co_varnames

    symbols = [sp.Symbol(var) for var in varnames]

    expr = func(*symbols)

    derivatives = list()
    for var in symbols:
        deriv = sp.diff(expr, var)
        derivatives.append(deriv)

    lambda_deriv = [sp.lambdify(symbols, deriv, 'numpy') for deriv in derivatives]

    print()
    for var,derivative in zip(symbols,derivatives):
        print(
            f"Derivada de '{expr}' respecto de {var}: {derivative}"
        )
    print()

    return lambda_deriv

def density (mass, volume):
    return mass/volume

def err_propagation(func, data, sigma):
    
    """
    Calculate the uncertainty of a given function via the method
    of error propagation
    
    Parameters:
    -----------

    func: callable
        Function to which to the uncertainty
    data: array_like
        Experimental data
    sigma: array_like
        Must have the same dimensions as 'data'

    Returns:
    --------
    

    """
    # sigma = sigma.T
    data = data.T

    derivatives = get_derivatives(func)

    evaluated_derivatives = [deriv(*data[i])
      for i,deriv in enumerate(derivatives)]

    # n_params = func.__code__.co_argcount


    def sigma_func():
        """"
        terms = []
        for i in range(n_params)YA ESTOY HACIENDO CUALQUIER COSA ME VOY A DORMIR:
            term = (deriv(*data[i]) * sigma[i])**2
            terms.append(term)
        
        return terms
        """
        pass

    terminos = sigma_func()
    # Bastante seguro que todo esto está mal pero lo sigo después
    
    return np.sqrt(sum(terminos))

"""     TESTINN     """

def crear_datos():
    # Simulan ser datos de masa y volumen con sus errores

    dat = np.array([
        [i for i in range(1, 11)],  # masa
        [1+i/4 for i in range(1, 11)]  # volumen
    ])

    err = np.array([
        [m * 0.3 for m in dat[0]],
        [.2 + v * 0.2 for v in dat[1]],
    ])

    print('\t',
        'Masas: ', dat[0],'\n',
        'Volumenes: ', dat[1], '\n'
    )
    print('-----')
    print(
        'Mass error: ', err[0], '\n',
        'Vol error: ', err[1],'\n'
    )

    return dat, err

datos, error = crear_datos()

def test():

    densidad = density(*datos)
    errores_ = err_propagation(density, datos, error)

    print(
        'Valores de densidad: ', densidad
    )
    print(
        'Errores de densidad: ', errores_
    )

drho_dm = lambda v: 1/v
drho_dv = lambda m,v: -m/(v**2)

def error_densidad_teorico(mass, vol, sigma_mass, sigma_vol):

    term_dm = ( drho_dm(vol) * sigma_mass )**2
    term_dv = ( drho_dv(mass,vol) * sigma_vol )**2

    err = np.sqrt(term_dm + term_dv)

    print()
    print(
        'Errores teóricos: ', err
    )

test()
error_densidad_teorico(*datos, *error)

def test_get_deriv():

    density_deriv = get_derivatives(density)

    # Evaluar las derivadas en ciertos datos (por ejemplo, masa=10 y volumen=2)
    mass_value = datos[0][0]
    volume_value = datos[1][0]

    # Evaluar cada derivada en esos valores
    for i, deriv_func in enumerate(density_deriv):
        result = deriv_func(mass_value, volume_value)
        print(f"Derivada {i+1} evaluada en (masa={mass_value}, volumen={volume_value}): {result}")

test_get_deriv()