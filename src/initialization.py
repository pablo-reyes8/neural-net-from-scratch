import numpy as np

def iniciar_parametros(shape_nn:list , inicialization='Rand' , he_init = None , escala = 0.01):
    """Initialize neural network parameters.

    Supports standard random scaling or selective He initialization.

    Args:
        shape_nn (list of int): Layer sizes including input, e.g. [n_x, n_h1, ..., n_y].
        initialization (str, optional): 
            - 'Rand': random normal * scale  
            - 'He_Normal': He initialization on layers flagged in `he_init`.  
            Defaults to 'Rand'.
        he_init (list of bool, optional): Flags (one per layer transition) indicating
            which layers use He initialization when `initialization='He_Normal'`.
            Length must be `len(shape_nn) - 1`. Defaults to None.
        scale (float, optional): Scaling factor for the 'Rand' method. Defaults to 0.01.

    Returns:
        dict or str:
            If successful, returns a dict with keys:
                W{i} (ndarray): weights of layer i, shape (shape_nn[i], shape_nn[i-1])
                b{i} (ndarray): biases of layer i, shape (shape_nn[i], 1)
            If invalid options are provided, returns an error message string.

    Raises:
        ValueError: 
            - If `initialization` is not 'Rand' or 'He_Normal'.
            - If `he_init` length mismatches `len(shape_nn) - 1` when using 'He_Normal'.
    """

    parameters = {}

    dim_layer = len(shape_nn)
    bools = [False] + [bool(x) for x in he_init]

    if inicialization == 'He_Normal' and he_init == None:
        bools = [False] + [1 for _ in len(shape_nn)-1]

    if inicialization == 'Rand':
        for i in range(1, dim_layer):
            parameters['W' + str(i)] = np.random.randn(shape_nn[i],shape_nn[i-1]) * escala
            parameters['b' + str(i)] = np.zeros((shape_nn[i], 1))

    elif inicialization == 'He_Normal':
        if len(he_init) != (len(bools)-1):
            raise ValueError('`he_init` debe tener longitud igual a `len(shape_nn) - 1`')
        
        for i in range(1, dim_layer):
            if bools[i]:
                parameters['W' + str(i)] = np.random.randn(shape_nn[i],shape_nn[i-1]) * np.sqrt(2/shape_nn[i-1])
                parameters['b' + str(i)] = np.zeros((shape_nn[i], 1))
            else:
                parameters['W' + str(i)] = np.random.randn(shape_nn[i],shape_nn[i-1]) * escala
                parameters['b' + str(i)] = np.zeros((shape_nn[i], 1))
    else:
        raise ValueError("`inicialization` debe ser 'Rand' o 'He_Normal'")
        
    return parameters


def iniciar_params_adam(parameters):
    """Initialize Adam optimizer’s first and second moment variables.

    Args:
        parameters (dict): Network parameters with keys:
            - 'W{i}' (np.ndarray): weight matrix for layer i.
            - 'b{i}' (np.ndarray): bias vector for layer i.
          There should be 2·L entries for an L-layer network.

    Returns:
        tuple of dict:
            v (dict): Initialized first moment estimates with keys
                'dW{i}' and 'db{i}', each an array of zeros matching
                the shape of the corresponding parameter.
            s (dict): Initialized second moment estimates with keys
                'dW{i}' and 'db{i}', each an array of zeros matching
                the shape of the corresponding parameter.

    """
     
    layers  = len(parameters) // 2 
    v = {}
    s = {}

    for l in range(1, layers + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v, s
