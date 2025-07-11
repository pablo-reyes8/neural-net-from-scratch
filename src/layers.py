import numpy as np
from activations import * 

def one_layer_forward(A_prev , w , b , activation):
    """Perform forward propagation for a single neural network layer.

    Args:
        A_prev (np.ndarray): Activations from the previous layer,
            shape (size_prev, m), where m is the batch size.
        w (np.ndarray): Weights matrix for the current layer,
            shape (size_current, size_prev).
        b (np.ndarray): Bias vector for the current layer,
            shape (size_current, 1).
        activation (str): Name of the activation function to apply.
            Supported values: 'relu', 'sigmoid', 'tanh', 'softmax'.

    Returns:
        tuple:
            A (np.ndarray): Activations for the current layer,
                shape (size_current, m).
            forward_vars (tuple): Cached values (Z, w, b) for use in backpropagation.

    Raises:
        KeyError: If `activation` is not one of the supported keys.
    """

    activaciones = {
    'relu': ReLu(),
    'sigmoid': Sigmoid(),
    'tanh': Tanh(),
    'leaky_relu': LeakyReLu(),
    'softmax': Softmax()}

    Z = w @ A_prev + b
    if activation == 'relu':
        A = activaciones['relu'](Z)

    elif activation == 'tanh':
        A = activaciones['tanh'](Z)

    elif activation == 'sigmoid':
        A = activaciones['sigmoid'](Z)

    elif activation == 'softmax':
        A = activaciones['softmax'](Z)

    forward_vars = (Z , w , b)

    return A , forward_vars



def forward_pass(X , parameters , activations):
    """Perform a full forward pass through a multi-layer neural network.

    Args:
        X (np.ndarray): Input data of shape (n_x, m), where m is the number of examples.
        parameters (dict): Dictionary containing network parameters:
            - 'W{i}' (np.ndarray): weight matrix for layer i.
            - 'b{i}' (np.ndarray): bias vector for layer i.
          There should be 2·L entries for an L-layer network.
        activations (list of str): List of activation names to apply at each layer,
            length L. Supported: 'relu', 'sigmoid', 'tanh', 'softmax'.

    Returns:
        tuple:
            A_final (np.ndarray): Activations from the last layer, shape (n_y, m).
            caches (dict): Cached values for backpropagation. Contains for each layer i:
                - 'A{i-1}': activations from previous layer,
                - 'Z{i}': pre-activation,
                - 'W{i}', 'b{i}': parameters for layer i,
                - 'A{i}': activations of layer i.

    Raises:
        ValueError: If the number of activations does not match the number of layers.
    """

    A = X 
    layers = len(parameters) // 2 
    if len(activations) != layers: 
        raise ValueError('El numero de funciones de activacion no coincide')
    
    caches = {}
    caches['A' + str(0)] = X

    for i in range(1, layers):
        A_prev = A
        A ,  forward_vars =  one_layer_forward(A_prev , parameters['W' + str(i)] , parameters['b' + str(i)] , activations[i-1])
        caches['W' + str(i)] = forward_vars[1]
        caches['b' + str(i)] = forward_vars[2]
        caches['Z' + str(i)] = forward_vars[0]
        caches['A' + str(i)] = A
    

    A_final , forward_vars_final =  one_layer_forward(A , parameters['W' + str(layers)] , parameters['b' + str(layers)] , activations[layers-1])
    caches['W' + str(layers)] = forward_vars_final[1]
    caches['b' + str(layers)] = forward_vars_final[2]
    caches['Z' + str(layers)] = forward_vars_final[0]

    return A_final , caches 


def back_propagation(A_fianl , labels , activations , caches , regularization = False , lambda_reg = None):
    """Perform backpropagation through a multi-layer neural network, with optional L2 regularization.

    Computes gradients of the loss with respect to parameters for each layer,
    assuming the final activation is either sigmoid (binary classification)
    or softmax (multi-class classification). When `regularization='L2'`, adds
    the derivative of the L2 penalty (λ/m·W) to each dW.

    Args:
        A_final (np.ndarray): Output activations from the last layer, shape (n_y, m).
        labels (np.ndarray): True labels, shape (n_y, m).
        activations (list of str): Activation names for each layer, length L.
            Supported: 'relu', 'sigmoid', 'tanh', 'softmax' (only for final layer).
        caches (dict): Cached forward values from `forward_pass`, containing for
            each layer i:
            - 'Z{i}': pre-activation, shape (n_i, m)
            - 'W{i}': weights, shape (n_i, n_{i-1})
            - 'A{i}': activations, shape (n_i, m)
        regularization (bool or str, optional): If 'L2', apply L2 regularization.
            If False (default), no regularization.
        lambda_reg (float, optional): Regularization strength λ for L2 penalty.
            Must be ≥ 0. Default is 0.0.

    Returns:
        dict: Gradients with keys for each layer i:
            - 'dZ{i}': gradient of loss w.r.t. Z{i}
            - 'dW{i}': gradient of loss w.r.t. W{i} (including λ·W/m if L2)
            - 'db{i}': gradient of loss w.r.t. b{i}

    Raises:
        ValueError: If the final activation is not 'sigmoid' or 'softmax',
                    or if `lambda_reg` < 0 when using L2 regularization.
    """

    activation_names = {'relu': ReLu() , 'sigmoid': Sigmoid() , 'tanh': Tanh() ,'leaky_relu': LeakyReLu()}
    back_prop_caches = {}
    layers = len(activations)
    m = labels.shape[1]

    if regularization == 'L2' and (lambda_reg is None or lambda_reg < 0):
        raise ValueError("Para L2 regularization, lambda_reg debe ser ≥ 0")
    
    #Compute Last Layer derivates:
    if activations[-1] in ('sigmoid', 'softmax'):
        dZ_fin = A_fianl - labels
        dW_fin = 1/m * (dZ_fin @ caches['A' + str(layers-1)].T)

        if regularization == 'L2' :
            dW_fin = dW_fin + (lambda_reg/m) * caches['W' + str(layers)]

        db_fin = 1/m * np.sum(dZ_fin , axis=1 , keepdims=True)

        back_prop_caches['dZ' + str(layers)]  , back_prop_caches['dW' + str(layers)] , back_prop_caches['db' + str(layers)] = dZ_fin ,dW_fin ,db_fin

    else:
        return f'La funcion de activacion final no es valida'
        
    # Calcular gradientes de todas las capas ocultas: 
    for i in range(layers-1 ,0 , -1):
        act_actual = activations[i-1]
        dA_i = caches['W' + str(i+1)].T @ back_prop_caches['dZ' + str(i+1)]
        dZ_i = dA_i * activation_names[act_actual](caches['Z' + str(i)] , forw_prop=False )
        dW_i = 1/m * (dZ_i @ caches['A' + str(i-1)].T)

        if regularization == 'L2':
            dW_i = dW_i + (lambda_reg/m) * caches['W' + str(i)]

        db_i = 1/m * np.sum(dZ_i , axis=1 , keepdims=True)

        back_prop_caches['dZ' + str(i)]  , back_prop_caches['dW' + str(i)]  , back_prop_caches['db' + str(i)] = dZ_i , dW_i ,db_i
    
    return back_prop_caches 
