import numpy as np
import copy

def update_gd_parameters(parameters, back_caches, lr):
    """Update network parameters using vanilla gradient descent.

    Args:
        parameters (dict): Current parameters of the network, with keys:
            - 'W{i}' (np.ndarray): weight matrix for layer i.
            - 'b{i}' (np.ndarray): bias vector for layer i.
        back_caches (dict): Gradients from backpropagation, with keys:
            - 'dW{i}' (np.ndarray): gradient of loss w.r.t. W{i}.
            - 'db{i}' (np.ndarray): gradient of loss w.r.t. b{i}.
        lr (float): Learning rate.

    Returns:
        dict: Updated parameters after one gradient descent step. Each
            W{i} and b{i} is adjusted by subtracting lr * gradient.
    """

    parameters = copy.deepcopy(parameters)
    layers = len(parameters) // 2 

    for l in range(layers):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - lr * back_caches["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - lr * back_caches["db" + str(l + 1)]

    return parameters


def adam(back_caches , parameters , v , s , t , learning_rate = 0.01 , beta1 = 0.9, beta2 = 0.999 ,epsilon = 1e-8):
    """Update parameters using the Adam optimization algorithm.

    Applies bias-corrected first and second moment estimates to perform
    an adaptive gradient update on each parameter.

    Args:
        back_caches (dict): Gradients from backprop, with keys:
            - 'dW{i}' and 'db{i}' for each layer i.
        parameters (dict): Current network parameters, with keys:
            - 'W{i}' and 'b{i}' for each layer i.
        v (dict): Exponential moving averages of past gradients (first moment).
        s (dict): Exponential moving averages of past squared gradients (second moment).
        t (int): Time step (iteration count) for bias correction.
        learning_rate (float, optional): Step size. Defaults to 0.01.
        beta1 (float, optional): Decay rate for the first moment. Defaults to 0.9.
        beta2 (float, optional): Decay rate for the second moment. Defaults to 0.999.
        epsilon (float, optional): Small constant to prevent division by zero. Defaults to 1e-8.

    Returns:
        dict: Updated parameters dictionary with the same keys as `parameters`.

    """

    parameters = copy.deepcopy(parameters)
    v_corrected = {}
    s_corrected = {}
    layers = len(parameters) // 2
    
    for i in range(1, layers + 1):
        v["dW" + str(i)] = beta1*(v["dW" + str(i)]) + (1-beta1) * back_caches["dW" + str(i)]
        v["db" + str(i)] = beta1*(v["db" + str(i)]) + (1-beta1) * back_caches["db" + str(i)]

        s["dW" + str(i)] = beta2*(s["dW" + str(i)]) + (1-beta2) * (back_caches["dW" + str(i)]**(2))
        s["db" + str(i)] = beta2*(s["db" + str(i)]) + (1-beta2) * (back_caches["db" + str(i)]**(2))

        v_corrected["dW" + str(i)] = v["dW" + str(i)] / (1 - beta1**t)
        v_corrected["db" + str(i)] = v["db" + str(i)] / (1 - beta1**t)

        s_corrected["dW" + str(i)] = s["dW" + str(i)] / (1 - beta2**t)
        s_corrected["db" + str(i)] = s["db" + str(i)] / (1 - beta2**t)

        
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate*(v_corrected["dW" + str(i)]/( np.sqrt( s_corrected["dW"+str(i)] ) + epsilon ))
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate*(v_corrected["db" + str(i)]/( np.sqrt( s_corrected["db"+str(i)] ) + epsilon ))
    
    return parameters 


def learning_decay(learning_rate0, epoch_num, decay_rate):
    """Compute an updated learning rate using inverse time decay.

    The learning rate is scaled by 1 / (1 + decay_rate * epoch_num), so it
    decreases over epochs.

    Args:
        learning_rate0 (float): Initial learning rate before decay.
        epoch_num (int): Current epoch index (often starting at 0).
        decay_rate (float): Decay coefficient controlling the rate of decrease.

    Returns:
        float: Decayed learning rate:
            learning_rate0 / (1 + decay_rate * epoch_num)
    """

    learning_rate = 1 / ((1 + decay_rate * epoch_num)) * learning_rate0
    return learning_rate

