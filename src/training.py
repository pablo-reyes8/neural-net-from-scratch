import numpy as np 
from layers import * 
from initialization import * 
from optimizers import * 
from utils import * 


def compute_cost(A_final , labels , tipe , caches ,regularization = False , lambda_reg = None ):
    """Compute the loss between predicted probabilities and true labels, with optional L2 regularization.

    Supports binary cross-entropy for two-class problems or categorical
    cross-entropy for multi-class problems. When `regularization=True`, adds
    an L2 penalty term λ/(2m)·∑‖W‖² over all weight matrices stored in `caches`.

    Args:
        A_final (np.ndarray): Predicted probabilities, shape (n_y, m).
        labels (np.ndarray): True labels, shape (n_y, m) or (m,); will be
            reshaped to (1, m) if necessary.
        tipe (str): Type of cost to compute:
            - 'BinaryCrossEntropy': binary cross-entropy loss.
            - 'CrossEntropy': categorical cross-entropy loss.
        caches (dict, optional): Dictionary of cached values from forward pass.
            Used to extract weight matrices 'W1', 'W2', … when `regularization=True`.
        regularization (bool, optional): If True, include L2 penalty. Defaults to False.
        lambda_reg (float, optional): Regularization strength λ for L2 penalty.
            Must be ≥ 0. Defaults to 0.0.

    Returns:
        float: The scalar loss value (including regularization term if enabled).

    Raises:
        ValueError: If `tipe` is not one of the supported cost types.
    """

    # Ensure labels are shape (n_y, m)
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)

    m = labels.shape[1]
    eps = 1e-15
    A_safe = np.clip(A_final, eps, 1 - eps)

    # Compute base cost
    if tipe == 'BinaryCrossEntropy':
        logprobs = (labels * np.log(A_safe) +
                    (1 - labels) * np.log(1 - A_safe))
        cost = - (1 / m) * np.sum(logprobs)

    elif tipe == 'CrossEntropy':
        logprobs = labels * np.log(A_safe)
        cost = - (1 / m) * np.sum(logprobs)

    else:
        raise ValueError("`tipe` must be 'BinaryCrossEntropy' or 'CrossEntropy'")

    # Add L2 regularization term if requested
    if regularization:
        if lambda_reg < 0:
            raise ValueError("`lambda_reg` must be non-negative for L2 regularization")
        
        if caches is None:
            raise ValueError("`caches` must be provided when using regularization")
        
        l2_sum = sum(np.sum(W**2) for name, W in caches.items() if name.startswith('W'))
        cost += (lambda_reg / (2 * m)) * l2_sum

    return float(np.squeeze(cost))


def predict(X , y , parameters, activations , train = False , reg = False , lambda_reg=None , way=None):
    """Make predictions with a trained binary classification network and evaluate performance.

    Args:
        X (np.ndarray): Input features of shape (n_x, m), where m is the number of examples.
        y (np.ndarray): True binary labels of shape (1, m) or (m,); will be compared element-wise.
        parameters (dict): Learned network parameters (weights and biases).
        activations (list of str): Activation names for each layer, length L.
        train (bool, optional): If False (default), prints cost and accuracy on the given data
            and returns predictions; if True, returns only accuracy (for training).

    Returns:
        np.ndarray or float:
            - If `train=False`: Returns `preds`, a (1, m) array of binary predictions {0,1}.
            - If `train=True`: Returns `accuracy`, the fraction of correct predictions.
    """

    cost_total = 0
    A_pred, _ = forward_pass(X, parameters, activations ) 

    if reg == False:
        cost_total = cost_total + compute_cost(A_pred , y , 'BinaryCrossEntropy', parameters)
    elif reg == 'L2':
        cost_total = cost_total + compute_cost(A_pred , y , 'BinaryCrossEntropy', parameters , reg , lambda_reg)
        
    preds = (A_pred > 0.5).astype(int) 
    accuracy = np.mean(preds == y) 

    if train == False and way== 'val':
        print(f"En validacion el modelo logro: cost = {cost_total:.6f} — accuracy = {accuracy*100:.2f}%")
        return preds
    elif train == False and way== 'test':
        print(f"En testeo el modelo logro: cost = {cost_total:.6f} — accuracy = {accuracy*100:.2f}%")
        return preds
    else:
        return accuracy
    

def model_nn_scratch(X, Y, layers_dims , activations, optimizer , init=None ,He_inits = None, learning_rate = 0.001, mini_batch_size = 128, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True, decay=None, decay_rate=1 , regularization = None , lambda_l2 = 0.001):
    
    """Train a deep neural network from scratch with optional optimizers, learning-rate decay, and L2 regularization.

    This function supports vanilla gradient descent or Adam, inverse-time learning-rate decay,
    and optional L2 weight decay. It returns the learned parameters and the cost history.

    Args:
        X (np.ndarray): Input data of shape (n_x, m), where m is the number of examples.
        Y (np.ndarray): True labels of shape (1, m).
        layers_dims (list[int]): Neuron counts per layer, including input and output sizes.
        activations (list[str]): Activation for each layer (length L), supports:
            'relu', 'sigmoid', 'tanh', 'softmax' (softmax only for final layer).
        optimizer (str): Optimization method, either 'gd' (gradient descent) or 'Adam'.
        init (str or None, optional): Weight initialization method:
            - None: random normal scaled by 0.01,
            - 'He': He normal initialization (requires `He_inits` flags).
        He_inits (list[bool], optional): Flags for applying He init per hidden layer (length L).
        learning_rate (float, optional): Initial learning rate. Defaults to 0.001.
        mini_batch_size (int, optional): Size of each mini-batch. Defaults to 128.
        beta (float, optional): Momentum parameter (currently unused). Defaults to 0.9.
        beta1 (float, optional): Exponential decay rate for Adam first moment. Defaults to 0.9.
        beta2 (float, optional): Exponential decay rate for Adam second moment. Defaults to 0.999.
        epsilon (float, optional): Small constant for Adam to prevent division by zero. Defaults to 1e-8.
        num_epochs (int, optional): Number of training epochs. Defaults to 5000.
        print_cost (bool, optional): If True, prints cost and accuracy at intervals. Defaults to True.
        decay (float or None, optional): If provided, applies inverse-time decay to the learning rate.
        decay_rate (float, optional): Decay rate for inverse-time schedule. Defaults to 1.
        regularization (str or None, optional): If 'L2', apply L2 regularization; otherwise no regularization.
        lambda_l2 (float, optional): L2 regularization strength λ (must be ≥ 0). Defaults to 0.0.

    Returns:
        tuple:
            parameters (dict): Learned weights 'W1'…'WL' and biases 'b1'…'bL'.
            costs (list[float]): Cost history per epoch (averaged over all examples).

    Raises:
        ValueError: If `init` or `optimizer` is unrecognized, or if `lambda_l2` is negative when using L2.

    Example:
        >>> params, cost_hist = model_nn_scratch(
        ...     X_train, Y_train,
        ...     layers_dims=[n_x, 64, 32, 1],
        ...     activations=['relu', 'relu', 'sigmoid'],
        ...     optimizer='Adam',
        ...     init='He',
        ...     He_inits=[True, True, False],
        ...     learning_rate=0.001,
        ...     num_epochs=1000,
        ...     regularization='L2',
        ...     lambda_l2=0.01
        ... )
    """
    
    costs = [] 
    t = 0
    m = X.shape[1]
    learning_rate0 = learning_rate

    # Initialize parameters
    if init == None:
        parameters = iniciar_parametros(layers_dims)
    elif init == 'He':
        parameters = iniciar_parametros(layers_dims , inicialization='He_Normal' , he_init = He_inits)
    else:
        raise ValueError('Inicializacion de pesos no valida')

    # Initialize optimizer state
    if optimizer == 'Adam':
        v, s = iniciar_params_adam(parameters)
    elif optimizer == 'gd':
        pass 
    else:
        raise ValueError('Optimizacion no reconocidio')
    
    # Determine print interval
    if num_epochs < 100:
        print_interval = 10
    else:
        print_interval = max(1, num_epochs // 10)

    # Training loop
    for i in range(num_epochs):
        minibatches = random_mini_batches(X, Y, mini_batch_size)
        epoch_cost = 0

        for minibatch_X, minibatch_Y in minibatches:

            A_final , caches_forward = forward_pass(minibatch_X , parameters , activations)

            if regularization == 'L2':
                epoch_cost = epoch_cost + compute_cost(A_final , minibatch_Y , 'BinaryCrossEntropy', caches_forward , regularization , lambda_l2)
                caches_backwards = back_propagation(A_final , minibatch_Y , activations , caches_forward , regularization , lambda_l2)
            else:
                epoch_cost = epoch_cost + compute_cost(A_final , minibatch_Y , 'BinaryCrossEntropy', caches_forward)
                caches_backwards = back_propagation(A_final , minibatch_Y , activations , caches_forward)
            
        
            if optimizer == "gd":
                parameters = update_gd_parameters(parameters, caches_backwards, learning_rate)
            elif optimizer == "Adam":
                t = t + 1
                parameters = adam(caches_backwards , parameters , v , s , t , learning_rate)

        # Average cost per example
        cost_avg = epoch_cost / m
        costs.append(cost_avg)
        
        if decay:
            learning_rate = learning_decay(learning_rate0, i, decay_rate)

        if print_cost and (i % print_interval == 0 or i == num_epochs-1):

            if regularization == 'L2':
                accuracy  = predict(minibatch_X ,  minibatch_Y , parameters, activations , train = True , reg=regularization , lambda_reg=lambda_l2)
            else:
                accuracy  = predict(minibatch_X ,  minibatch_Y , parameters, activations , train = True)

            print(f"Epoch {i:4d}/{num_epochs}: cost = {cost_avg:.6f} — accuracy = {accuracy*100:.2f}%")
            if decay:
                print(f"   lr = {learning_rate:.6e}")
            costs.append(cost_avg)

    return parameters , costs