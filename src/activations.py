import numpy as np 

class ReLu:
    """Rectified Linear Unit (ReLU) activation function.

    Applies the ReLU operation during forward propagation and computes
    its derivative during backpropagation.

    Usage:
        relu = ReLu()
        a = relu(z)             # forward: max(0, z)
        da_dz = relu(z, False)  # backward: derivative w.r.t. z
    """

    def __call__(self, z, forw_prop=True):
        """Compute ReLU activation or its derivative.

        Args:
            z (array-like): Input array of pre-activation values.
            forw_prop (bool, optional): If True (default), returns the
                forward pass ReLU outputs; if False, returns the derivative
                dReLU/dz for backpropagation.

        Returns:
            np.ndarray: Element-wise ReLU activations (if `forw_prop=True`)
                or element-wise gradients (if `forw_prop=False`).
        """

        z = np.asarray(z, dtype=np.float64)
        
        if forw_prop:
            return np.maximum(0, z)
        
        # Derivate
        dz = np.zeros_like(z)
        dz[z > 0] = 1
        return dz

class LeakyReLu:
    """Leaky Rectified Linear Unit (Leaky ReLU) activation function,
    sin __init__, con pendiente α fija.

    Forward:  f(z) = z     if z > 0
                    α·z    otherwise

    Backward: f'(z) = 1     if z > 0
                    α       otherwise
    """

    def __call__(self, z, forw_prop = True , alpha = 0.01):
        """
        Args:
            z (array-like): pre-activations.
            forw_prop (bool): True → forward; False → derivada.
        """
        z = np.asarray(z, dtype=np.float64)

        if forw_prop:
            return np.where(z > 0, z, alpha * z)
        
        # Derivate
        dz = np.ones_like(z)
        dz[z < 0] = alpha
        return dz



class Sigmoid:
    """Sigmoid activation function.

    Computes the logistic sigmoid during forward propagation and its
    derivative during backpropagation.

    Usage:
        sig = Sigmoid()
        a = sig(z)             # forward: 1 / (1 + exp(-z))
        da_dz = sig(z, False)  # backward: sigmoid(z) * (1 − sigmoid(z))
    """

    def __call__(self, z, forw_prop=True):
        """Compute sigmoid activation or its gradient.

        Args:
            z (array-like): Input pre-activation values.
            forw_prop (bool, optional): If True (default), returns the
                sigmoid outputs; if False, returns the derivative
                dσ/dz for backpropagation.

        Returns:
            np.ndarray: Element-wise sigmoid activations (if `forw_prop=True`)
                or gradients (if `forw_prop=False`).
        """

        z = np.asarray(z, dtype=np.float64)
        z = np.clip(z, -500, 500)

        sigm = 1 / (1 + np.exp(-z))
        if forw_prop:
            return sigm
        
        # Derivate
        return sigm * (1 - sigm)

class Tanh:
    """Hyperbolic tangent (tanh) activation function.

    Computes the tanh activation during forward propagation and its
    derivative during backpropagation.

    Usage:
        tanh = Tanh()
        a = tanh(z)             # forward: tanh(z)
        da_dz = tanh(z, False)  # backward: 1 − tanh(z)**2
    """

    def __call__(self, z, forw_prop=True):
        """Compute tanh activation or its gradient.

        Args:
            z (array-like): Input pre-activation values.
            forw_prop (bool, optional): If True (default), returns the
                tanh outputs; if False, returns the derivative
                d(tanh)/dz for backpropagation.

        Returns:
            np.ndarray: Element-wise tanh activations (if `forw_prop=True`)
                or gradients (if `forw_prop=False`).
        """

        z = np.asarray(z, dtype=np.float64)
        z = np.clip(z, -500, 500)

        tan = np.tanh(z)
        if forw_prop:
            return tan
        # Derivate

        return 1 - tan**2

class Softmax:
    """Softmax activation function.

    Computes the softmax probabilities during forward propagation and the
    corresponding Jacobian matrices during backpropagation.

    Usage:
        softmax = Softmax()
        A = softmax(z)             # forward: softmax across each column
        J = softmax(z, False)      # backward: Jacobian(s) dA/dz
    """

    def __call__(self, z, forw_prop=True):
        """Compute softmax activation or its Jacobian.

        Args:
            z (np.ndarray): Input array of shape (n, m), where each column
                is a set of pre-activation values for n classes and m examples.
            forw_prop (bool, optional): If True (default), returns the softmax
                outputs; if False, returns the Jacobian matrix for each example.

        Returns:
            np.ndarray:
                - If `forw_prop=True`: array of shape (n, m) with softmax
                  probabilities for each column.
                - If `forw_prop=False`:
                    - shape (n, n) if `z` is 1D (n,) (single example),
                    - shape (m, n, n) if `z` is 2D (n, m) (one Jacobian per column).
        """

        shift = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(shift)
        A = exp_z / np.sum(exp_z, axis=0, keepdims=True)
        if forw_prop:
            return A
        if z.ndim == 1:
            return np.diag(A) - np.outer(A, A)
        
        n, m = A.shape

        # Derivate
        jacobianas = np.zeros((m, n, n))
        for i in range(m):
            ai = A[:, i]
            jacobianas[i] = np.diag(ai) - np.outer(ai, ai)
        return jacobianas
    

