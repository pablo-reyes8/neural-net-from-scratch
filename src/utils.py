import numpy as np 
import math 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

def random_mini_batches(X, Y, mini_batch_size=64, seed=None):
    """Generate a list of random mini-batches from (X, Y).

    Args:
        X (np.ndarray): Input data of shape (n_x, m), where m is the number of examples.
        Y (np.ndarray): Labels of shape (n_y, m) or (m,); will be reshaped to (1, m) if needed.
        mini_batch_size (int, optional): Size of each mini-batch. Defaults to 64.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        list of tuples: A list where each element is a tuple
            `(mini_batch_X, mini_batch_Y)` with shapes
            `(n_x, mini_batch_size)` and `(n_y, mini_batch_size)` (except the last batch
            which may be smaller if m is not divisible by `mini_batch_size`).
    """

    m = X.shape[1]
    if Y.ndim == 1:
        Y = Y.reshape(1, m)

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    mini_batches = []
    num_complete = math.floor(m / mini_batch_size)
    for k in range(num_complete):
        start = k * mini_batch_size
        end   = start + mini_batch_size

        mini_batches.append((shuffled_X[:, start:end],shuffled_Y[:, start:end]))

    if m % mini_batch_size != 0:
        start = num_complete * mini_batch_size
        mini_batches.append((shuffled_X[:, start: ],shuffled_Y[:, start: ]))

    return mini_batches

def train_val_test_split(X, y, train_pct, val_pct, test_pct):
    """Split datasets into training, validation, and test sets by specified proportions.

    Converts inputs to NumPy arrays, ensures correct shapes, and splits along the
    first axis (samples). Supports 1D or row-vector `y`.

    Args:
        X (array-like): Features of shape (m, n) or (n, m), where m is number of samples.
        y (array-like): Labels of shape (m,) or (1, m) or (m, 1).
        train_pct (float): Relative proportion for the training set.
        val_pct (float): Relative proportion for the validation set.
        test_pct (float): Relative proportion for the test set.

    Returns:
        tuple:
            X_train (np.ndarray): Training features.
            X_val (np.ndarray): Validation features.
            X_test (np.ndarray): Test features.
            y_train (np.ndarray): Training labels.
            y_val (np.ndarray): Validation labels.
            y_test (np.ndarray): Test labels.

    Raises:
        ValueError: If `y` has more than 2 dimensions.
    """
    
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim == 2 and X.shape[0] < X.shape[1]:
        X = X.T

    if y.ndim == 2:
        if y.shape[0] == 1:
            y = y.flatten()
        elif y.shape[1] == 1:
            y = y.reshape(-1)
    elif y.ndim != 1:
        raise ValueError(f"y debe ser 1-D o (1, m); llegó con ndim={y.ndim}")
    
    total = train_pct + val_pct + test_pct
    train_prop = train_pct / total
    val_prop = val_pct / total

    n = X.shape[0]
    idx = np.arange(n)

    train_end = int(np.floor(train_prop * n))
    val_end = train_end + int(np.floor(val_prop * n))

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx= idx[val_end:]

    X_train= X[train_idx]
    X_val= X[val_idx]
    X_test = X[test_idx]
    y_train= y[train_idx]
    y_val= y[val_idx]
    y_test= y[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def label_encoder(df, column, encoder=None):
    
    """Encode categorical values in a DataFrame column as integer labels.

    Fits (or uses) a scikit-learn LabelEncoder to transform the specified column
    in-place, converting each unique category to an integer.

    Args:
        df (pd.DataFrame): DataFrame containing the column to encode.
        column (str): Name of the column with categorical values.
        encoder (LabelEncoder, optional): Pre-instantiated encoder to use.
            If None (default), a new LabelEncoder is created and fitted.

    Returns:
        LabelEncoder: The fitted encoder instance, useful for inverse transforms
        or applying the same mapping to other datasets.
    """

    if encoder is None:
        encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))
    return encoder

def manual_standardize(df, column):
    """Standardize a DataFrame column to zero mean and unit variance in-place.

    Computes the column’s mean and population standard deviation (ddof=0)
    then replaces values with their z-scores.

    Args:
        df (pd.DataFrame): DataFrame containing the target column.
        column (str): Name of the column to standardize.

    Returns:
        None: Modifies `df[column]` in place.
    """

    mean = df[column].mean()
    std  = df[column].std(ddof=0)  
    df[column] = (df[column] - mean) / std
    return 


def plot_error(cost, lr):
    """
    Plot the training cost over epochs with improved styling.

    Parameters
    ----------
    cost : array-like of shape (n_epochs,)
        Sequence of cost values recorded at each epoch (or aggregated per 100 epochs).
    lr : float
        Learning rate used during training.

    This function creates a line chart of cost vs. epoch number, 
    includes gridlines, highlights the minimum cost point, and 
    annotates it with the cost value for better interpretability.
    """

    cost = np.asarray(cost, dtype=float)
    epochs = np.arange(1, len(cost) + 1)
    

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, cost, linewidth=1.5)
    min_idx = cost.argmin() + 1
    min_cost = cost.min()
    ax.scatter(min_idx, min_cost)
    ax.annotate(
        f"min: {min_cost:.4f}",
        xy=(min_idx, min_cost),
        xytext=(min_idx, min_cost + (cost.max() - cost.min()) * 0.05),
        arrowprops=dict(arrowstyle='->', linewidth=0.8))
    
    ax.set_xlabel('Epoch (×100)')
    ax.set_ylabel('Cost')
    ax.set_title(f'Learning rate = {lr}')
    
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    
    plt.show()


def class_report_roc(y_test, y_pred):
    """
    Compute and display a classification report and ROC curve for binary classification.

    Parameters
    ----------
    y_test : array-like of shape (n_samples,) or (1, n_samples)
        True binary labels (0/1) or one-hot encoded labels.
    y_pred : array-like of shape (n_samples,) or (1, n_samples) or (2, n_samples)
        Predicted labels (0/1) when evaluating classification report, and
        predicted scores/probabilities for the positive class when plotting ROC.
        If passed as shape (2, n_samples), the second row is taken as the positive-class score.

    Returns
    -------
    None
        Prints the classification report and displays the ROC plot.
    """

    y_true = y_test.ravel()
    y_pred_flat = y_pred.ravel()

    report = classification_report(
        y_true,
        y_pred_flat,target_names=['Clase 0', 'Clase 1'],digits=3)
    
    print("Classification Report:\n", report)

    if y_pred.ndim == 2 and y_pred.shape[0] == 2:
        scores = y_pred[1, :].ravel()
    else:
        scores = y_pred_flat

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=1.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1, label='Baseline')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    ax.annotate(
        f"AUC = {roc_auc:.4f}",
        xy=(0.6, 0.1),
        xycoords='axes fraction',
        fontsize=10,
        arrowprops=dict(arrowstyle='->', linewidth=0.8))
    fig.tight_layout()
    plt.show()