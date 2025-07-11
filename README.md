# Neural Network From Scratch

## Overview  
This project provides a modular Python framework for building and training feed-forward neural networks entirely from scratch. You can experiment with binary classification or full multi-class setups (via softmax), customize layer sizes, activation functions, weight initializations (random or He), optimizers (vanilla gradient descent or Adam), learning-rate schedules (including decay), and L2 regularization. All core logic lives in standalone modules under `src/`, while Jupyter notebooks illustrate end-to-end usage and hyperparameter experiments.

## Key Findings  
- **Stable convergence:** Training curves show smooth cost decrease and consistent 90+ % accuracy on test splits.  
- **Hyperparameter flexibility:** A rich set of knobs (learning rate, batch size, decay, regularization strength) makes it easy to fine-tune performance.  
- **Binary & multi-class support:** Softmax output and cross-entropy loss handle >2 classes without extra code.  
- **Modular design:** Clear separation between activations, layers, optimizers, and training logic enables rapid prototyping and extension.

## Project Structure

| File / Notebook                                    | Purpose                                                                          |
|----------------------------------------------------|----------------------------------------------------------------------------------|
| **Data/Customer-Churn-Records.csv**                | Customer churn dataset for binary classification examples.                       |
| **Data/Diabetes_1.csv**                            | Diabetes dataset for un balanced classification scenarios.                       |
| **Data/schizophrenia_dataset.csv**                 | Schizophrenia dataset for clasification.                                         |
| **jupyter_notebooks/nn_scratch Full.ipynb**        | End-to-end demo: code walkthrough, training loop, and model evaluation. This Jupyter includes all the function that you can find in `src/` in one place.          |
| **jupyter_notebooks/training_experiments.ipynb**   | This jupyter uses `src/` functions to train and adjust Hyperparameter, learning-rate decay studies, and accuracy/cost plots on the diabetes and Schizophrenia dataset.     |
| **src/activations.py**                             | Definitions of ReLU, Sigmoid, Tanh, LeakyReLu, and Softmax activation functions.           |
| **src/initialization.py**                          | Weight initialization routines: random scaling and He normal.                    |
| **src/layers.py**                                  | Layer operations: `one_layer_forward`, `forward_pass`, and `back_propagation`.   |
| **src/optimizers.py**                              | Optimizers and schedules: `update_gd_parameters`, `adam`, and `learning_decay`.  |
| **src/training.py**                                | Training interface: `model_nn_scratch`, cost functions, and `predict`.           |
| **src/utils.py**                                   | Preprocessing helpers: mini-batch generator, train/val/test split, standardization, label encoding and functions for ploting the error and ROC curve. |




## Collaborate  
Contributions, issue reports, and pull requests are very welcome! Feel free to open an issue if you’d like to suggest new features (dropout, regression outputs, alternative optimizers) or improvements to the existing code.

## License  
This project is released under the MIT License.  

