# HomOpt: A Flexible Homotopy-Based
Hyperparameter Optimization Method

## Overview
This README accompanies the our paper titled _HomOpt: A Flexible Homotopy-Based
Hyperparameter Optimization Method_, providing a detailed guide on setting up and running the experiments discussed in the paper. It includes code examples, particularly focusing on the use of a Generalized Additive Model (GAM) based Homotopy Optimization Method (HomOpt) within the SHADHO framework for hyperparameter optimization.

`HomOpt` is a robust homotopy-based hyperparameter optimization method we have integrated within the SHADHO framework. Currently, HomOpt in SHADHO exclusively supports the Generalized Additive Model (GAM) as a surrogate model. This method efficiently navigates the complex parameter space of models by utilizing continuous deformation between successive surrogate models, enhancing the search for optimal hyperparameters.

## Prerequisites
To use `HomOpt` with SHADHO, ensure the following packages are installed:
- Python 3.6+
- SHADHO
- PyGAM (for GAM surrogate modeling)
- Scipy (for optimization routines)
- NumPy
- Scikit-learn (for data preprocessing and SVM example)

## Installation
Ensure Python and pip are available on your system. Install the required libraries using pip:
```bash
pip install shadho pygam scipy numpy scikit-learn
python -m shadho.installers.workqueue
```

## Setting Up the Optimization Task

### Define the Objective Function and Search Space

Create a Python script, for example `example/svm_optimization_with_homopt.py`, where you define both the SVM model's search space for hyperparameters and the objective function using GAM as the surrogate model.

Here's a basic setup involving an SVM classifier:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from shadho import Shadho, spaces
from pygam import GAM, s, l
from scipy.optimize import minimize
from scipy.stats import uniform

def objective_function(params):
    # Load the digits dataset from sklearn
    X, y = load_digits(return_X_y=True)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Create and train the SVM model with parameters from SHADHO
    model = SVC(C=params['C'], kernel='rbf', gamma=params['gamma'])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, predictions)
    
    # SHADHO minimizes the objective, so return negative accuracy as 'loss'
    return {'loss': -accuracy}

```

### Configure and Run SHADHO with HomOpt

The HomOpt method is used to optimize the SVM by navigating the hyperparameter space using the surrogate models created by GAM. Ensure to initialize the HomOpt method with appropriate settings for your optimization task.

```python
def configure_shadho():
    # Setup the search space
    search_space = setup_search_space()
    
    # Configure the HomOpt method
    inner_method = random()
    hom_method = HOM(inner_method=inner_method, k=0.5, iterations=5,
                     jitter_strength=0.005, warm_up=20)
    
    # Set the HomOpt method in SHADHO
    shadho = Shadho('svm', objective_function, search_space, method=hom_method, timeout=-1, max_tasks=100, await_pending=False)
    
    return shadho
```
## Execution
Run the optimization by executing your Python script:

```python
python svm_optimization_with_homopt.py
```
This command starts the SHADHO framework and employs the `HomOpt` method, iterating through the hyperparameter space defined in the setup_search_space() method using a GAM surrogate to optimize the hyperparameters of an SVM model.

## Limitations and Future Work
Currently, the `HomOpt` method in SHADHO supports only the GAM surrogate model. Plans to expand the range of surrogate models include integrating polynomial regression models and other ensemble methods, which would provide a more robust framework capable of handling a broader range of optimization tasks.

The `HomOpt` method within SHADHO offers a sophisticated approach to hyperparameter optimization by integrating a GAM surrogate model. While currently limited to GAMs, the method's framework is designed for expansion and increased versatility in future releases. 

