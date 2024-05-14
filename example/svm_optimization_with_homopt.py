import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from shadho import Shadho, spaces
from pyrameter.methods import random
from pyrameter.methods.hom import HOM

def objective_function(params):
    """
    Define the objective function that trains an SVM model and evaluates its accuracy.

    Parameters:
    - params (dict): Dictionary containing the hyperparameters 'C' and 'gamma'.

    Returns:
    - dict: A dictionary with the keys 'loss' (negative accuracy).
    """
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

def setup_search_space():
    """
    Defines the search space for the SVM's hyperparameters.

    Returns:
    - spaces (dict): A dictionary with the search space for 'C' and 'gamma'.
    """
    # Define the search space for C and gamma using loguniform distribution
    search_space = {
        'C': spaces.log2_uniform(-5, 15),
        'gamma': spaces.log10_uniform(-3, 3)
    }
    return search_space

def configure_shadho():
    """
    Configures and returns a SHADHO optimizer with the HomOpt method.

    Returns:
    - shadho (Shadho object): Configured SHADHO optimizer.
    """
    # Setup the search space
    search_space = setup_search_space()
    
    # Configure the HomOpt method
    inner_method = random()
    hom_method = HOM(inner_method=inner_method, k=0.5, iterations=5,
                     jitter_strength=0.005, warm_up=20)
    
    # Set the HomOpt method in SHADHO
    shadho = Shadho('svm', objective_function, search_space, method=hom_method, timeout=-1, max_tasks=100, await_pending=False)
    
    return shadho

def main():
    """
    Main function to run the SHADHO optimizer.
    """
    # Configure SHADHO
    shadho = configure_shadho()
    
    # Run SHADHO optimization for an hour
    shadho.run()  # Run for 1 hour

if __name__ == '__main__':
    main()
