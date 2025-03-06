import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pygam import GAM, s
import json
from shadho import Shadho, spaces
from pyrameter.methods import random, hom

def branin(params):
    x1, x2 = params['x1'], params['x2']
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * ((x2 - b * x1**2 + c * x1 - r) ** 2) + s * (1 - t) * np.cos(x1) + s

def visualize_gam(gam, X, y):
    """ Visualize the smoothing functions for each predictor in the GAM along with the Branin function """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['Effect of x1 on the Branin function', 'Effect of x2 on the Branin function', 'Optimization Landscape']
    min_x1, min_x2 = X[np.argmin(y), :]
    global_min_x1, global_min_x2 = [-np.pi, 12.275], [np.pi, 2.275]  # Known global minima locations for the Branin function

    # Plot GAM partial dependencies
    for i, ax in enumerate(axs[:-1]):
        XX = gam.generate_X_grid(term=i)
        pd, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        ax.plot(XX[:, i], pd, label='Partial Dependence')
        ax.fill_between(XX[:, i], confi[:, 0], confi[:, 1], color='r', alpha=0.1, label='95% CI')
        ax.set_title(titles[i])
        ax.set_xlabel(f'x{i+1}')
        ax.set_ylabel('Effect on output')
        ax.legend()

    # Plot the actual Branin function with optimization points
    x1 = np.linspace(-5, 10, 400)
    x2 = np.linspace(0, 15, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = branin({'x1': X1, 'x2': X2})
    cp = axs[2].contourf(X1, X2, Z, levels=50, cmap=cm.viridis)
    fig.colorbar(cp, ax=axs[2], label='Branin function value')
    axs[2].scatter(X[:, 0], X[:, 1], color='white', s=50, zorder=5, edgecolor='black', label='Optimization Points')
    axs[2].scatter(min_x1, min_x2, color='red', s=100, zorder=5, edgecolor='black', label='Best Point')
    # axs[2].scatter(global_min_x1, global_min_x2, color='gold', s=100, zorder=5, edgecolor='black', label='Global Minima')
    axs[2].set_xlabel('x1')
    axs[2].set_ylabel('x2')
    axs[2].set_title(titles[2])
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def run_optimization():
    # Setup your space and optimization as usual
    space = {'x1': spaces.uniform(-5, 10), 'x2': spaces.uniform(0, 15)}

    # Initialize your optimizer here with Shadho
    # For demonstration, using a simplified setup
    # Assume 'method' is some method initialized as per your setup
    opt = Shadho('branin', branin, space, method=hom(inner_method=random()), timeout=300, max_tasks=100)

    # Run optimization
    opt.run()

# Assuming the path to the results.json is correctly set
def load_results(results_path):
    with open(results_path, 'r') as file:
        data = json.load(file)

    # Extract parameters and objective values from trials
    hyperparameters = []
    objectives = []
    for trial in data[0]['trials']:  # Assuming the structure you provided
        if trial['status'] == 4:  # Status 4 might indicate completed trials
            params = trial['hyperparameters']
            obj = trial['objective']
            hyperparameters.append(params)
            objectives.append(obj)

    return np.array(hyperparameters), np.array(objectives)

if __name__ == '__main__':
    # Run the optimization and save the results
    run_optimization()
    
    # Load results from the JSON file
    X, y = load_results('results.json')
    
    # Fit a GAM to the results
    gam = GAM(s(0) + s(1), n_splines=10).fit(X, y)
    
    # Visualize the effects of each predictor
    visualize_gam(gam, X, y)
