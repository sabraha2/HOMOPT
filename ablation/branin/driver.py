import numpy as np
import argparse
from shadho import Shadho, spaces
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from pyrameter.methods import random, tpe, bayes, pso, smac, hom
from pygam import GAM

METHODS = {
    'bayes': bayes,
    'hom': hom,
    'pso': pso,
    'random': random,
    'smac': smac,
    'tpe': tpe
}

class HOM(hom):
    """Extended Homotopy Optimization Method to include GAM parameters"""
    def __init__(self, inner_method, k=0.5, iterations=5, jitter_strength=0.005, warm_up=20, n_splines=25, lam=1e-4):
        super().__init__(inner_method)
        self.warm_up = warm_up
        self.k = k
        self.iterations = iterations
        self.jitter_strength = jitter_strength
        self.n_splines = n_splines
        self.lam = lam

    def generate(self, trial_data, domains):
        n_trials = trial_data.shape[0]
        if n_trials < self.warm_up or n_trials % 5 in [0, 2]:
            return self.inner_method.generate(trial_data, domains)
        elif n_trials % 5 in [3, 4]:
            best_features = trial_data[np.argmin(trial_data[:, -1]), :-1]
            scaled_variance = np.var(trial_data[:, :-1], axis=0) * self.jitter_strength
            return best_features + np.random.uniform(-scaled_variance, scaled_variance)
        else:
            features, losses = trial_data[:, :-1], trial_data[:, -1]
            idx = np.argmin(losses)
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            gam1 = GAM(lam=self.lam, n_splines=self.n_splines)
            gam1.fit(features, losses)

            opt_vars = features[idx]
            bounds = [domain.bounds for domain in domains]

            for i in range(self.iterations):
                res = minimize(
                    lambda params: gam1.predict(params.reshape(1, -1)),
                    opt_vars,
                    method='L-BFGS-B',
                    bounds=bounds
                )
                opt_vars = res.x

            return scaler.inverse_transform(opt_vars.reshape(1, -1))[0]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimization of the Branin function with adjustable HOMOPT and GAM parameters.")
    parser.add_argument('--inner_method', choices=METHODS.keys(), help='The inner method to optimize.')
    parser.add_argument('--warm_up', type=int, help='The number of warm-up iterations.')
    parser.add_argument('--k', type=float, help='The fraction of data used for the inner method.')
    parser.add_argument('--jitter_strength', type=float, help='The jitter strength for perturbation.')
    parser.add_argument('--max_iter', type=int, help='The number of maximum iterations.')
    parser.add_argument('--n_splines', type=int, default=25, help='Number of splines for GAM.')
    parser.add_argument('--lambda_reg', type=float, default=1e-4, help='Regularization parameter for GAM.')
    return parser.parse_args()

def branin(params):
    x1, x2 = params['x1'], params['x2']
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi**2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return a * ((x2 - b * x1**2 + c * x1 - r) ** 2) + s * (1 - t) * np.cos(x1) + s

if __name__ == '__main__':
    args = parse_args()
    space = {'x1': spaces.uniform(-5, 10), 'x2': spaces.uniform(0, 15)}
    
    method = HOM(
        inner_method=METHODS[args.inner_method](),
        k=args.k,
        jitter_strength=args.jitter_strength,
        iterations=args.max_iter,
        warm_up=args.warm_up,
        n_splines=args.n_splines,
        lam=args.lambda_reg
    )
    
    opt = Shadho('branin', branin, space, method=method, timeout=-1, max_tasks=100, await_pending=False)
    opt.run()
