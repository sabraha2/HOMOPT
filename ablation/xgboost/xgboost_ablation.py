import numpy as np

import math
from shadho import Shadho, spaces
import sys 
import argparse
import numpy as np
from pyrameter.methods import random, tpe, bayes, pso, smac, hom
from shadho.benchmarks import convert_config_to_shadho, run_benchmark


from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from hpobench.benchmarks.ml.rf_benchmark import RandomForestBenchmark
from hpobench.benchmarks.ml.lr_benchmark import LRBenchmark
from hpobench.benchmarks.ml.svm_benchmark import SVMBenchmark
import functools

METHODS = {
    'bayes': bayes,
    'hom': hom,
    'pso': pso,
    'random': random,
    'smac': smac,
    'tpe': tpe
}

BENCHMARKS = {
    'XGBoostBenchmark': XGBoostBenchmark, 
    'NNBenchmark': NNBenchmark, 
    'RandomForestBenchmark': RandomForestBenchmark, 
    'LRBenchmark': LRBenchmark, 
    'SVMBenchmark': SVMBenchmark
}


def parse_args(args=None):
    """Parse inner_method, warm_up, k, jitter_strength, and max_iter from command line arguments."""
    p = argparse.ArgumentParser(description=sys.modules[__name__].__doc__)
    p.add_argument('--inner_method', type=str,
        choices=['bayes', 'ncqs', 'hom', 'pso', 'random', 'smac', 'tpe'],
        help='The inner method to optimize.')
    p.add_argument('--warm_up', type=int, 
        help='The number of warm-up iterations.')
    p.add_argument('--k', type=float, 
        help='The number of k iterations.')
    p.add_argument('--jitter_strength', type=float, 
        help='The jitter strength.')
    p.add_argument('--max_iter', type=int, 
        help='The number of max iterations.')
    return p.parse_args(args)


def branin(params):
    x1 = params['x1']
    x2 = params['x2']
    a = 1
    b = 5.1/(4*np.pi**2)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s

if __name__ == '__main__':

    b = BENCHMARKS['XGBoostBenchmark'](task_id=31, rng=7)
    obj = functools.partial(run_benchmark, b)
    config = b.get_configuration_space(seed=7)
    space = convert_config_to_shadho(config)

    args = parse_args()

    opt = Shadho(
                'XGBoost', 
                obj, 
                space, 
                method=hom(
                    inner_method=METHODS[args.inner_method](), 
                    warm_up=args.warm_up,
                    k=args.k,
                    jitter_strength=args.jitter_strength,
                    iterations = args.max_iter
                ), 
                timeout = -1,
                max_tasks=100, 
                await_pending=False)

    opt.run()
