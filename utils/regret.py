import numpy as np
import pandas as pd

def compute_scaling_laws(Algorithm, Environment, *args, **kwargs):
    """
    This function computes the regret of an algorithm given an environment,
    at different time horizons T.
    """
    T_values = np.linspace(20000, 400000, 20, dtype=int, endpoint=True)
    regret_values = []
    for T in T_values:
        environment = Environment(T)
        algorithm = Algorithm(T, environment)
        algorithm.run()
        algo_gft = algorithm.get_final_gft()
        if kwargs.get("policy_regret", False):
            _, max_gft = environment.get_policy_gft()
        else:
            _, max_gft = environment.get_best_expert()
        regret = max_gft - algo_gft
        regret_values.append(regret)
    return T_values, regret_values