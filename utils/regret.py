import numpy as np
import pandas as pd
from environments.base import DummyEnvironment
from learners.experts import GFTMax, DummyAlgorithm
from .valuations import construct_sequence_with_lipschitz_valuations
from typing import Tuple, List
from tqdm import tqdm

def compute_scaling_laws(Algorithm:DummyAlgorithm, Environment:DummyEnvironment, *args, **kwargs)\
    ->Tuple[np.ndarray, List[float]]:
    """
    This function computes the regret of an algorithm given an environment,
    at different time horizons T.
    Possible kwargs: policy_regret, adhoc_valuations.
    """
    T_values = np.linspace(10000, 100000, 10, dtype=int, endpoint=True)
    L_values = [100] #np.logspace(1, 10, 10, dtype=int, endpoint=True)
    regret_values = []
    for T in T_values:
        for L in tqdm(L_values):
            # Create environment based on variables found
            if kwargs.get("policy_regret", False) and kwargs.get("adhoc_valuations", False):
                contexts, valuations = construct_sequence_with_lipschitz_valuations(T, L)
                environment = Environment(T, contexts, valuations)
            else:
                environment = Environment(T)

            # Create algorithm
            if kwargs.get("policy_regret", False) and kwargs.get("adhoc_valuations", False):
                algorithm = Algorithm(T, environment, L)
            else:
                algorithm = Algorithm(T, environment)
            algorithm.run()
            algo_gft = algorithm.get_final_gft()

            # Calculate the regret based on the settings
            if kwargs.get("policy_regret", False):
                if kwargs.get("adhoc_valuations", False):
                    max_gft = environment.get_policy_gft_having_adhoc_valuations() # faster computation
                else:
                    _, max_gft = environment.get_policy_gft()
            else:
                _, max_gft = environment.get_best_expert()
            regret = max_gft - algo_gft
            regret_values.append(regret)

    return T_values, L_values, regret_values