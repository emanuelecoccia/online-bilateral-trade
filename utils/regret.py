import numpy as np
import pandas as pd
from environments.base import BaseEnvironment
from environments.contextual import ContextualEnvironment
from learners.experts import BaseAlgorithm
from .valuations import construct_sequence_with_lipschitz_valuations
from typing import Tuple, List
from tqdm import tqdm

def compute_scaling_laws(
        Algorithm:BaseAlgorithm, Environment:BaseEnvironment, 
        T_horizons:list[int], *args, **kwargs
        )->Tuple[list[int], list[float]]:
    """
    This function computes the regret of an algorithm given an environment,
    at different time horizons T.
    """
    regret_values = []
    for T in T_horizons:
        # Create environment based on variables found
        environment = Environment(T)

        # Create algorithm
        algorithm = Algorithm(T, environment)
        algorithm.run()
        algo_gft = algorithm.get_final_gft()

        # Calculate the regret based on the settings
        _, max_gft = environment.get_best_expert()
        regret = max_gft - algo_gft
        regret_values.append(regret)

    return T_horizons, regret_values

def compute_scaling_laws_with_policy_regret(
        Algorithm:BaseAlgorithm, Environment:ContextualEnvironment, 
        T_horizons:list[int], Lipschitz_constants:list[float], sequence_constructor=None,
        adhoc_valuations:bool=True, *args, **kwargs
        )->Tuple[list[int], list[int], list[list[float]]]:
    """
    This function computes the regret of an algorithm given an environment,
    at different time horizons T and Lipschitz constants L.
    We presume that the valuations are built ad-hoc, so that for each context
    there is only one pair of valuations and the valuation sequence is Lipschitz just like the policy.
    In the opposite case, for one context there are multiple pairs of valuations
    and the optimal policy is able to choose the best one.
    """
    regret_values = []
    for T in T_horizons:
        regret_values_for_T = []
        for L in Lipschitz_constants:
            # Create environment based on variables found
            if sequence_constructor:
                # Construct the sequence of valuations from the given constructor
                contexts, valuations = sequence_constructor(T, L)
                environment = Environment(T, contexts, valuations)
            else:
                environment = Environment(T)

            # Create algorithm
            algorithm = Algorithm(T, environment, L)
            algorithm.run()
            algo_gft = algorithm.get_final_gft()

            # Calculate the regret based on the settings
            if adhoc_valuations:
                max_gft = environment.get_policy_gft_having_adhoc_valuations()
            else:
                _, max_gft = environment.get_policy_gft() # much slower method, based on a sweeping algorithm
            regret = max_gft - algo_gft
            regret_values_for_T.append(regret)
        
        regret_values.append(regret_values_for_T)

    return T_horizons, Lipschitz_constants, regret_values