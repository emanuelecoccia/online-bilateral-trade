import numpy as np
from typing import Union, Tuple

def construct_sequence_with_lipschitz_valuations(T:int, Lipschitz_constant:float)\
    ->Tuple[np.ndarray, np.ndarray]:
    """
    This function constructs a sequence of valuations and a sequence of contexts,
    such that the valuations are Lipschitz continuous with respect to the contexts.
    """
    s_dot = np.random.random(size=T)
    b_dot = np.random.random(size=T)
    s = np.zeros(T)
    b = np.zeros(T)
    for t in range(T):
        if s_dot[t] > b_dot[t]:
            s_dot[t], b_dot[t] = b_dot[t], s_dot[t]

    s, b = wobbly_function(s_dot, b_dot, Lipschitz_constant)
    contexts = np.stack([s_dot, b_dot], axis=1)
    valuations = np.stack([s, b], axis=1)
    return contexts, valuations

def wobbly_function(s_dot:Union[float, np.ndarray], b_dot:Union[float, np.ndarray], Lipschitz_constant:float)\
    ->Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    This function generates a wobbly function that is Lipschitz with respect to the context.
    It returns a pair of private valuations.
    """
    sinusoidal_residual = (np.sin(s_dot*Lipschitz_constant) * np.cos(b_dot*Lipschitz_constant))/2 + 0.5
    s = s_dot + sinusoidal_residual * (b_dot - s_dot) * 0.5
    b = s + (b_dot - s_dot) * 0.5
    return s, b