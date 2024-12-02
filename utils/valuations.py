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

    s, b = sinusoidal_function(s_dot, b_dot, Lipschitz_constant)
    contexts = np.stack([s_dot, b_dot], axis=1)
    valuations = np.stack([s, b], axis=1)
    return contexts, valuations

def sinusoidal_function(s_dot:Union[float, np.ndarray], b_dot:Union[float, np.ndarray], Lipschitz_constant:float)\
    ->Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    This function generates a wobbly function that is Lipschitz with respect to the context.
    It returns a pair of private valuations.
    """
    sinusoidal_residual = (np.sin(s_dot*Lipschitz_constant) * np.cos(b_dot*Lipschitz_constant))/2 + 0.5
    s = s_dot + sinusoidal_residual * (b_dot - s_dot) * 0.5
    b = s + (b_dot - s_dot) * 0.5
    return s, b

def construct_logarithmic_lower_bound(T:int, Lipschitz_constant:float)\
    ->Tuple[np.ndarray, np.ndarray]:
    """
    This function constructs a sequence of valuations and a sequence of contexts,
    such that the valuations are Lipschitz continuous with respect to the contexts.
    """
    # The contexts are:
    # (0, 1), 
    # (0, 0.5), (0.5, 1), 
    # (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1),
    # ...
    # However if the Lipschitz constant allows it,
    # we can create M contexts equally spaced between parent and child.
    # Assuming the distance between two points is the l2 norm:
    if Lipschitz_constant < 2*np.sqrt(2):
        raise ValueError("The Lipschitz constant is too small to construct the sequence.")
    M = int(Lipschitz_constant/(2*np.sqrt(2)))

    # Initialize two lists of contexts:
    # One is for rescaling the valuations, the other is for the actual contexts
    contexts_for_rescaling_vals = [np.array([0, 1])]
    actual_contexts = [np.array([0, 1])]

    # Construct the contexts
    counter = 1
    i = 1
    while counter < T:
        j = 0
        while j < 2**i and counter < T:
            # Append the fake contexts
            m = 0
            secondary_counter = counter
            while m < M and secondary_counter < T:
                contexts_for_rescaling_vals.append(np.array([j/(2**i), (j+1)/(2**i)]))
                m += 1
                secondary_counter += 1
            # Append the actual contexts
            if j%2==0:
                m = 0
                while m < M and counter < T:
                    x = j/(2**i)
                    y = (j+1)/(2**i) + (m/M)/(2**i)
                    actual_contexts.append(np.array([x, y]))
                    m += 1
                    counter += 1
            else:
                m = 0
                while m < M and counter < T:
                    x = j/(2**i) - (m/M)/(2**i)
                    y = (j+1)/(2**i)
                    actual_contexts.append(np.array([x, y]))
                    m += 1
                    counter += 1
            j += 1
        i += 1

    # Convert the lists to numpy arrays
    contexts_for_rescaling_vals = np.array(contexts_for_rescaling_vals)
    actual_contexts = np.array(actual_contexts)

    # The valuations are [0, 1/4] and [3/4, 1] at random, and rescaled to the "fake" contexts
    left_point = np.array([0, 1/4])
    right_point = np.array([3/4, 1])
    points = np.vstack([left_point, right_point])
    valuation_sequence_indices = np.random.choice(2, size=T)
    valuation_sequence = points[valuation_sequence_indices]

    # Rescale the valuation sequence to be inside the order book
    scaling_factors = (contexts_for_rescaling_vals[:, 1] - contexts_for_rescaling_vals[:, 0])#**2
    rescaled_s = contexts_for_rescaling_vals[:, 0] + valuation_sequence[:, 0] * scaling_factors
    rescaled_b = contexts_for_rescaling_vals[:, 1] - (1 - valuation_sequence[:, 1]) * scaling_factors
    valuation_sequence = np.vstack([rescaled_s, rescaled_b]).T

    return actual_contexts, valuation_sequence